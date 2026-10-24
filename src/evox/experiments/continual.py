"""Causal continual replay shared by the Phase C comparative policies."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

from evox.adaptation.data import AdaptationData, LabelPolicy, ReplayBuffer, Samples, balanced_subset, priority
from evox.adaptation.strategies import Action, AdaptationEngine, StrategyConfig
from evox.drift.analyzer import DriftAnalyzer
from evox.drift.scenarios import ScenarioSpec, generate_scenario
from evox.evaluation.candidate import EvaluationSets, classification_metrics, counterfactual_replay, model_metrics
from evox.evaluation.drift import MemberConfig, compute_signals, fused_alarms, run_members, static_run
from evox.evolution.controller import ActionEstimate, ControllerThresholds, EvolutionController, UtilityWeights
from evox.evolution.proxy import ProxyEvaluator, ProxyHoldouts
from evox.evolution.safety import SafetyGate
from evox.experiments.evolve import budget_detector_errors, sample_set, with_frozen_costs
from evox.lifecycle.manager import EvolutionLifecycle, MonitorConfig, RollbackMonitor
from evox.lifecycle.registry import FileModelRegistry
from evox.models.ids import IDSModel, UnsupportedUpdate

SYSTEMS = ("static", "always_retrain", "always_incremental", "always_replay", "ssf_inspired",
           "active_incremental", "evox", "evox_no_gate", "evox_rules_only", "evox_utility_only")
FIXED_ACTIONS = dict(always_retrain=Action.RETRAIN, always_incremental=Action.INCREMENTAL,
                    always_replay=Action.REPLAY, ssf_inspired=Action.STRATEGIC, active_incremental=Action.INCREMENTAL)


@dataclass(frozen=True)
class ReplayConfig:
    model: str = "random_forest"
    trees: int = 60
    reference_cap: int = 6000
    holdout_cap: int = 512
    recent_batches: int = 12
    analysis_wait: int = 8
    mmd_samples: int = 200
    mmd_null: int = 40
    counterfactual_batches: int = 10
    training_budget: float = 60.0
    harmful_tolerance: float = 0.03
    recovery_tolerance: float = 0.02


def grouped_roles(source_ids):
    """50/15/15/10/10 percent source groups: train/proxy/gate/monitor/report."""
    values = [priority(str(s), 173) % 20 for s in source_ids]
    return np.array([0 if x < 10 else 1 if x < 13 else 2 if x < 16 else 3 if x < 18 else 4
                     for x in values], dtype=int)


def historical_roles(samples):
    """Stratify by class so each independent historical role retains rare classes."""
    roles = np.empty(len(samples), int)
    for c in np.unique(samples.y):
        idx = sorted(np.flatnonzero(samples.y == c), key=lambda i: priority(samples.ids[i], 174))
        for j, i in enumerate(idx):
            roles[i] = 1 + j % 4
    return roles


@dataclass
class ScenarioContext:
    scenario: object
    initial: IDSModel
    reference: Samples
    samples: Samples
    roles: np.ndarray
    historical: dict[int, Samples]
    signatures: list
    config: ReplayConfig
    policy: LabelPolicy

    def training_data(self, now, parent_version, strategy):
        s, bs = self.samples, self.scenario.spec.batch_size
        # Contiguous batch construction makes slicing cheaper than rehashing the
        # full future stream on every decision.
        end = max(0, min(len(s), (now - self.policy.delay_batches + 1) * bs))
        released = self.policy.available(s.take(np.arange(end)), now)
        indices = np.array([int(x.rsplit(":", 1)[1]) for x in released.ids], int)
        roles = self.roles[indices]
        train = released.take(roles == 0)
        memory = ReplayBuffer(strategy.memory_capacity, s.X.shape[1], self.scenario.spec.seed)
        memory.add(self.reference, self.policy, now)
        memory.add(train, self.policy, now)
        window = released.batches >= max(0, now - self.policy.delay_batches - self.config.recent_batches + 1)
        recent = released.take(window & (roles == 0))
        subsets = {r: balanced_subset(released.take(window & (roles == r)), self.config.holdout_cap, r)
                   for r in (1, 2, 3, 4)}
        excluded = tuple(x for r in (1, 2, 3, 4) for part in (subsets[r], self.historical[r]) for x in part.ids)
        return AdaptationData(recent, memory.samples, now, parent_version, excluded), subsets


def prepare_context(source, spec: ScenarioSpec, config: ReplayConfig, detector, policy):
    sc = generate_scenario(source, spec)
    s = sc.stream
    ref = balanced_subset(sample_set(s.X_ref_train, s.y_ref_train, spec.name + ":reference"),
                          config.reference_cap, spec.seed)
    initial = IDSModel(config.model, spec.seed, dict(n_estimators=config.trees, n_jobs=1)).fit(ref.X, ref.y)
    samples = sample_set(s.X_stream, s.y_stream, spec.name + ":flow",
                         np.arange(len(s.y_stream)) // spec.batch_size)
    hist = sample_set(s.X_ref_val, s.y_ref_val, spec.name + ":historical")
    hr = historical_roles(hist)
    historical = {r: balanced_subset(hist.take(hr == r), config.holdout_cap, r) for r in (1, 2, 3, 4)}
    if any(not len(x) for x in historical.values()):
        raise ValueError("Insufficient historical holdout support for four independent roles")
    run = static_run(sc, initial)
    sig = compute_signals(sc, run, seed=spec.seed, mmd_n_sub=config.mmd_samples, mmd_null=config.mmd_null)
    sig = budget_detector_errors(sig, samples, policy)
    cfg = MemberConfig(**{k: v for d in detector["members"].values() for k, v in d.items()})
    alarms = fused_alarms(run_members(sig, cfg, label_delay=policy.delay_batches),
                          detector["fusion"]["weights"], detector["fusion"]["theta"]).alarms
    analyzer = DriftAnalyzer(sig.ref_psi_detector, cfg.psi_threshold, cfg.mmd_mult * sig.mmd_null_q99,
                             ref_val_conf=initial.predict_with_confidence(s.X_ref_val)[1])
    Xb = s.X_stream.reshape(spec.n_batches, spec.batch_size, -1)
    signatures = [analyzer.characterize(a, min(a + config.analysis_wait, spec.n_batches - 1),
                                        sig.psi_stat, sig.mmd_stat, Xb, run.conf_batches) for a in alarms]
    return ScenarioContext(sc, initial, ref, samples, grouped_roles(sc.source_ids), historical,
                           signatures, config, policy)


def audit_delta(candidate, parent, recent, historical):
    n, p = model_metrics(candidate.model, recent), model_metrics(parent, recent)
    h, ph = model_metrics(candidate.model, historical), model_metrics(parent, historical)
    return dict(delta_f1_new=n["macro_f1"] - p["macro_f1"],
                delta_f1_historical=h["macro_f1"] - ph["macro_f1"])


def pilot_candidates(context, strategy):
    """Initial-parent candidates on VALIDATION ONLY for threshold selection."""
    from evox.config import TEST_SEEDS, VALIDATION_SEEDS
    from evox.evaluation.candidate import CandidateEvaluator

    spec = context.scenario.spec
    if spec.family != "validation" or spec.seed not in VALIDATION_SEEDS or spec.seed in TEST_SEEDS:
        raise ValueError("Gate pilots require validation scenarios/seeds")
    engine, evaluator, records = AdaptationEngine(strategy), CandidateEvaluator(), []
    for sig in context.signatures:
        data, heldout = context.training_data(sig.now_batch, "v1", strategy)
        if not len(heldout[2]) or not len(heldout[4]):
            continue
        for action in Action:
            if action == Action.NO_OP:
                continue
            try:
                c = engine.apply(action, context.initial, data, context.policy)
                evaluation = evaluator.evaluate(c, context.initial, EvaluationSets(heldout[2], context.historical[2]),
                                                 context.policy, sig.now_batch, forbidden_ids=context.reference.ids)
            except (ValueError, UnsupportedUpdate):
                continue
            records.append(dict(family=spec.family, seed=spec.seed, scenario=spec.name, action=action.value,
                                evaluation=evaluation, audit=audit_delta(c, context.initial, heldout[4], context.historical[4])))
    return records


def recovery_metrics(batch_rows, truth, tolerance):
    """Recovery is two consecutive attack-supported batches within the pre-drift F1 tolerance."""
    recovered, censored = [], 0
    scored = sorted([e for e in truth["events"] if e["scored"]], key=lambda e: e["start_batch"])
    for i, event in enumerate(scored):
        onset = event["start_batch"]
        pre = [r["binary_f1"] for r in batch_rows[max(0, onset - 5):onset] if r["n_attack"]]
        end = scored[i + 1]["start_batch"] if i + 1 < len(scored) else len(batch_rows)
        if not pre:
            censored += 1; continue
        target, run, found = float(np.mean(pre)) - tolerance, 0, None
        for b in range(onset, end):
            row = batch_rows[b]
            run = run + 1 if row["n_attack"] and row["binary_f1"] >= target else 0
            if run == 2:
                found = b - 1 - onset; break
        if found is None: censored += 1
        else: recovered.append(found)
    return dict(recovery_batches_mean=float(np.mean(recovered)) if recovered else None,
                recovery_events_observed=len(recovered), recovery_events_censored=censored)


def simulate(context: ScenarioContext, system: str, output_dir: Path, gate: SafetyGate, frozen: dict,
             strategy: StrategyConfig, proxy_config: StrategyConfig, data_tag: str):
    if system not in SYSTEMS:
        raise ValueError("Unknown comparative policy")
    sc, cfg, policy = context.scenario, context.config, context.policy
    metadata = dict(scenario=sc.spec.name, system=system, family=sc.spec.family, data_tag=data_tag,
                    algorithm=cfg.model, seed=sc.spec.seed, magnitude=sc.spec.magnitude, kind=sc.spec.kind)
    registry = FileModelRegistry(output_dir)
    try:
        registry.initialize(context.initial, context.reference.ids, metadata, context.reference.snapshot())
        life = EvolutionLifecycle(registry, gate)
        initial_gate = model_metrics(context.initial, context.historical[2])
        gate_best = {c: m["f1"] for c, m in initial_gate["per_class"].items()}
        initial_report = model_metrics(context.initial, context.historical[4])
        report_best = {c: m["f1"] for c, m in initial_report["per_class"].items()}
        controller = EvolutionController(UtilityWeights(**frozen["weights"]),
                                         ControllerThresholds(**frozen["thresholds"]),
                                         frozen["cost_scale"], frozen["latency_scale"],
                                         mode={"evox_rules_only": "rules_only", "evox_utility_only": "utility_only"}.get(system, "rules_utility"))
        engine, proxy = AdaptationEngine(strategy), ProxyEvaluator(proxy_config)
        scheduled = {}
        for sig in context.signatures:
            scheduled.setdefault(sig.now_batch, []).append(sig)
        batches, decisions, probes, yp_all, versions = [], [], [], [], []
        monitor, seen_classes, forward = None, set(context.initial.known_), []
        adaptations_seconds, cpu_seconds, proxy_seconds, proxy_cpu, peak_python = 0., 0., 0., 0., 0
        peak_rss = psutil.Process().memory_info().rss
        max_model = (registry.root / registry.versions["v1"]["artifact"]).stat().st_size
        unsupported, attempts, rollbacks, promotions, rejected, no_ops = 0, 0, 0, 0, 0, 0
        inference_times = []
        bs = sc.spec.batch_size
        for batch in sc.stream.iter_batches():
            current, version = life.holder.get()
            started = time.perf_counter(); pred = current.predict(batch.X)
            inference_times.append((time.perf_counter() - started) * 1e6 / len(batch.X))
            yp_all.append(pred); versions.append(version)
            # Report labels are used solely for retrospective metrics, never by
            # controller/gate/monitor. Score before ANY update at this batch.
            mask = context.roles[batch.row_start:batch.row_end] == 4
            scores = classification_metrics(batch.y[mask], pred[mask]) if mask.any() else None
            batches.append(dict(batch=batch.index, version=version, n_report=int(mask.sum()),
                                binary_f1=scores["binary"]["f1"] if scores else None,
                                n_attack=scores["binary"]["n_attack"] if scores else 0))
            for c in np.unique(batch.y[mask]):
                if c not in seen_classes:
                    selected = mask & (batch.y == c)
                    baseline_pred = context.initial.predict(batch.X[selected])
                    forward.append(float(np.mean(pred[selected] != 0) - np.mean(baseline_pred != 0)))
            seen_classes.update(np.unique(batch.y[mask]).tolist())
            matured = batch.index - policy.delay_batches
            if monitor is not None and matured >= 0:
                lo, hi = matured * bs, (matured + 1) * bs
                idx = np.flatnonzero(context.roles[lo:hi] == 3) + lo
                monitoring = policy.available(context.samples.take(idx), batch.index)
                evidence = monitor.observe(monitoring, policy, batch.index, versions[matured])
                if evidence and evidence["rolled_back"]:
                    rollbacks += 1
            if system != "static":
                for sig in scheduled.get(batch.index, []):
                    parent, parent_version = life.holder.get()
                    data, held = context.training_data(batch.index, parent_version, strategy)
                    context_record = dict(signature=sig.to_dict(), system=system, label_policy=asdict_policy(policy))
                    action, record, unavailable = Action.NO_OP, None, {}
                    if system in FIXED_ACTIONS:
                        action = FIXED_ACTIONS[system]
                    elif len(held[1]):
                        started, cpu_started = time.perf_counter(), time.process_time()
                        est, _, _, unavailable, _ = proxy.evaluate(parent, data, policy,
                                              ProxyHoldouts(held[1], context.historical[1]))
                        proxy_seconds += time.perf_counter() - started
                        proxy_cpu += time.process_time() - cpu_started
                        ep = dict(estimates=est, data=data, policy=policy, base=parent)
                        est = with_frozen_costs(ep, frozen["profiles"], strategy)
                        record = controller.decide(sig, est, parent_version, f"{sc.spec.name}:{sig.signature_id}",
                                                   cfg.training_budget, unavailable=unavailable)
                        action = Action(record.chosen_action)
                        context_record["controller"] = record.to_dict()
                    if not len(held[2]) or not len(data.recent):
                        action = Action.NO_OP
                        context_record["reason"] = "insufficient_released_training_or_gate_data"
                    if system == "active_incremental" and len(data.recent):
                        # Equal label access; use the least-confident half of
                        # already-released labels, not extra oracle queries.
                        confidence = parent.predict_with_confidence(data.recent.X)[1]
                        keep = np.argsort(confidence, kind="stable")[:max(2, len(confidence) // 2)]
                        data = AdaptationData(data.recent.take(keep), data.memory, data.now_batch,
                                              data.parent_version, data.excluded_ids)
                    try:
                        candidate = engine.apply(action, parent, data, policy)
                    except (ValueError, UnsupportedUpdate) as exc:
                        unsupported += 1
                        registry.record_event("ADAPTATION_UNAVAILABLE", dict(action=action.value, reason=str(exc),
                                                                            **context_record))
                        continue
                    proxy_ids = tuple(held[1].ids) + tuple(context.historical[1].ids)
                    result = life.process(candidate, EvaluationSets(held[2], context.historical[2]), policy, batch.index,
                                          gate_best, {c: m["f1"] for c, m in initial_gate["per_class"].items()},
                                          forbidden_ids=proxy_ids, context=context_record,
                                          bypass_gate=system in FIXED_ACTIONS or system == "evox_no_gate")
                    decisions.append(dict(batch=batch.index, action=action.value, status=result.status,
                                          version=result.version, parent_version=parent_version,
                                          gate_reason=result.decision.get("reason"), **candidate.profile))
                    if result.status == "NO_OP":
                        no_ops += 1; continue
                    if result.duplicate:
                        continue
                    attempts += 1
                    adaptations_seconds += candidate.profile["training_seconds"]
                    cpu_seconds += candidate.profile.get("cpu_seconds", 0.)
                    peak_python = max(peak_python, candidate.profile["peak_python_bytes"])
                    peak_rss = max(peak_rss, candidate.profile.get("sampled_peak_rss_bytes", 0))
                    max_model = max(max_model, candidate.profile["model_bytes"])
                    if result.status == "DEPLOYED":
                        promotions += 1
                        for c, m in result.evaluation["historical"]["per_class"].items():
                            gate_best[c] = max(gate_best.get(c, 0.), m["f1"])
                        if system.startswith("evox"):
                            monitor = RollbackMonitor(life, parent, batch.index, context.historical[3], MonitorConfig())
                    else:
                        rejected += 1
                    if "delta_f1_historical" in result.evaluation:
                        probes.append((candidate, parent, result.status, result.version, batch.index,
                                       result.evaluation["delta_f1_historical"]))
            if batch.index % 10 == 0:
                # Independent reporting anchor; these scores do not affect the policy.
                for c, m in model_metrics(life.holder.get()[0], context.historical[4])["per_class"].items():
                    report_best[c] = max(report_best.get(c, 0.), m["f1"])
        report_mask = context.roles == 4
        all_pred = np.concatenate(yp_all)
        overall = classification_metrics(context.samples.y[report_mask], all_pred[report_mask])
        final_history = model_metrics(life.holder.get()[0], context.historical[4])
        for c, m in final_history["per_class"].items():
            report_best[c] = max(report_best.get(c, 0.), m["f1"])
        forgetting = float(np.mean([report_best[c] - m["f1"] for c, m in final_history["per_class"].items()]))
        counterfactuals = []
        final_clock = sc.spec.n_batches - 1 + policy.delay_batches
        future = context.samples.take(report_mask)
        for c, parent, status, version, when, hist_delta in probes:
            cf = counterfactual_replay(c, parent, future, policy, final_clock, when + 1,
                                       min(when + cfg.counterfactual_batches, sc.spec.n_batches - 1),
                                       hist_delta, cfg.harmful_tolerance)
            cf.update(status=status, version=version, decision_batch=when)
            cf["horizon_truncated"] = when + cfg.counterfactual_batches >= sc.spec.n_batches
            counterfactuals.append(cf)
            registry.record_event("COUNTERFACTUAL_EVALUATED", cf)
        observed = [c for c in counterfactuals if not c["censored"]]
        summary = dict(**metadata, macro_f1=overall["macro_f1"], **overall["binary"], n_report=overall["n"],
                       historical_macro_f1=final_history["macro_f1"], forgetting=forgetting,
                       backward_transfer=final_history["macro_f1"] - initial_report["macro_f1"],
                       novel_forward_transfer_binary=float(np.mean(forward)) if forward else None,
                       initial_fit_seconds=context.initial.fit_seconds_, training_seconds=adaptations_seconds,
                       training_cpu_seconds=cpu_seconds, proxy_seconds=proxy_seconds, proxy_cpu_seconds=proxy_cpu,
                       peak_python_bytes=peak_python, max_model_bytes=max_model,
                       sampled_peak_rss_bytes=peak_rss,
                       latency_ms_per_1k=float(np.median(inference_times)),
                       attempts=attempts, updates=promotions, rejected=rejected, rollbacks=rollbacks,
                       no_ops=no_ops, unavailable_updates=unsupported,
                       harmful_promotions=sum(c["harmful"] for c in observed if c["status"] == "DEPLOYED"),
                       harmful_rejections=sum(c["harmful"] for c in observed if c["status"] == "REJECTED"),
                       unnecessary_updates=sum(c["unnecessary"] for c in observed if c["status"] == "DEPLOYED"),
                       counterfactual_censored=sum(c["censored"] for c in counterfactuals),
                       successful_adaptation_rate=float(np.mean([c["successful"] for c in observed if c["status"] == "DEPLOYED"]))
                           if any(c["status"] == "DEPLOYED" for c in observed) else None,
                       **recovery_metrics(batches, sc.truth, cfg.recovery_tolerance))
        registry.record_event("RUN_COMPLETED", summary)
        return summary, [dict(**metadata, **d) for d in decisions], [dict(**metadata, **c) for c in counterfactuals], \
            [dict(**metadata, **r) for r in batches], overall["per_class"]
    finally:
        registry.close()


def asdict_policy(policy):
    return dict(delay_batches=policy.delay_batches, labelled_fraction=policy.labelled_fraction, seed=policy.seed)
