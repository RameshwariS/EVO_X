"""Phase B shadow experiment: detect -> characterize -> rank -> build candidate.

Run --synthetic --quick for a pipeline check. Real runs require a Phase 2 stream
and a Phase A detector configuration tagged REAL. No model is deployed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from evox.adaptation.strategies import Action, AdaptationEngine, STRATEGIES, StrategyConfig
from evox.adaptation.data import AdaptationData, LabelPolicy, ReplayBuffer, Samples, balanced_subset, priority
from evox.config import LABEL_DELAY_BATCHES, METRICS_DIR, RESULTS_DIR, STREAM_DIR, TEST_SEEDS, VALIDATION_SEEDS
from evox.drift.analyzer import DriftAnalyzer
from evox.evaluation.drift import MemberConfig, compute_signals, fused_alarms, run_members, static_run
from evox.drift.scenarios import KINDS, MAGNITUDES, LabSource, ScenarioSpec, generate_scenario
from evox.evolution.controller import (ActionEstimate, ControllerThresholds, EvolutionController,
                                      publish_decision, tune_weights)
from evox.evolution.proxy import ProxyEvaluator, ProxyHoldouts
from evox.models.ids import IDSModel
from evox.data.stream import EvoStream, build_stream
from evox.streaming.transport import InMemoryTransport, KafkaTransport
from evox.data.synthetic import generate_synthetic_week

log = logging.getLogger("phase_b")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def split_roles(source_ids):
    """Assign ALL appearances of the same source flow to a single data role.

    The Phase A generator samples with replacement. Splitting by replay-row
    number would allow the same original flow into both training and holdout.
    60% source groups train; 20% proxy; 20% independent tuning audit.
    """
    return np.array([0 if (p := priority(str(s), 71) % 10) < 6 else 1 if p < 8 else 2 for s in source_ids])


def sample_set(X, y, prefix, batches=None):
    n = len(y)
    return Samples(X, y, np.array([f"{prefix}:{i}" for i in range(n)]),
                   np.full(n, -1) if batches is None else batches)


def budget_detector_errors(sig, samples, policy):
    """Error detectors get the same labelled fraction as the adaptation engine.

    run_members enforces the batch delay; here we withhold unlabelled rows.
    Confidence/feature detectors remain label-free and use the complete batch.
    """
    if policy.labelled_fraction == 1:
        return sig
    mask = np.array([priority(s, policy.seed) / 2**64 < policy.labelled_fraction for s in samples.ids])
    mask = mask.reshape(sig.n_batches, sig.batch_size)
    errors = [sig.run.err_flags[i][mask[i]] for i in range(sig.n_batches)]
    return replace(sig, run=replace(sig.run, err_flags=errors))


def load_source(args):
    if args.synthetic:
        stream = build_stream(generate_synthetic_week(60_000, seed=2), reference_fraction=0.5)
        tag = "SYNTHETIC"
    else:
        p = Path(args.stream_dir)
        if not (p / "stream.npz").is_file():
            raise ValueError(f"No processed stream at {p}. Build it first, or use --synthetic for a pipeline check.")
        stream, tag = EvoStream.load(p), "REAL"
    return LabSource.from_evostream(stream, seed=0), tag


def prepare_scenario(source, spec, args, detector, proxy_config, full_config, policy):
    sc = generate_scenario(source, spec)
    s = sc.stream
    ref = sample_set(s.X_ref_train, s.y_ref_train, spec.name + ":ref")
    ref = balanced_subset(ref, args.reference_cap, spec.seed)
    base = IDSModel(args.model, seed=spec.seed,
                    params=dict(n_estimators=args.trees, n_jobs=1)).fit(ref.X, ref.y)
    samples = sample_set(s.X_stream, s.y_stream, spec.name + ":flow",
                         np.arange(len(s.y_stream)) // spec.batch_size)
    run = static_run(sc, base)
    sig = compute_signals(sc, run, seed=spec.seed, mmd_n_sub=args.mmd_samples, mmd_null=args.mmd_null)
    sig = budget_detector_errors(sig, samples, policy)
    cfg = MemberConfig(**{k: v for d in detector["members"].values() for k, v in d.items()})
    members = run_members(sig, cfg, label_delay=policy.delay_batches)
    alarms = fused_alarms(members, detector["fusion"]["weights"], detector["fusion"]["theta"]).alarms
    analyzer = DriftAnalyzer(sig.ref_psi_detector, cfg.psi_threshold, cfg.mmd_mult * sig.mmd_null_q99,
                             ref_val_conf=base.predict_with_confidence(s.X_ref_val)[1])
    history = sample_set(s.X_ref_val, s.y_ref_val, spec.name + ":historical")
    history_roles = np.array([priority(x, 72) % 2 for x in history.ids])
    hist_proxy = balanced_subset(history.take(history_roles == 0), args.holdout_cap, 1)
    hist_audit = balanced_subset(history.take(history_roles == 1), args.holdout_cap, 2)
    roles = split_roles(sc.source_ids)
    memory = ReplayBuffer(full_config.memory_capacity, s.n_features, spec.seed)
    memory.add(ref, policy, 0)
    proxy = ProxyEvaluator(proxy_config)
    episodes, profiles = [], []
    Xb = s.X_stream.reshape(spec.n_batches, spec.batch_size, -1)
    for event_idx, alarm in enumerate(alarms):
        now = min(alarm + args.analysis_wait, spec.n_batches - 1)
        signature = analyzer.characterize(alarm, now, sig.psi_stat, sig.mmd_stat, Xb, run.conf_batches)
        released = policy.available(samples, now)
        # Select by sample identity after labels are released, never by truth events.
        pos = np.array([int(x.rsplit(":", 1)[1]) for x in released.ids], dtype=int)
        rroles = roles[pos]
        memory.add(released.take(rroles == 0), policy, now)
        window = released.batches >= max(0, now - policy.delay_batches - args.recent_batches + 1)
        recent = released.take(window & (rroles == 0))
        holdout = ProxyHoldouts(balanced_subset(released.take(window & (rroles == 1)), args.holdout_cap, 1), hist_proxy)
        audit = ProxyHoldouts(balanced_subset(released.take(window & (rroles == 2)), args.holdout_cap, 2), hist_audit)
        excluded = tuple(holdout.recent.ids) + tuple(holdout.historical.ids) + tuple(audit.recent.ids) + tuple(audit.historical.ids)
        data = AdaptationData(recent, memory.samples, now, f"{spec.name}:v1", excluded)
        event_id = f"{spec.name}:alarm{event_idx}:batch{alarm}"
        ep = dict(signature=signature, family=spec.family, seed=spec.seed, scenario=spec.name,
                  kind=spec.kind, magnitude=spec.magnitude, event_id=event_id, data=data,
                  base=base, budget_seconds=args.budget_seconds, policy=policy,
                  holdout=holdout, audit=audit)
        if not len(holdout.recent) or not len(audit.recent) or not len(hist_proxy) or not len(hist_audit):
            ep.update(estimates={Action.NO_OP: ActionEstimate()}, outcomes={}, profiles=[],
                      unavailable={a.value: "insufficient_released_holdout" for a in Action if a != Action.NO_OP},
                      baseline={}, tuneable=False)
        else:
            estimates, outcomes, measured, unavailable, baseline = proxy.evaluate(base, data, policy, holdout, audit)
            ep.update(estimates=estimates, outcomes=outcomes, profiles=measured, unavailable=unavailable,
                      baseline=baseline, tuneable=True)
            profiles.extend(dict(scenario=spec.name, family=spec.family, seed=spec.seed, **p) for p in measured)
        episodes.append(ep)
    log.info("%s: %d detected events", spec.name, len(episodes))
    return episodes, profiles, dict(scenario=spec.name, family=spec.family, seed=spec.seed,
                                    kind=spec.kind, magnitude=spec.magnitude, n_alarms=len(alarms))


def freeze_profiles(rows):
    profiles = {}
    for action in Action:
        group = [r for r in rows if r["action"] == action.value]
        if group:
            profiles[action.value] = dict(training_seconds_per_row=float(np.median([
                r["training_seconds"] / r["n_train"] for r in group])),
                added_latency_ms_per_1k=float(np.median([max(0.0, r["latency_ms_per_1k"] -
                                                               r["parent_latency_ms_per_1k"]) for r in group])))
    return profiles


def with_frozen_costs(ep, profiles, full_config):
    estimates = {Action.NO_OP: ActionEstimate()}
    recent, memory = ep["data"].released(ep["policy"])
    recent = balanced_subset(recent, full_config.recent_capacity, ep["base"].seed)
    memory = balanced_subset(memory, full_config.memory_capacity, ep["base"].seed)
    for action, estimate in ep["estimates"].items():
        action = Action(action)
        if action == Action.NO_OP or action.value not in profiles:
            continue
        n = len(STRATEGIES[action](full_config).training_data(ep["base"], recent, memory))
        p = profiles[action.value]
        estimates[action] = replace(estimate, cost_seconds=p["training_seconds_per_row"] * n,
                                   latency_ms_per_1k=p["added_latency_ms_per_1k"])
    return estimates


def run_decisions(episodes, controller, full_config, out, transport, tag, save_candidates):
    engine, rows, ablations = AdaptationEngine(full_config), [], []
    for ep in episodes:
        record = controller.decide(ep["signature"], ep["estimates"], ep["data"].parent_version,
                                   ep["event_id"], ep["budget_seconds"],
                                   context=dict(data_tag=tag, scenario=ep["scenario"], family=ep["family"],
                                                seed=ep["seed"], proxy_baseline=ep["baseline"],
                                                label_policy=asdict(ep["policy"]),
                                                proxy_training_seconds=sum(p["training_seconds"] for p in ep["profiles"]),
                                                candidate_training_is_shadow_only=True), unavailable=ep["unavailable"])
        candidate = engine.apply(record.chosen_action, ep["base"], ep["data"], ep["policy"])
        if save_candidates and candidate.action != Action.NO_OP.value:
            candidate.model.save(out / "candidates" / (candidate.candidate_id + ".joblib"))
        publish_decision(record, candidate, path=out / "evolution_events.jsonl", transport=transport)
        # Exercise the transport path even without Kafka; leave Kafka messages for subscribers.
        if isinstance(transport, InMemoryTransport):
            from evox.config import TOPIC_EVOLUTION
            message = transport.poll(TOPIC_EVOLUTION)
            if message is None or json.loads(message.value)["event_id"] != ep["event_id"]:
                raise RuntimeError("Evolution event transport roundtrip failed")
        rows.append(dict(event_id=ep["event_id"], scenario=ep["scenario"], family=ep["family"], seed=ep["seed"],
                         kind=ep["kind"], magnitude=ep["magnitude"], data_tag=tag, action=record.chosen_action,
                         rule=record.rule, estimated_cost_seconds=ep["estimates"].get(Action(record.chosen_action),
                                                                                   ActionEstimate()).cost_seconds,
                         proxy_training_seconds=sum(p["training_seconds"] for p in ep["profiles"]),
                         **candidate.profile))
        for mode in ("rules_only", "utility_only", "rules_utility"):
            c = EvolutionController(controller.weights, controller.thresholds, controller.cost_scale,
                                    controller.latency_scale, mode)
            r = c.decide(ep["signature"], ep["estimates"], ep["data"].parent_version,
                         ep["event_id"], ep["budget_seconds"])
            outcome = ep["outcomes"].get(r.chosen_action, {})
            ablations.append(dict(event_id=ep["event_id"], scenario=ep["scenario"], family=ep["family"],
                                  data_tag=tag, mode=mode, action=r.chosen_action, **outcome))
    return rows, ablations


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true", help="one seed/family, medium magnitude; pipeline check")
    ap.add_argument("--model", choices=("random_forest", "xgboost"), default="random_forest")
    ap.add_argument("--stream-dir", default=str(STREAM_DIR))
    ap.add_argument("--detector-config", default=str(METRICS_DIR / "phase6_tuned_config.json"))
    ap.add_argument("--output-dir")
    ap.add_argument("--trees", type=int, default=60)
    ap.add_argument("--reference-cap", type=int, default=6000)
    ap.add_argument("--holdout-cap", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--mmd-samples", type=int, default=200)
    ap.add_argument("--mmd-null", type=int, default=40)
    ap.add_argument("--analysis-wait", type=int, default=8)
    ap.add_argument("--recent-batches", type=int, default=12)
    ap.add_argument("--label-delay", type=int, default=LABEL_DELAY_BATCHES)
    ap.add_argument("--labelled-fraction", type=float, default=1.0)
    ap.add_argument("--budget-seconds", type=float, default=60.0, help="per-event candidate training budget")
    ap.add_argument("--transport", choices=("memory", "kafka"), default="memory")
    ap.add_argument("--save-candidates", action="store_true")
    a = ap.parse_args(argv)
    if min(a.trees, a.reference_cap, a.holdout_cap, a.batch_size, a.mmd_samples, a.mmd_null, a.recent_batches) < 2:
        ap.error("Counts/caps must be at least 2")
    if a.analysis_wait < 0 or a.budget_seconds < 0 or not np.isfinite(a.budget_seconds):
        ap.error("Analysis wait and budget must be nonnegative (budget finite)")
    if a.quick:
        a.trees, a.reference_cap, a.holdout_cap = min(a.trees, 16), min(a.reference_cap, 1500), min(a.holdout_cap, 128)
        # Preserve the Phase A batch size: smaller batches alter PSI noise and
        # would invalidate a frozen detector threshold even in a smoke run.
        a.mmd_samples, a.mmd_null = min(a.mmd_samples, 120), min(a.mmd_null, 20)
    policy = LabelPolicy(a.label_delay, a.labelled_fraction)
    full_config = StrategyConfig(recent_capacity=512, memory_capacity=512, selection_capacity=384) if a.quick else StrategyConfig()
    proxy_config = replace(full_config, recent_capacity=128 if a.quick else 256,
                           memory_capacity=128 if a.quick else 256, selection_capacity=128 if a.quick else 256)
    out = Path(a.output_dir) if a.output_dir else RESULTS_DIR / "runs" / ("phase_b_smoke" if a.quick else "phase_b")
    if out.exists() and any(out.iterdir()):
        ap.error(f"Output directory is not empty: {out}. Use a new --output-dir to preserve previous runs.")
    try:
        source, tag = load_source(a)
        detector_path = Path(a.detector_config)
        detector = json.loads(detector_path.read_text(encoding="utf-8"))
        if detector["data_tag"] != tag:
            raise ValueError("Detector config data_tag differs from the requested source; use a matching Phase A config")
        if a.label_delay != detector["label_delay_batches"]:
            raise ValueError("Label delay differs from the frozen Phase A config; supply a matching config")
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    val_seeds, test_seeds = (VALIDATION_SEEDS[:1], TEST_SEEDS[:1]) if a.quick else (VALIDATION_SEEDS, TEST_SEEDS)
    mags = ("med",) if a.quick else MAGNITUDES
    thresholds = ControllerThresholds()
    protocol = dict(data_tag=tag, quick=a.quick, algorithm=a.model, python=platform.python_version(), args=vars(a),
                    package_versions={p: version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost",
                                                               "scipy", "river", "joblib")},
                    validation_seeds=val_seeds, test_seeds=test_seeds, thresholds=asdict(thresholds),
                    full_strategy_config=asdict(full_config), proxy_strategy_config=asdict(proxy_config),
                    label_policy=asdict(policy), weight_grid=[0, 0.25, 0.5, 1],
                    objective="mean(0.5*(audit_recent_macro_f1 + audit_historical_macro_f1) - 0.05*normalized_cost)",
                    split="source-flow grouped 60/20/20 train/proxy/audit; historical validation split in half",
                    cost_model="median validation proxy seconds/row * final training rows; validation-only added latency",
                    unnecessary_update="candidate new-data macro-F1 gain < 0.01 on independent evaluation; Phase C also counts gate rejections",
                    no_automatic_promotion=True, detector_config_sha256=hashlib.sha256(detector_path.read_bytes()).hexdigest(),
                    source_meta=source.meta)
    # Written BEFORE validation selection and BEFORE creating any test scenarios.
    write_json(out / "preregistered_protocol.json", protocol)
    write_json(out / "detector_config.json", detector)
    val, measured, scenarios = [], [], []
    def prepare(family, seeds):
        episodes, profs, scenario_rows = [], [], []
        specs = [ScenarioSpec(k, m, seed, family, batch_size=a.batch_size)
                 for seed in seeds for k in KINDS for m in mags]
        specs += [ScenarioSpec("stationary", "low", seed, family, batch_size=a.batch_size) for seed in seeds]
        for spec in specs:
            es, ps, sr = prepare_scenario(source, spec, a, detector, proxy_config, full_config, policy)
            episodes.extend(es); profs.extend(ps); scenario_rows.append(sr)
        return episodes, profs, scenario_rows
    val, measured, scenarios = prepare("validation", val_seeds)
    frozen_profiles = freeze_profiles(measured)
    for ep in val:
        ep["estimates"] = with_frozen_costs(ep, frozen_profiles, full_config)
    cost_scale = max([e.cost_seconds for ep in val for e in ep["estimates"].values()] + [1e-6])
    latency_scale = max([e.latency_ms_per_1k for ep in val for e in ep["estimates"].values()] + [1e-6])
    tuneable = [ep for ep in val if ep["tuneable"]]
    if not tuneable:
        ap.error("No validation alarms have enough released holdout labels to select weights; review label budget/detector settings.")
    weights, sensitivity = tune_weights(tuneable, cost_scale, latency_scale, thresholds=thresholds)
    pd.DataFrame(sensitivity).assign(data_tag=tag).to_csv(out / "weight_sensitivity.csv", index=False)
    frozen = dict(data_tag=tag, weights=weights.normalized(), thresholds=asdict(thresholds), cost_scale=cost_scale,
                  latency_scale=latency_scale, profiles=frozen_profiles, validation_seeds=val_seeds,
                  written_before_test_scenarios_built=True)
    write_json(out / "frozen_controller.json", frozen)
    log.info("Weights and cost profiles frozen. Preparing held-out test scenarios.")
    test, test_profiles, test_scenarios = prepare("test", test_seeds)
    for ep in test:
        ep["estimates"] = with_frozen_costs(ep, frozen_profiles, full_config)
    controller = EvolutionController(weights, thresholds, cost_scale, latency_scale)
    transport = InMemoryTransport() if a.transport == "memory" else KafkaTransport(group_id="evox-phase-b")
    try:
        if a.transport == "kafka":
            transport.create_topics()
        rows, ablations = run_decisions(val + test, controller, full_config, out, transport, tag, a.save_candidates)
        transport.flush()
    finally:
        transport.close()
    pd.DataFrame(rows).to_csv(out / "candidate_profiles.csv", index=False)
    pd.DataFrame(measured + test_profiles).assign(data_tag=tag).to_csv(out / "proxy_profiles.csv", index=False)
    pd.DataFrame(ablations).to_csv(out / "ablations.csv", index=False)
    pd.DataFrame(scenarios + test_scenarios).assign(data_tag=tag).to_csv(out / "scenario_coverage.csv", index=False)
    summary = dict(data_tag=tag, quick=a.quick, n_scenarios=len(scenarios) + len(test_scenarios),
                   n_detected_events=len(rows), n_validation_events=len(val), n_test_events=len(test),
                   actions=pd.Series([r["action"] for r in rows]).value_counts().to_dict(),
                   n_weight_vectors=len(sensitivity), deployment_status="SHADOW_ONLY")
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text(
        f"# Phase B run ({tag}{'; PIPELINE CHECK ONLY' if tag == 'SYNTHETIC' else ''})\n\n"
        f"Algorithm: {a.model}. Quick mode: {a.quick}. Scenarios: {summary['n_scenarios']}. "
        f"Detected events: {len(rows)}. Weight vectors: {len(sensitivity)}.\n\n"
        f"Selected actions: {json.dumps(summary['actions'])}. All candidates remain SHADOW_ONLY.\n\n"
        "The source-flow grouped holdouts prevent resampled flows crossing training/proxy/audit roles. "
        "Labels obey the recorded delay and fraction. Weights and cost profiles use validation scenarios only. "
        "Ablation metrics are proxy-candidate estimates, not deployed continual-learning results. "
        "Training memory is Python allocation peak plus serialized model size; native allocation/RSS is not measured. "
        "Phase C must independently evaluate and gate full candidates before deployment.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
