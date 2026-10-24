"""Phase C: validation-selected Safety Gate and closed-loop comparative replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import zipfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from evox.adaptation.data import LabelPolicy
from evox.adaptation.strategies import StrategyConfig
from evox.config import RESULTS_DIR, STREAM_DIR, TEST_SEEDS, VALIDATION_SEEDS
from evox.data.stream import EvoStream, build_stream
from evox.data.synthetic import generate_synthetic_week
from evox.drift.scenarios import KINDS, MAGNITUDES, LabSource, ScenarioSpec
from evox.evaluation.comparison import hypotheses, paired_statistics, plot_comparison
from evox.evolution.safety import tune_gate
from evox.evolution.safety import SafetyGate
from evox.evolution.controller import ActionEstimate, ControllerThresholds, EvolutionController, UtilityWeights
from evox.experiments.continual import ReplayConfig, SYSTEMS, pilot_candidates, prepare_context, simulate
from evox.experiments.evolve import write_json

log = logging.getLogger("phase_c")


def code_digest():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode()); h.update(path.read_bytes())
    return h.hexdigest()


def source_digest(source):
    h = hashlib.sha256()
    h.update(json.dumps(dict(classes=source.classes, features=source.feature_names), sort_keys=True).encode())
    arrays = [source.X_train, source.y_train, source.X_val, source.y_val]
    arrays += [pool[c] for family,pool in sorted(source.pools.items()) for c in sorted(pool)]
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.shape).encode()); h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def save_source_snapshot(output):
    package = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*.py")):
            archive.write(path, "src/evox/" + path.relative_to(package).as_posix())
        for name in ("pyproject.toml", "requirements.txt"):
            path = package.parent.parent / name
            if path.exists():
                archive.write(path, name)


def load_frozen_controller(path, data_tag, model):
    path = Path(path)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    protocol_path = path.parent / "preregistered_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else frozen
    if frozen.get("data_tag") != data_tag or protocol.get("algorithm") != model:
        raise ValueError("Phase B controller source tag/algorithm does not match this experiment")
    seeds = frozen.get("validation_seeds", [])
    if not seeds or not set(seeds) <= set(VALIDATION_SEEDS) or set(seeds) & set(TEST_SEEDS):
        raise ValueError("Controller weights must have been selected on validation seeds only")
    if not frozen.get("written_before_test_scenarios_built"):
        raise ValueError("Phase B controller does not record freezing before test generation")
    for key in ("weights", "thresholds", "cost_scale", "latency_scale", "profiles"):
        if key not in frozen:
            raise ValueError("Incomplete Phase B controller configuration")
    if "label_policy" not in protocol:
        raise ValueError("Phase B configuration must include its label policy")
    EvolutionController(UtilityWeights(**frozen["weights"]), ControllerThresholds(**frozen["thresholds"]),
                        frozen["cost_scale"], frozen["latency_scale"])
    for action, profile in frozen["profiles"].items():
        from evox.adaptation.strategies import Action
        Action(action)
        ActionEstimate(cost_seconds=profile["training_seconds_per_row"],
                       latency_ms_per_1k=profile["added_latency_ms_per_1k"])
    return frozen, protocol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controller-config", required=True, help="frozen_controller.json from a matching Phase B run")
    ap.add_argument("--detector-config", help="defaults to the detector snapshot beside the Phase B controller")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true", help="one seed/family, medium magnitude; pipeline check only")
    ap.add_argument("--model", choices=("random_forest", "xgboost"), default="random_forest")
    ap.add_argument("--stream-dir", default=str(STREAM_DIR))
    ap.add_argument("--output-dir")
    ap.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    ap.add_argument("--test-seeds", type=int, nargs="+", default=list(TEST_SEEDS))
    ap.add_argument("--trees", type=int)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--budget-seconds", type=float, default=60.)
    ap.add_argument("--latency-budget", type=float, default=100.)
    a = ap.parse_args(argv)
    if a.batch_size < 2 or a.budget_seconds <= 0 or a.latency_budget <= 0 or not np.isfinite([a.budget_seconds, a.latency_budget]).all():
        ap.error("Batch size must be >=2 and budgets finite and positive")
    if len(a.systems) != len(set(a.systems)) or len(a.test_seeds) != len(set(a.test_seeds)):
        ap.error("Systems and test seeds must be unique")
    if not a.test_seeds or min(a.test_seeds) < 0 or set(a.test_seeds) & set(VALIDATION_SEEDS):
        ap.error("Test seeds must be nonnegative and disjoint from validation seeds")
    tag = "SYNTHETIC" if a.synthetic else "REAL"
    out = Path(a.output_dir) if a.output_dir else RESULTS_DIR / "runs" / ("phase_c_smoke" if a.quick else "phase_c")
    if out.exists() and any(out.iterdir()):
        ap.error("Output directory must be empty; choose a fresh --output-dir")
    try:
        frozen, previous = load_frozen_controller(a.controller_config, tag, a.model)
        if a.batch_size != previous.get("args", {}).get("batch_size", a.batch_size):
            raise ValueError("Batch size differs from the frozen Phase B protocol; calibrate a matching controller first")
        policy = LabelPolicy(**previous["label_policy"])
        detector_path = Path(a.detector_config) if a.detector_config else Path(a.controller_config).parent / "detector_config.json"
        detector = json.loads(detector_path.read_text(encoding="utf-8"))
        if detector["data_tag"] != tag or detector["label_delay_batches"] != policy.delay_batches:
            raise ValueError("Phase A source tag/label delay differs from the frozen Phase B protocol")
        if a.synthetic:
            stream = build_stream(generate_synthetic_week(60_000, seed=2), reference_fraction=.5)
        else:
            if not (Path(a.stream_dir) / "stream.npz").exists():
                raise ValueError("No real processed stream; run build-stream first (no synthetic fallback)")
            stream = EvoStream.load(a.stream_dir)
            if str(stream.meta.get("dataset", "")).upper().startswith("SYNTHETIC") or stream.meta.get("ordering_source") == "synthetic_scenario":
                raise ValueError("A synthetic/scenario stream cannot be reported as a REAL source")
        source = LabSource.from_evostream(stream, seed=0)
    except (OSError, KeyError, ValueError) as exc:
        ap.error(str(exc))
    trees = a.trees or previous.get("args", {}).get("trees", 60)
    if trees < 2:
        ap.error("At least two trees/boosting rounds are required")
    if a.quick:
        trees = min(trees, 16)
    strategy = StrategyConfig(**previous.get("full_strategy_config", {}))
    proxy = StrategyConfig(**previous.get("proxy_strategy_config", asdict(strategy)))
    config = ReplayConfig(model=a.model, trees=trees, reference_cap=1500 if a.quick else 6000,
                          holdout_cap=128 if a.quick else 512, mmd_samples=120 if a.quick else 200,
                          mmd_null=20 if a.quick else 40, training_budget=a.budget_seconds)
    val_seeds = VALIDATION_SEEDS[:1] if a.quick else VALIDATION_SEEDS
    test_seeds = tuple(a.test_seeds[:1] if a.quick else a.test_seeds)
    mags = ("med",) if a.quick else MAGNITUDES
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(data_tag=tag, quick=a.quick, algorithm=a.model, args=vars(a), replay=asdict(config),
                    label_policy=asdict(policy), strategy=asdict(strategy), proxy_strategy=asdict(proxy),
                    validation_seeds=list(val_seeds), test_seeds=list(test_seeds), magnitudes=list(mags),
                    systems=a.systems, role_split="source-grouped 50/15/15/10/10 train/proxy/gate/monitor/report",
                    common_triggers="frozen Phase A detector on the initial sentinel model; identical across policies",
                    threshold_grid=dict(new_gain=[0,.01,.02], historical_drop=[.02,.05], forgetting=[.03,.1],
                                        fpr_increase=[.005,.02], training_seconds=[a.budget_seconds/2,a.budget_seconds]),
                    gate_selection="initial-parent validation pilots; independent audit gain/retention minus normalized training cost",
                    harmful="future reporting macro-F1 loss >0.03 or gate historical F1 loss >0.03 vs frozen parent",
                    unnecessary="promoted candidate future reporting macro-F1 gain <0.01",
                    rollback="only EVO-X policies; two released monitor batches, >=32 rows; independent monitor source groups",
                    final_reporting="oracle benchmark labels on independent report rows, never passed to decisions",
                    counters="attempts include rejected candidates; updates are promotions; rollback switches are separate",
                    hypotheses="seed-paired Wilcoxon with Holm correction; synthetic/quick results always inconclusive",
                    f1_noninferiority="one-sided paired Wilcoxon for EVO-X minus each fixed policy + 0.02; Holm adjusted",
                    memory_measure="10ms sampled process RSS (includes resident/native state), Python allocation peak, serialized model size",
                    source_meta=source.meta, dropped_classes=source.dropped_classes,
                    code_sha256=code_digest(), source_sha256=source_digest(source),
                    phase_b_config_sha256=hashlib.sha256(Path(a.controller_config).read_bytes()).hexdigest(),
                    detector_sha256=hashlib.sha256(detector_path.read_bytes()).hexdigest(),
                    package_versions={p: version(p) for p in ("evo-x","numpy","pandas","scikit-learn","xgboost","scipy","river","psutil")})
    write_json(out / "preregistered_protocol.json", protocol)
    save_source_snapshot(out / "source_snapshot.zip")
    write_json(out / "frozen_controller.json", frozen)
    write_json(out / "detector_config.json", detector)
    pilot, coverage = [], []
    def specs(family, seeds):
        return [ScenarioSpec(kind, mag, seed, family, batch_size=a.batch_size)
                for seed in seeds for kind in KINDS for mag in mags] + [
                ScenarioSpec("stationary", "low", seed, family, batch_size=a.batch_size) for seed in seeds]
    for spec in specs("validation", val_seeds):
        ctx = prepare_context(source, spec, config, detector, policy)
        records = pilot_candidates(ctx, strategy)
        pilot.extend(records)
        coverage.append(dict(scenario=spec.name, family="validation", seed=spec.seed, n_alarms=len(ctx.signatures)))
        log.info("validation %s: %d pilot candidates", spec.name, len(records))
    if not pilot:
        ap.error("No validation candidates with sufficient released gate/audit data; cannot tune the Safety Gate")
    thresholds, sensitivity = tune_gate(pilot, a.budget_seconds, a.latency_budget)
    pd.DataFrame(sensitivity).assign(data_tag=tag).to_csv(out / "gate_sensitivity.csv", index=False)
    write_json(out / "validation_pilots.json", pilot)
    write_json(out / "frozen_gate.json", dict(data_tag=tag, algorithm=a.model, thresholds=asdict(thresholds),
                validation_seeds=list(val_seeds), written_before_test_scenarios_built=True))
    log.info("Gate frozen. Constructing TEST scenarios now.")
    all_results, decisions, counterfactuals, batches, per_class = [], [], [], [], []
    for spec in specs("test", test_seeds):
        ctx = prepare_context(source, spec, config, detector, policy)
        coverage.append(dict(scenario=spec.name, family="test", seed=spec.seed, n_alarms=len(ctx.signatures)))
        for system in a.systems:
            row, events, probes, batch_rows, classes = simulate(ctx, system, out / "registries" / spec.name / system,
                                                               SafetyGate(thresholds), frozen, strategy, proxy, tag)
            all_results.append(row); decisions.extend(events); counterfactuals.extend(probes); batches.extend(batch_rows)
            per_class.extend(dict(scenario=spec.name, system=system, seed=spec.seed, class_id=c, **m) for c,m in classes.items())
            log.info("test %s / %s: F1 %.3f, updates %d, rejected %d, rollbacks %d", spec.name, system,
                     row["macro_f1"], row["updates"], row["rejected"], row["rollbacks"])
        # A long run retains completed scenarios if interrupted later.
        pd.DataFrame(all_results).to_csv(out / "per_scenario.csv", index=False)
    results = pd.DataFrame(all_results)
    results.groupby("system").agg({m: ["mean","std"] for m in ("macro_f1","f1","forgetting","updates",
                                   "training_seconds","proxy_seconds","harmful_promotions","rollbacks")}).to_csv(out / "summary.csv")
    (pd.DataFrame(decisions) if decisions else pd.DataFrame(columns=["scenario","system","seed","batch","action","status"])).to_csv(out / "candidate_events.csv", index=False)
    (pd.DataFrame(counterfactuals) if counterfactuals else pd.DataFrame(columns=["scenario","system","seed","candidate_id","status","harmful","censored"])).to_csv(out / "counterfactuals.csv", index=False)
    pd.DataFrame(batches).to_csv(out / "batch_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(coverage).assign(data_tag=tag).to_csv(out / "scenario_coverage.csv", index=False)
    paired = paired_statistics(results)
    paired.to_csv(out / "paired_statistics.csv", index=False)
    hypothesis = hypotheses(results, paired, tag, a.quick)
    write_json(out / "hypotheses.json", hypothesis)
    plot_comparison(results, out / "comparison.png")
    summary = dict(data_tag=tag, quick=a.quick, algorithm=a.model, test_scenarios=results.scenario.nunique(),
                   systems=len(a.systems), runs=len(results), gate_vectors=len(sensitivity),
                   promotions=int(results.updates.sum()), rejections=int(results.rejected.sum()),
                   rollbacks=int(results.rollbacks.sum()), hypotheses={k:v["status"] for k,v in hypothesis.items()})
    write_json(out / "summary.json", summary)
    means = results.groupby("system")[["macro_f1","forgetting","updates","training_seconds","proxy_seconds"]].mean()
    lines = [f"# Phase C ({tag}{'; PIPELINE CHECK ONLY' if tag=='SYNTHETIC' or a.quick else ''})", "",
             f"Algorithm: {a.model}. Test scenarios: {summary['test_scenarios']}. Policies: {len(a.systems)}.", "",
             "| Policy | Macro-F1 | Forgetting | Updates | Train seconds | Proxy seconds |",
             "|---|---:|---:|---:|---:|---:|"]
    lines += ["| " + name + " | " + " | ".join(f"{x:.4f}" for x in row) + " |" for name,row in means.iterrows()]
    lines += ["", "## Hypotheses", ""] + [f"- {h}: **{r['status']}**. {r['evidence']}" for h,r in hypothesis.items()]
    lines += ["", "All policies share source-flow partitions, label access and sentinel drift triggers. "
              "Final scores use independent reporting rows; gate and monitoring data are separate. "
              "Fixed policies deploy valid candidates unconditionally and do not use rollback; EVO-X variants use the monitor. "
              "SSF-inspired and active-incremental policies are approximations, not reproductions of SSF/HOIDS. "
              "Counterfactuals compare a frozen candidate with its frozen parent for ten future batches. "
              "Resources include training/proxy CPU time, Python allocation peaks, serialized model sizes and 10ms sampled process RSS; "
              "RSS includes the surrounding process and may miss shorter peaks. "
              "Full real-data multi-seed runs are required before research claims."]
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
