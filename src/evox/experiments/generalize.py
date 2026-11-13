"""Phase D1: generalisation experiments (RQ6/H5 algorithm transfer, RQ7 dataset portability).

Sub-commands
------------
adapter-check   Verify a dataset adapter on the REAL files before any experiment relies on it.
build-stream    Build a processed EvoStream for UNSW-NB15 or CSE-CIC-IDS2018 (never touches the
                CICIDS2017 stream or the ``phase2_*`` reports).
transfer        Run a frozen controller tuned on one algorithm against another algorithm
                (zero re-tuning of the decision logic) and compare with the native controller.
report          Combine Phase C runs from several datasets/algorithms into portability tables.

Cross-dataset recipe (per dataset; each step is an existing command)::

    evox generalize build-stream --dataset unsw-nb15 --raw-dir data/raw_unsw
    evox detect  --stream-dir data/processed/unsw_nb15_stream
    evox evolve  --stream-dir data/processed/unsw_nb15_stream --detector-config <detector.json> --output-dir ...
    evox compare --stream-dir data/processed/unsw_nb15_stream --controller-config <frozen_controller.json> ...
    evox generalize report --run CICIDS2017/random_forest=<dir> --run UNSW-NB15/random_forest=<dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evox.config import BENIGN_LABEL, LABEL_COLUMN, PROCESSED_DIR, RESULTS_DIR, TIMESTAMP_COLUMN
from evox.evaluation.generalization import (FIXED_POLICIES, advantage_table, decision_agreement, portability_table,
                                            summarize_transfer, utility_rank_correlation)

log = logging.getLogger("phase_d")

DATASETS = {
    "unsw-nb15": dict(name="UNSW-NB15", slug="unsw_nb15", loader="load_unsw_nb15"),
    "cse-cic-ids2018": dict(name="CSE-CIC-IDS2018", slug="cse_cic_ids2018", loader="load_cse_cic_ids2018"),
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# adapter-check
# ---------------------------------------------------------------------------
def check_adapter_frame(frame: pd.DataFrame, min_flows: int = 120) -> dict:
    """Sanity checks for an adapted frame (canonical schema expected by ``build_stream``).

    Returns ``dict(ok, checks=[{name, ok, detail}], summary={...})``.  A failed check
    means the adapter must be fixed before any Phase D number is trusted.
    """
    checks = []

    def add(name, ok, detail=""):
        checks.append(dict(name=name, ok=bool(ok), detail=str(detail)))

    add("has_label_column", LABEL_COLUMN in frame.columns, LABEL_COLUMN)
    if LABEL_COLUMN not in frame.columns:
        return dict(ok=False, checks=checks, summary={})
    labels = frame[LABEL_COLUMN].astype(str).str.strip()
    counts = labels.value_counts()
    attack = counts.drop(BENIGN_LABEL, errors="ignore")
    add("has_benign", BENIGN_LABEL in counts.index, f"{int(counts.get(BENIGN_LABEL, 0))} benign flows")
    add("at_least_two_attack_classes", len(attack) >= 2, f"{len(attack)} attack classes")
    usable = int((attack >= min_flows).sum())
    add("attack_classes_with_min_flows", usable >= 2, f"{usable} classes have >= {min_flows} flows")
    has_ts = TIMESTAMP_COLUMN in frame.columns
    ts_ok = False
    if has_ts:
        ts = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
        ts_ok = ts.notna().mean() >= 0.99
        add("timestamps_parse", ts_ok, f"{100 * ts.notna().mean():.2f}% parsed")
        spread = (ts.max() - ts.min()) if ts.notna().any() else pd.Timedelta(0)
        add("timestamps_span_time", spread > pd.Timedelta(0), f"span {spread}")
    else:
        add("timestamps_parse", False, "no Timestamp column: the stream would fall back to row order")
    features = [c for c in frame.columns if c not in (LABEL_COLUMN, TIMESTAMP_COLUMN)
                and pd.api.types.is_numeric_dtype(frame[c])]
    add("numeric_features_present", len(features) >= 10, f"{len(features)} numeric feature columns")
    nan_share = float(frame[features].isna().mean().mean()) if features else 1.0
    add("nan_share_reasonable", nan_share < 0.2, f"{100 * nan_share:.2f}% missing")
    constant = [c for c in features if frame[c].nunique(dropna=True) <= 1]
    add("few_constant_features", len(constant) <= 0.3 * max(1, len(features)), f"{len(constant)} constant columns")
    summary = dict(rows=int(len(frame)), n_features=len(features), class_counts={k: int(v) for k, v in counts.items()},
                   nan_share=nan_share, constant_features=constant, timestamp_ok=bool(ts_ok))
    return dict(ok=all(c["ok"] for c in checks), checks=checks, summary=summary)


def load_dataset(key: str, raw_dir: Path, subsample_every: int = 1) -> pd.DataFrame:
    from evox.data import adapters
    loader = getattr(adapters, DATASETS[key]["loader"])
    return loader(Path(raw_dir)) if key == "unsw-nb15" else loader(Path(raw_dir), subsample_every)


def cmd_adapter_check(a) -> int:
    try:
        frame = load_dataset(a.dataset, a.raw_dir, a.subsample_every)
    except (OSError, ValueError) as exc:
        print(f"Cannot load {a.dataset}: {exc}")
        return 1
    result = check_adapter_frame(frame, a.min_flows)
    for c in result["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    if a.json:
        write_json(a.json, result)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# build-stream
# ---------------------------------------------------------------------------
def cmd_build_stream(a) -> int:
    from evox.data.stream import build_stream
    from evox.evaluation.stream import validate_stream, write_reports
    info = DATASETS[a.dataset]
    out = Path(a.out_dir) if a.out_dir else PROCESSED_DIR / f"{info['slug']}_stream"
    try:
        frame = load_dataset(a.dataset, a.raw_dir, a.subsample_every)
    except (OSError, ValueError) as exc:
        print(f"Cannot load {a.dataset}: {exc}")
        return 1
    check = check_adapter_frame(frame)
    if not check["ok"] and not a.force:
        for c in check["checks"]:
            if not c["ok"]:
                print(f"[FAIL] {c['name']}: {c['detail']}")
        print("Adapter checks failed; fix the adapter or pass --force to build anyway.")
        return 1
    # fix_pm is a CICIDS2017-only timestamp repair and must not touch other datasets.
    stream = build_stream(frame, a.reference_fraction, a.reference_val_fraction, a.batch_size,
                          deduplicate=not a.no_dedup, fix_pm=False, dataset_name=info["name"])
    stream.save(out)
    write_reports(stream, tag=f"phase_d_{info['slug']}")   # never overwrites the phase2_* CICIDS2017 reports
    checks = validate_stream(stream)
    for name, (ok, msg) in checks.items():
        log.info("  [%s] %s %s", "PASS" if ok else "FAIL", name, msg)
    log.info("Stream saved to %s; novel-class share of live stream %.1f%%", out, 100 * stream.novel_mask().mean())
    return 0 if all(ok for ok, _ in checks.values()) else 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def parse_run_spec(text: str):
    """``DATASET/ALGORITHM=DIR`` -> (dataset, algorithm, Path)."""
    m = re.fullmatch(r"([^/=]+)/([^/=]+)=(.+)", text)
    if not m or m.group(2) not in ("random_forest", "xgboost"):
        raise ValueError(f"Bad --run {text!r}: expected DATASET/(random_forest|xgboost)=RUN_DIR")
    return m.group(1), m.group(2), Path(m.group(3))


def load_phase_c_run(path: Path, algorithm: str) -> tuple[pd.DataFrame, dict]:
    table, proto = path / "per_scenario.csv", path / "preregistered_protocol.json"
    if not table.is_file() or not proto.is_file():
        raise ValueError(f"{path} is not a finished Phase C run (needs per_scenario.csv and preregistered_protocol.json)")
    protocol = json.loads(proto.read_text(encoding="utf-8"))
    if protocol.get("algorithm") != algorithm:
        raise ValueError(f"{path} was run with {protocol.get('algorithm')!r}, not {algorithm!r}")
    return pd.read_csv(table), protocol


def plot_portability(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = table[(table.kind == "ALL") & (table.baseline.isin(["best_fixed", "always_retrain", "static"]))]
    baselines = [b for b in ("static", "always_retrain", "best_fixed") if b in set(data.baseline)]
    groups = sorted(set(zip(data.dataset, data.algorithm)))
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(groups) + 2), 4.5))
    width = 0.8 / max(1, len(baselines))
    for j, b in enumerate(baselines):
        vals = [float(data[(data.dataset == d) & (data.algorithm == m) & (data.baseline == b)].advantage.iloc[0])
                if len(data[(data.dataset == d) & (data.algorithm == m) & (data.baseline == b)]) else np.nan
                for d, m in groups]
        ax.bar(np.arange(len(groups)) + j * width, vals, width, label=f"vs {b}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(groups)) + width * (len(baselines) - 1) / 2)
    ax.set_xticklabels([f"{d}\n{m}" for d, m in groups], fontsize=8)
    ax.set_ylabel("EVO-X minus baseline (macro-F1)")
    ax.set_title("Where the advantage holds, shrinks or disappears (pooled scenarios)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)


def cmd_report(a) -> int:
    out = Path(a.output_dir) if a.output_dir else RESULTS_DIR / "runs" / "phase_d_generalization"
    if out.exists() and any(out.iterdir()):
        print("Output directory must be empty; choose a fresh --output-dir")
        return 2
    tables, flags, protocols = [], [], {}
    try:
        for spec in a.run:
            dataset, algorithm, path = parse_run_spec(spec)
            per_scenario, protocol = load_phase_c_run(path, algorithm)
            tables.append(advantage_table(per_scenario, dataset, algorithm))
            flags.append(dict(dataset=dataset, algorithm=algorithm, data_tag=protocol.get("data_tag"),
                              quick=bool(protocol.get("quick")), n_seeds=int(per_scenario.seed.nunique()),
                              run_dir=str(path)))
            protocols[(dataset, algorithm)] = protocol
        advantages = pd.concat(tables, ignore_index=True)
        ref = a.reference_dataset or parse_run_spec(a.run[0])[0]
        portable = portability_table(advantages, ref, a.margin, a.retain)
    except (ValueError, OSError) as exc:
        print(str(exc))
        return 2
    out.mkdir(parents=True, exist_ok=True)
    portable.to_csv(out / "portability.csv", index=False)
    pd.DataFrame(flags).to_csv(out / "runs_included.csv", index=False)
    plot_portability(portable, out / "portability.png")
    pipeline_only = any(f["data_tag"] != "REAL" or f["quick"] or f["n_seeds"] < 6 for f in flags)
    all_rows = portable[(portable.kind == "ALL") & (portable.baseline == "best_fixed")]
    lines = [f"# Phase D1 portability ({'PIPELINE CHECK ONLY' if pipeline_only else 'REAL, >=6 seeds'})", "",
             f"Reference dataset: **{ref}**. 'Holds' means the same-sign advantage keeps at least "
             f"{a.retain:.0%} of the reference size; margin {a.margin}.", "",
             "| Dataset | Algorithm | EVO-X F1 | Best-fixed F1 | Advantage | Reference | Verdict | Cost ratio |",
             "|---|---|---:|---:|---:|---:|---|---:|"]
    for _, r in all_rows.iterrows():
        cost = "n/a" if r.cost_ratio is None or pd.isna(r.cost_ratio) else f"{r.cost_ratio:.2f}"
        lines.append(f"| {r.dataset} | {r.algorithm} | {r.f1_evox:.4f} | {r.f1_base:.4f} | {r.advantage:+.4f} | "
                     f"{r.reference_advantage:+.4f} | {r.portability} | {cost} |")
    per_kind = portable[(portable.kind != "ALL") & (portable.baseline == "best_fixed")]
    if len(per_kind):
        lines += ["", "## Per drift kind (vs best fixed policy in hindsight)", "",
                  "| Dataset | Algorithm | Kind | Advantage | Verdict |", "|---|---|---|---:|---|"]
        lines += [f"| {r.dataset} | {r.algorithm} | {r.kind} | {r.advantage:+.4f} | {r.portability} |"
                  for _, r in per_kind.iterrows()]
    lines += ["", "The advantage is reported per dataset and per drift kind and is never averaged across datasets. "
              "`best_fixed` picks, for every scenario, the best fixed policy in hindsight (the strictest comparison). "
              "Runs on synthetic data, quick runs, or fewer than six seeds validate the pipeline only. "
              "Datasets differ in features, attack families and (for UNSW-NB15) categorical columns dropped by the adapter."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "summary.json", dict(reference_dataset=ref, pipeline_check_only=pipeline_only, runs=flags,
                                          verdicts={f"{r.dataset}/{r.algorithm}": r.portability for _, r in all_rows.iterrows()}))
    print((out / "report.md").read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------------------
# transfer
# ---------------------------------------------------------------------------
def build_controller(frozen: dict, mode: str = "rules_utility"):
    from evox.evolution.controller import ControllerThresholds, EvolutionController, UtilityWeights
    return EvolutionController(UtilityWeights(**frozen["weights"]), ControllerThresholds(**frozen["thresholds"]),
                               frozen["cost_scale"], frozen["latency_scale"], mode=mode)


def collect_decisions(ctx, native_frozen, source_frozen, strategy, proxy_config, budget, label):
    """Ask both controllers to decide on IDENTICAL proxy estimates at the same decision points.

    Trajectories of full replays diverge as soon as two controllers choose differently, so
    decision agreement is measured at fixed decision points on the initial parent model.
    """
    from evox.evolution.proxy import ProxyEvaluator, ProxyHoldouts
    from evox.experiments.evolve import with_frozen_costs
    native, source = build_controller(native_frozen), build_controller(source_frozen)
    proxy, rows = ProxyEvaluator(proxy_config), []
    for sig in ctx.signatures:
        data, held = ctx.training_data(sig.now_batch, "v1", strategy)
        if not len(held[1]) or not len(data.recent):
            continue
        est, _, _, unavailable, _ = proxy.evaluate(ctx.initial, data, ctx.policy, ProxyHoldouts(held[1], ctx.historical[1]))
        ep = dict(estimates=est, data=data, policy=ctx.policy, base=ctx.initial)
        decisions = {}
        for name, controller, frozen in (("native", native, native_frozen), ("transfer", source, source_frozen)):
            costs = with_frozen_costs(ep, frozen["profiles"], strategy)
            record = controller.decide(sig, costs, "v1", f"{ctx.scenario.spec.name}:{sig.signature_id}", budget,
                                       unavailable=unavailable)
            decisions[name] = (record.chosen_action, {k: v["total"] for k, v in record.utilities.items()})
        key = f"{ctx.scenario.spec.name}:{sig.signature_id}"
        agree = decision_agreement({key: decisions["native"][0]}, {key: decisions["transfer"][0]})
        rows.append(dict(scenario=ctx.scenario.spec.name, seed=ctx.scenario.spec.seed, batch=sig.now_batch,
                         native_action=decisions["native"][0], transfer_action=decisions["transfer"][0],
                         agree=agree["agreement"],
                         rank_correlation=utility_rank_correlation(decisions["native"][1], decisions["transfer"][1]),
                         label=label))
    return rows


def aggregate_agreement(rows: list[dict]) -> pd.DataFrame:
    """One row per scenario with its agreement rate (for the seed-level summary)."""
    if not rows:
        return pd.DataFrame(columns=["scenario", "seed", "n_decisions", "agreement", "rank_correlation"])
    frame = pd.DataFrame(rows)
    return frame.groupby(["scenario", "seed"]).agg(n_decisions=("agree", "size"), agreement=("agree", "mean"),
                                                    rank_correlation=("rank_correlation", "mean")).reset_index()


def _load_source(synthetic: bool, stream_dir: str):
    from evox.data.stream import EvoStream, build_stream
    from evox.data.synthetic import generate_synthetic_week
    from evox.drift.scenarios import LabSource
    if synthetic:
        stream = build_stream(generate_synthetic_week(60_000, seed=2), reference_fraction=.5)
    else:
        if not (Path(stream_dir) / "stream.npz").exists():
            raise ValueError("No real processed stream; build it first (there is no synthetic fallback)")
        stream = EvoStream.load(stream_dir)
        if str(stream.meta.get("dataset", "")).upper().startswith("SYNTHETIC") or \
                stream.meta.get("ordering_source") == "synthetic_scenario":
            raise ValueError("A synthetic/scenario stream cannot be reported as a REAL source")
    return LabSource.from_evostream(stream, seed=0)


def cmd_transfer(a) -> int:
    from evox.adaptation.data import LabelPolicy
    from evox.adaptation.strategies import StrategyConfig
    from evox.drift.scenarios import KINDS, MAGNITUDES, ScenarioSpec
    from evox.evolution.safety import GateThresholds, SafetyGate
    from evox.experiments.compare import code_digest, load_frozen_controller, save_source_snapshot
    from evox.experiments.continual import ReplayConfig, prepare_context, simulate

    out = Path(a.output_dir) if a.output_dir else RESULTS_DIR / "runs" / "phase_d_transfer"
    if out.exists() and any(out.iterdir()):
        print("Output directory must be empty; choose a fresh --output-dir")
        return 2
    tag = "SYNTHETIC" if a.synthetic else "REAL"
    try:
        target_dir = Path(a.target_run)
        native, protocol = load_frozen_controller(target_dir / "frozen_controller.json", tag, a.model)
        source, _ = load_frozen_controller(a.source_controller, tag, a.source_model)
        if a.source_model == a.model:
            print("Note: source and target algorithm are identical; this is a self-transfer sanity check, not RQ6.")
        detector = json.loads((target_dir / "detector_config.json").read_text(encoding="utf-8"))
        gate_file = json.loads((target_dir / "frozen_gate.json").read_text(encoding="utf-8"))
        if gate_file.get("data_tag") != tag or gate_file.get("algorithm") != a.model:
            raise ValueError("The target run's gate does not match this experiment's data tag/algorithm")
        thresholds = GateThresholds(**gate_file["thresholds"])
        if (source.get("label_policy") or {}) != (native.get("label_policy") or {}) and source.get("label_policy") and native.get("label_policy"):
            raise ValueError("Source and target controllers were frozen under different label policies")
        policy = LabelPolicy(**protocol["label_policy"])
        config = ReplayConfig(**protocol["replay"])
        strategy = StrategyConfig(**protocol["strategy"])
        proxy_config = StrategyConfig(**protocol["proxy_strategy"])
        batch_size = int(protocol["args"]["batch_size"])
        seeds = list(a.test_seeds or protocol["test_seeds"])
        mags = list(protocol["magnitudes"])
        quick = bool(protocol.get("quick"))
        source_data = _load_source(a.synthetic, a.stream_dir)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(str(exc))
        return 2
    gate = SafetyGate(thresholds)
    specs = [ScenarioSpec(k, m, s, "test", batch_size=batch_size) for s in seeds for k in KINDS for m in mags]
    specs += [ScenarioSpec("stationary", "low", s, "test", batch_size=batch_size) for s in seeds]
    if a.kinds:
        specs = [sp for sp in specs if sp.kind in a.kinds]
    if a.limit:
        specs = specs[:a.limit]
    if not specs:
        print("No scenarios selected")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "controllers.json", dict(native=native, transferred=source, source_algorithm=a.source_model,
                                              target_algorithm=a.model, gate=gate_file))
    write_json(out / "preregistered_protocol.json", dict(
        data_tag=tag, quick=quick, algorithm=a.model, source_algorithm=a.source_model, args=vars(a),
        design="frozen source-algorithm controller (weights, thresholds, cost scales, cost profiles) run unchanged on the "
               "target algorithm; target gate and detector are the target's own validation-selected settings",
        native_reference="target algorithm's own frozen controller", seeds=seeds, magnitudes=mags,
        agreement="both controllers decide on identical proxy estimates at the same decision points (initial parent)",
        code_sha256=code_digest()))
    save_source_snapshot(out / "source_snapshot.zip")
    systems = ["static"] + list(a.systems)
    results, events, agreements = [], [], []
    for spec in specs:
        ctx = prepare_context(source_data, spec, config, detector, policy)
        for system in systems:
            row, ev, *_ = simulate(ctx, system, out / "registries" / spec.name / system, gate, native, strategy, proxy_config, tag)
            results.append(row); events.extend(ev)
        for label, frozen in (("evox_native", native), ("evox_transfer", source)):
            row, ev, *_ = simulate(ctx, "evox", out / "registries" / spec.name / label, gate, frozen, strategy, proxy_config, tag)
            row["system"] = label
            row["controller_source"] = a.model if label == "evox_native" else a.source_model
            results.append(row)
            events.extend(dict(e, system=label) for e in ev)
        agreements.extend(collect_decisions(ctx, native, source, strategy, proxy_config, config.training_budget, spec.name))
        log.info("%s: %s", spec.name, {r["system"]: round(r["macro_f1"], 3) for r in results if r["scenario"] == spec.name})
        pd.DataFrame(results).to_csv(out / "per_scenario.csv", index=False)
    per_scenario = pd.DataFrame(results)
    pd.DataFrame(events).to_csv(out / "candidate_events.csv", index=False)
    pd.DataFrame(agreements).to_csv(out / "decision_points.csv", index=False)
    agreement = aggregate_agreement(agreements)
    agreement.to_csv(out / "decision_agreement.csv", index=False)
    summary = summarize_transfer(per_scenario, agreement, tag, quick)
    summary.update(source_algorithm=a.source_model, target_algorithm=a.model, scenarios=len(specs))
    write_json(out / "summary.json", summary)
    tb = per_scenario.groupby("system")[["macro_f1", "updates", "training_seconds", "forgetting"]].mean()
    lines = [f"# Phase D1 cross-algorithm transfer ({tag}{'; PIPELINE CHECK ONLY' if tag == 'SYNTHETIC' or quick else ''})", "",
             f"Controller tuned on **{a.source_model}**, run unchanged on **{a.model}**. Scenarios: {len(specs)}.", "",
             "| System | Macro-F1 | Updates | Train seconds | Forgetting |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {n} | {r.macro_f1:.4f} | {r.updates:.2f} | {r.training_seconds:.3f} | {r.forgetting:.4f} |" for n, r in tb.iterrows()]
    ratio = summary["transfer_ratio_pooled"]
    lines += ["", f"- Transfer ratio (share of native gain over static kept): {'n/a' if ratio is None else f'{ratio:.2f}'}",
              f"- Action agreement at identical decision points: {summary['action_agreement']} over {summary['n_decision_points']} decisions",
              f"- Utility rank correlation: {summary['utility_rank_correlation']}",
              f"- H5: **{summary['H5']['status']}**. {summary['H5']['evidence']}", "",
              "Cost profiles and cost scales belong to the source algorithm; the gate and detector are the target's own. "
              "Full runs need six or more test seeds on REAL data before H5 can be called."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("macro_f1", "transfer_ratio_pooled", "action_agreement", "H5")}, indent=2))
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evox generalize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    ds = list(DATASETS)

    p = sub.add_parser("adapter-check", help="verify a dataset adapter on the real files")
    p.add_argument("--dataset", choices=ds, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--subsample-every", type=int, default=1)
    p.add_argument("--min-flows", type=int, default=120)
    p.add_argument("--json", type=Path, help="write the check results here")
    p.set_defaults(func=cmd_adapter_check)

    p = sub.add_parser("build-stream", help="build a processed stream for another dataset")
    p.add_argument("--dataset", choices=ds, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out-dir")
    p.add_argument("--subsample-every", type=int, default=1)
    p.add_argument("--reference-fraction", type=float, default=0.30)
    p.add_argument("--reference-val-fraction", type=float, default=0.20)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--force", action="store_true", help="build even if adapter checks fail")
    p.set_defaults(func=cmd_build_stream)

    p = sub.add_parser("transfer", help="cross-algorithm controller transfer (RQ6/H5)")
    p.add_argument("--source-controller", required=True, help="frozen_controller.json tuned on the SOURCE algorithm")
    p.add_argument("--source-model", choices=("random_forest", "xgboost"), required=True)
    p.add_argument("--target-run", required=True, help="finished Phase C run directory of the TARGET algorithm")
    p.add_argument("--model", choices=("random_forest", "xgboost"), required=True, help="TARGET algorithm")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--stream-dir", default=str(PROCESSED_DIR / "cicids2017_stream"))
    p.add_argument("--systems", nargs="*", default=["always_retrain"], choices=FIXED_POLICIES,
                   help="extra fixed policies run alongside static/native/transferred EVO-X")
    p.add_argument("--test-seeds", type=int, nargs="+")
    p.add_argument("--kinds", nargs="+")
    p.add_argument("--limit", type=int, help="run only the first N scenarios (pipeline check)")
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_transfer)

    p = sub.add_parser("report", help="combine Phase C runs into portability tables")
    p.add_argument("--run", action="append", required=True, metavar="DATASET/ALGORITHM=DIR")
    p.add_argument("--reference-dataset")
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--retain", type=float, default=0.5)
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_report)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sys.exit(main())
