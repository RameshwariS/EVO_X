"""
Fill the paper's Results section (§5) from saved experiment runs.

    python -m evox paper-fill --compare-dir results/runs/phase_c_real \
        [--detect-dir results/metrics] [--transfer-dir RUN] [--portability-dir RUN] \
        [--paper docs/reports/paper_draft.md] [--output docs/reports/paper_results.md]

Every number is read from the run files; nothing is typed in by hand.

Reportability gate (matches the paper draft's own rule: "do not paste synthetic or quick-run numbers"):
the tool REFUSES to write unless the Phase C run has data_tag == REAL, quick == false and at least six distinct
test seeds.  ``--allow-synthetic`` writes the file anyway, stamped NOT REPORTABLE on top and in every section,
so the layout can be checked before the real run exists.  With ``--paper`` the generated text replaces the
"## 5. Results" section of a COPY of the draft (written next to --output as *_filled.md); the original is never edited.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_TEST_SEEDS = 6
SYSTEM_ORDER = ["static", "always_retrain", "always_incremental", "always_replay", "ssf_inspired", "active_incremental",
                "evox", "evox_no_gate", "evox_rules_only", "evox_utility_only"]
ABLATIONS = ["evox", "evox_no_gate", "evox_rules_only", "evox_utility_only"]


# ------------------------------------------------------------------ helpers
def _load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _fmt(x, d=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def _ms(s: pd.Series, d=3) -> str:
    s = s.dropna()
    if s.empty:
        return "n/a"
    return f"{s.mean():.{d}f} ± {(s.std(ddof=1) if len(s) > 1 else 0.0):.{d}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _order(systems) -> list[str]:
    return [s for s in SYSTEM_ORDER if s in set(systems)] + sorted(set(systems) - set(SYSTEM_ORDER))


# -------------------------------------------------------- reportability gate
def check_compare_run(run: Path) -> tuple[list[str], dict]:
    """Return (problems, info).  Empty problems == reportable."""
    summary = _load_json(run / "summary.json")
    per = pd.read_csv(run / "per_scenario.csv")
    test = per[per.family == "test"]
    n_seeds = int(test.seed.nunique())
    problems = []
    if summary.get("data_tag") != "REAL":
        problems.append(f"data_tag is {summary.get('data_tag')!r}, not REAL")
    if summary.get("quick"):
        problems.append("run used --quick")
    if n_seeds < MIN_TEST_SEEDS:
        problems.append(f"only {n_seeds} distinct test seed(s); need >= {MIN_TEST_SEEDS}")
    return problems, dict(summary=summary, per=test, n_seeds=n_seeds)


# ----------------------------------------------------------------- sections
def section_detection(detect_dir: Path | None) -> tuple[str, list[str]]:
    probs = []
    if detect_dir is None or not (detect_dir / "phase6_detector_comparison.csv").exists():
        return "### 5.1 Detection\n\n[RESULT: detector tables not provided — pass `--detect-dir`]\n", probs
    d = pd.read_csv(detect_dir / "phase6_detector_comparison.csv")
    if (d.data_tag != "REAL").any():
        probs.append("detector comparison is not from a REAL run")
    rows = [[r.detector, f"{r.detection_rate:.3f}", f"{r.n_detected}/{r.n_events}", _fmt(r.mean_delay, 2),
             _fmt(r.median_delay, 1), f"{r.fa_per_1000:.2f}"] for r in d.itertuples()]
    txt = ["### 5.1 Detection", "",
           "Detection rate, delay (batches after the true change point) and false alarms per 1000 batches on test scenarios "
           f"(tolerance window as in §4; data tag: {', '.join(sorted(d.data_tag.unique()))}).", "",
           md_table(["Detector", "Detection rate", "Detected/events", "Mean delay", "Median delay", "FA / 1000"], rows), ""]
    w = detect_dir / "phase6_fusion_vs_single_wilcoxon.csv"
    if w.exists():
        wd = pd.read_csv(w)
        txt += ["Fusion versus single detectors (paired Wilcoxon over seeds):", "", wd.round(4).to_markdown(index=False)
                if hasattr(wd, "to_markdown") and _has_tabulate() else wd.round(4).to_string(index=False), ""]
    return "\n".join(txt), probs


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


def section_comparison(info: dict, run: Path) -> str:
    per = info["per"]
    systems = _order(per.system.unique())
    cols = [("macro_f1", "Macro-F1"), ("f1", "Binary F1"), ("fpr", "FPR"), ("forgetting", "Forgetting"),
            ("updates", "Updates"), ("rejected", "Rejected"), ("rollbacks", "Rollbacks"),
            ("harmful_promotions", "Harmful promotions"), ("training_seconds", "Train s")]
    cols = [c for c in cols if c[0] in per.columns]
    rows = [[s] + [_ms(per[per.system == s][c], 3 if c not in ("updates", "rejected", "rollbacks", "harmful_promotions") else 2)
                   for c, _ in cols] for s in systems]
    out = ["### 5.2 Main comparison", "",
           f"Mean ± std over {per.groupby(['scenario']).ngroups} test scenarios "
           f"({info['n_seeds']} test seeds, algorithm {info['summary'].get('algorithm')}). "
           "Each system maintains its own continually updated deployed model; label budget and data access are identical.", "",
           md_table(["System"] + [c[1] for c in cols], rows), "",
           "**Per drift type (macro-F1, mean over scenarios of that type)**", ""]
    kinds = sorted(per.kind.unique())
    piv = per.pivot_table(index="system", columns="kind", values="macro_f1", aggfunc="mean").reindex(systems)
    out += [md_table(["System"] + kinds, [[s] + [_fmt(piv.loc[s, k]) for k in kinds] for s in systems]), ""]
    ps = run / "paired_statistics.csv"
    if ps.exists():
        p = pd.read_csv(ps)
        rows = [[r.baseline, r.metric, int(r.n_pairs), _fmt(r.mean_difference, 4), _fmt(r.p, 4), _fmt(r.holm_p, 4)]
                for r in p.itertuples()]
        out += ["**EVO-X minus baseline, paired (Wilcoxon signed-rank, Holm-adjusted)**", "",
                md_table(["Baseline", "Metric", "Pairs", "Mean diff.", "p", "Holm p"], rows), ""]
    out += ["Figure: `comparison.png` in the run directory.", ""]
    return "\n".join(out)


def section_hypotheses(run: Path) -> str:
    h = _load_json(run / "hypotheses.json")
    rows = [[k, v["claim"], v["status"], v["evidence"]] for k, v in h.items()]
    return "\n".join(["### 5.3 Hypotheses H1–H4", "", md_table(["", "Claim", "Status", "Evidence"], rows),
                      "", "Statuses are taken verbatim from the run's `hypotheses.json`; negative and inconclusive outcomes are reported as findings.", ""])


def section_ablations(info: dict, run: Path) -> str:
    per = info["per"]
    sysn = [s for s in ABLATIONS if s in set(per.system)]
    rows = [[s, _ms(per[per.system == s].macro_f1), _ms(per[per.system == s].updates, 2),
             _ms(per[per.system == s].harmful_promotions, 2), _ms(per[per.system == s].unnecessary_updates, 2)
             if "unnecessary_updates" in per else "n/a"] for s in sysn]
    out = ["### 5.4 Ablations", "", md_table(["Variant", "Macro-F1", "Updates", "Harmful promotions", "Unnecessary updates"], rows), ""]
    cf = run / "counterfactuals.csv"
    if cf.exists():
        c = pd.read_csv(cf)
        c = c[c.system.isin(sysn)]
        if len(c):
            g = c.groupby("system").agg(candidates=("candidate_id", "count"), harmful=("harmful", "sum"),
                                        censored=("censored", "sum")).reindex(sysn).dropna(how="all").reset_index()
            out += ["Counterfactual replay of candidates (harmful = would have lowered F1 beyond tolerance if deployed):", "",
                    md_table(list(g.columns), [[r[0]] + [int(x) for x in r[1:]] for r in g.itertuples(index=False)]), ""]
    gs = run / "gate_sensitivity.csv"
    if gs.exists():
        g = pd.read_csv(gs)
        out += [f"Gate threshold sensitivity: {len(g)} validation configurations evaluated "
                f"(accepted candidates {int(g.n_accepted.min())}–{int(g.n_accepted.max())}; "
                f"harmful accepted {int(g.harmful_accepted.min())}–{int(g.harmful_accepted.max())}). Full grid: `gate_sensitivity.csv`.", ""]
    return "\n".join(out)


def section_transfer(d: Path | None) -> tuple[str, list[str]]:
    if d is None or not (d / "summary.json").exists():
        return "### 5.5 Cross-algorithm transfer (H5)\n\n[RESULT: pass `--transfer-dir` with the Phase D transfer run]\n", []
    s = _load_json(d / "summary.json")
    probs = []
    if s.get("data_tag") != "REAL" or s.get("quick"):
        probs.append("cross-algorithm transfer run is not REAL/full")
    out = ["### 5.5 Cross-algorithm transfer (H5)", "", "```json", json.dumps({k: s[k] for k in s if k in
           ("macro_f1", "transfer_ratio_pooled", "action_agreement", "H5", "data_tag", "quick")}, indent=2, default=str), "```", ""]
    return "\n".join(out), probs


def section_portability(d: Path | None) -> tuple[str, list[str]]:
    if d is None or not (d / "portability.csv").exists():
        return "### 5.6 Cross-dataset portability (RQ7)\n\n[RESULT: pass `--portability-dir` with the Phase D portability run]\n", []
    s = _load_json(d / "summary.json") if (d / "summary.json").exists() else {}
    probs = ["portability run is a pipeline check only"] if s.get("pipeline_check_only") else []
    p = pd.read_csv(d / "portability.csv")
    num = p.select_dtypes("number").round(4)
    p = pd.concat([p.select_dtypes(exclude="number"), num], axis=1)
    rows = [[("n/a" if (isinstance(v, float) and np.isnan(v)) else v) for v in r] for r in p.itertuples(index=False)]
    return "\n".join(["### 5.6 Cross-dataset portability (RQ7)", "", md_table(list(p.columns), rows), ""]), probs


# --------------------------------------------------------------------- main
def build_results(a) -> tuple[str, list[str]]:
    run = Path(a.compare_dir)
    problems, info = check_compare_run(run)
    det, p1 = section_detection(Path(a.detect_dir) if a.detect_dir else None)
    tr, p5 = section_transfer(Path(a.transfer_dir) if a.transfer_dir else None)
    po, p6 = section_portability(Path(a.portability_dir) if a.portability_dir else None)
    problems = problems + p1 + p5 + p6
    banner = ""
    if problems:
        banner = ("> **NOT REPORTABLE — pipeline check only. Reasons: " + "; ".join(problems) + ". "
                  "Do not copy any number below into the paper. Re-run on real data (full mode, ≥ 6 test seeds) and regenerate.**\n\n")
    body = "\n".join([det, section_comparison(info, run), section_hypotheses(run), section_ablations(info, run), tr, po])
    text = banner + "## 5. Results\n\n" + body
    if problems:  # stamp each subsection so a partial copy-paste still carries the warning
        text = text.replace("\n### ", "\n### [NOT REPORTABLE] ")
    return text, problems


def inject(paper: Path, results: str, out: Path) -> Path:
    src = paper.read_text(encoding="utf-8")
    m = re.search(r"^## 5\. Results.*?(?=^## 6\.)", src, flags=re.S | re.M)
    if not m:
        raise SystemExit("Could not find '## 5. Results' ... '## 6.' in the paper draft")
    new = src[:m.start()] + results.rstrip() + "\n\n" + src[m.end():]
    target = out.with_name(out.stem + "_filled.md")
    target.write_text(new, encoding="utf-8")
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-dir", required=True, help="Phase C run directory (evox compare output)")
    ap.add_argument("--detect-dir", help="directory with phase6_detector_*.csv (default: none)")
    ap.add_argument("--transfer-dir", help="Phase D cross-algorithm transfer run")
    ap.add_argument("--portability-dir", help="Phase D cross-dataset portability run")
    ap.add_argument("--paper", help="paper draft; a filled COPY is written (original untouched)")
    ap.add_argument("--output", default="docs/reports/paper_results.md")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="write even if the run is not reportable (output is stamped NOT REPORTABLE)")
    a = ap.parse_args(argv)
    text, problems = build_results(a)
    if problems and not a.allow_synthetic:
        print("REFUSING to write paper results — the run is not reportable:")
        for p in problems:
            print("  -", p)
        print("Use --allow-synthetic only to preview the layout (output will be stamped NOT REPORTABLE).")
        return 1
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out}" + ("  [NOT REPORTABLE]" if problems else ""))
    if a.paper:
        print(f"wrote {inject(Path(a.paper), text, out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
