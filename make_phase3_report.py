"""
EVO-X Phase 3 Part B -- turn the run outputs into the Experiment 1 report section.

    python make_phase3_report.py            # reads results/metrics/phase3_*.csv, writes docs/Phase3_report_section.md

Every number is read from the CSVs; the wording is conditional on the numbers (e.g. the novelty-vs-shift sentence
is only emitted when the data supports a decomposition).  If the run used the synthetic stream the output is
stamped as NOT REPORTABLE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "Phase3_report_section.md"


def _ms(g: pd.Series, d: int = 3) -> str:
    g = g.dropna()
    if g.empty:
        return "n/a"
    return f"{g.mean():.{d}f} ± {g.std(ddof=1) if len(g) > 1 else 0:.{d}f}"


def build_report(S: pd.DataFrame, W: pd.DataFrame) -> str:
    tag = S["data_tag"].iloc[0] if "data_tag" in S else "UNKNOWN"
    protocol = S["train_protocol"].iloc[0] if "train_protocol" in S else "unknown"
    n_seeds = S["seed"].nunique()
    L = []
    if tag != "REAL":
        L += ["> **NOT REPORTABLE — this output was produced from a " + tag + " stream. Re-run on the real CICIDS2017 stream "
              "and regenerate before using any number below.**", ""]
    L += ["# Experiment 1 — Static IDS on the time-ordered stream (Synopsis §8.3)", "",
          "## Setup",
          f"Random Forest and XGBoost were trained once ({protocol.replace('_', ' ')}) and then replayed over the live stream in "
          f"batches without any adaptation. Each configuration was repeated over {n_seeds} seeds; values are mean ± std. "
          "Accuracy is not reported (§8.4).", "",
          "## Results", "",
          "| Metric | " + " | ".join(sorted(S.model.unique())) + " |",
          "|---|" + "---|" * S.model.nunique()]
    rows = [("Trained classes", "n_known_classes", 0), ("Training rows", "n_train", 0),
            ("Reference-val F1 (binary)", "refval_bin_f1", 3), ("Reference-val FPR", "refval_bin_fpr", 4),
            ("Stream F1 (binary)", "stream_bin_f1", 3), ("Stream recall (binary)", "stream_bin_recall", 3),
            ("Stream FPR", "stream_bin_fpr", 4), ("Stream F1, known-class flows only", "stream_known_only_bin_f1", 3),
            ("Stream macro-F1 (known classes)", "stream_known_macro_f1", 3),
            ("Novel-class flows flagged as attack", "stream_novel_flag_rate", 3),
            ("Share of stream flows from novel classes", "stream_novel_share", 3),
            ("Training time (s)", "fit_seconds", 1), ("Inference latency (ms / 1k flows)", "latency_ms_per_1k_flows", 1)]
    for label, col, d in rows:
        if col in S:
            L.append(f"| {label} | " + " | ".join(_ms(S[S.model == m][col], d) for m in sorted(S.model.unique())) + " |")
    L.append("")

    # degradation over time (windowed)
    L += ["## Degradation over the stream", ""]
    for m in sorted(S.model.unique()):
        w = W[(W.model == m) & (W.bin_n_attack > 0)].groupby("window").agg(rec=("bin_recall", "mean"), f1=("bin_f1", "mean"))
        if w.empty:
            L.append(f"* **{m}**: no window contains attack flows — degradation cannot be measured.")
            continue
        worst = w.rec.idxmin()
        L.append(f"* **{m}**: binary recall ranges from {w.rec.max():.3f} (best window) to {w.rec.min():.3f} "
                 f"(window {worst}); {(w.rec < 0.5).mean():.0%} of attack-containing windows have recall < 0.5.")
    L += ["", "Figure: `results/figures/phase3_degradation.png` (binary recall/F1 per window, confidence, class composition).", ""]

    # novelty vs shift
    L += ["## How much of the drop is novelty, and how much is shift?", ""]
    ok = False
    for m in sorted(S.model.unique()):
        g = S[S.model == m]
        if g.stream_known_only_n_attack.mean() > 0 and g.stream_novel_share.mean() > 0:
            ok = True
            gap = g.stream_known_only_bin_f1.mean() - g.stream_bin_f1.mean()
            L.append(f"* **{m}**: F1 on flows of already-known classes is {g.stream_known_only_bin_f1.mean():.3f} versus "
                     f"{g.stream_bin_f1.mean():.3f} on the whole stream (difference {gap:+.3f}); "
                     f"{g.stream_novel_share.mean():.1%} of stream flows belong to classes absent from training, "
                     f"and {g.stream_novel_flag_rate.mean():.1%} of those are nevertheless flagged as attacks.")
    if not ok:
        L.append("The stream contains no attack flows of already-known classes (or no novel-class flows), so the drop cannot be "
                 "decomposed into novelty and shift; state this explicitly rather than inferring one.")
    L.append("")

    # paired comparison (§8.6)
    L += ["## RF vs XGBoost (paired over seeds, §8.6)", ""]
    try:
        from scipy.stats import wilcoxon

        a = S[S.model == "random_forest"].sort_values("seed").stream_bin_f1.to_numpy()
        b = S[S.model == "xgboost"].sort_values("seed").stream_bin_f1.to_numpy()
        if len(a) == len(b) and len(a) >= 2 and not np.allclose(a, b):
            p = wilcoxon(a, b).pvalue
            L.append(f"Wilcoxon signed-rank on stream binary F1 across {len(a)} seeds: p = {p:.3f}. "
                     + ("With only 5 seeds the smallest attainable two-sided p is 0.0625, so significance at 0.05 is impossible; "
                        "use more seeds if a significance claim is needed." if len(a) == 5 else ""))
        else:
            L.append("Paired test not applicable (models absent, unequal seeds, or identical values).")
    except Exception as e:  # pragma: no cover
        L.append(f"Paired test skipped ({e}).")
    L += ["", "## Threats to validity",
          "Static models cannot name unseen classes, so multi-class scores on novel flows are zero by construction; "
          "synthetic or re-sequenced drift (Phase 5) is needed to study drift of known classes. CICIDS2017 labelling artefacts "
          "(§9.3) and minute-resolution timestamps apply; results are relative comparisons, not absolute real-world performance.", ""]
    return "\n".join(L)


def main() -> int:
    sp, wp = METRICS_DIR / "phase3_summary_per_seed.csv", METRICS_DIR / "phase3_window_metrics.csv"
    if not (sp.exists() and wp.exists()):
        print("Run phase3_static_ids.py first.")
        return 1
    text = build_report(pd.read_csv(sp), pd.read_csv(wp))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text[:300] + "...\n-> wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
