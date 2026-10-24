"""Paired seed-level statistics and cautious Phase C hypothesis reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_statistics(results: pd.DataFrame):
    """Aggregate scenarios within each seed before testing (no pseudoreplication)."""
    metrics = ("macro_f1", "updates", "forgetting", "harmful_promotions", "training_seconds")
    by_seed = results.groupby(["system", "seed"])[list(metrics)].mean()
    if "evox" not in results.system.unique():
        return pd.DataFrame(columns=["baseline", "metric", "n_pairs", "mean_difference", "p", "holm_p"])
    rows = []
    for baseline in sorted(set(results.system) - {"evox"}):
        for metric in metrics:
            a = by_seed.loc["evox", metric]
            b = by_seed.loc[baseline, metric]
            common = a.index.intersection(b.index)
            diff = a.loc[common].to_numpy() - b.loc[common].to_numpy()
            p = None
            if len(diff) >= 6:
                p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff, zero_method="pratt", method="auto").pvalue)
            rows.append(dict(baseline=baseline, metric=metric, n_pairs=len(diff),
                             mean_difference=float(np.mean(diff)) if len(diff) else None, p=p))
        a, b = by_seed.loc["evox", "macro_f1"], by_seed.loc[baseline, "macro_f1"]
        common = a.index.intersection(b.index)
        diff = a.loc[common].to_numpy() - b.loc[common].to_numpy()
        for name, alternative in (("macro_f1_noninferiority", "greater"), ("macro_f1_below_margin", "less")):
            shifted = diff + .02
            p = None if len(diff) < 6 else (1. if np.allclose(shifted, 0) else
                float(wilcoxon(shifted, zero_method="pratt", alternative=alternative, method="auto").pvalue))
            rows.append(dict(baseline=baseline, metric=name, n_pairs=len(diff),
                             mean_difference=float(np.mean(diff)), p=p))
    if not rows:
        return pd.DataFrame(columns=["baseline", "metric", "n_pairs", "mean_difference", "p", "holm_p"])
    table = pd.DataFrame(rows)
    table["holm_p"] = np.nan
    for metric in table.metric.unique():
        selected = table[(table.metric == metric) & table.p.notna()].sort_values("p")
        previous, count = 0., len(selected)
        for rank, (idx, row) in enumerate(selected.iterrows()):
            adjusted = max(previous, min(1., (count - rank) * row.p))
            table.loc[idx, "holm_p"] = adjusted
            previous = adjusted
    return table


def hypotheses(results: pd.DataFrame, paired: pd.DataFrame, data_tag: str, quick: bool):
    names = dict(H1="Strategy selection improves detection relative to fixed policies",
                 H2="Fewer updates while F1 stays within 0.02 of the best fixed policy",
                 H3="The gate reduces harmful promotions versus the no-gate ablation",
                 H4="Less historical forgetting than recent-only full retraining")
    report = {h: dict(claim=claim, status="inconclusive", evidence="") for h, claim in names.items()}
    if data_tag != "REAL" or quick or results.seed.nunique() < 6:
        for r in report.values():
            r["evidence"] = "Synthetic/quick runs or fewer than six paired seeds validate the pipeline only."
        return report
    fixed = [s for s in ("always_retrain", "always_incremental", "always_replay", "ssf_inspired", "active_incremental")
             if s in set(results.system)]
    means = results.groupby("system").mean(numeric_only=True)
    if "evox" not in means.index:
        return report
    def significant(baseline, metric, direction):
        rows = paired[(paired.baseline == baseline) & (paired.metric == metric)]
        return bool(len(rows) and pd.notna(rows.iloc[0].holm_p) and rows.iloc[0].holm_p < .05 and
                    direction * rows.iloc[0].mean_difference > 0)
    if fixed:
        supported = all(significant(s, "macro_f1", 1) for s in fixed)
        contradicted = any(significant(s, "macro_f1", -1) for s in fixed)
        report["H1"].update(status="supported" if supported else "not_supported" if contradicted else "inconclusive",
                            evidence="Seed-paired macro-F1 comparisons with Holm correction; see paired_statistics.csv.")
    if "always_retrain" in means.index and fixed:
        competitive = means.loc["evox", "macro_f1"] >= means.loc[fixed, "macro_f1"].max() - .02
        fewer = significant("always_retrain", "updates", -1)
        def margin_significant(system, metric):
            row = paired[(paired.baseline == system) & (paired.metric == metric)]
            return bool(len(row) and pd.notna(row.iloc[0].holm_p) and row.iloc[0].holm_p < .05)
        noninferior = all(margin_significant(s, "macro_f1_noninferiority") for s in fixed)
        inferior = any(margin_significant(s, "macro_f1_below_margin") for s in fixed)
        report["H2"].update(status="supported" if competitive and fewer and noninferior else
                            "not_supported" if inferior or significant("always_retrain", "updates", 1) else "inconclusive",
                            evidence=f"Descriptive F1 tolerance: {bool(competitive)}; Holm-adjusted non-inferiority: {noninferior}; significant update reduction: {fewer}.")
        report["H4"].update(status="supported" if significant("always_retrain", "forgetting", -1) else
                            "not_supported" if significant("always_retrain", "forgetting", 1) else "inconclusive",
                            evidence="Paired historical-anchor forgetting versus recent-only retraining.")
    if "evox_no_gate" in means.index:
        report["H3"].update(status="supported" if significant("evox_no_gate", "harmful_promotions", -1) else
                            "not_supported" if significant("evox_no_gate", "harmful_promotions", 1) else "inconclusive",
                            evidence="Finite-horizon frozen-parent counterfactual harmful-promotion counts; not a full alternate-policy replay.")
    return report


def plot_comparison(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (("macro_f1", "Reporting macro-F1"), ("forgetting", "Historical forgetting"),
               ("updates", "Promoted updates"), ("total_adaptation_seconds", "Training + proxy search (seconds)"))
    data = results.copy()
    data["total_adaptation_seconds"] = data.training_seconds + data.proxy_seconds
    mean = data.groupby("system").mean(numeric_only=True)
    std = data.groupby("system").std(numeric_only=True).fillna(0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (metric, title) in zip(axes.flat, metrics):
        ax.bar(mean.index, mean[metric], yerr=std[metric], capsize=2)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=55, labelsize=7)
    fig.suptitle("Phase C comparative replay (mean and spread across scenario/seed runs)")
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)
