# Phase C report draft — Modules 10–14

## Candidate evaluation and Safety Gate (Modules 10–11)

Candidates are evaluated on recent and historical holdouts independently of
training and proxy strategy selection. Source-flow grouping prevents repeated
examples crossing data roles. All online decisions respect label delay/budget.
The evaluator measures per-class detection, recent gain, historical retention,
forgetting, false positives, training cost and latency. Promotion requires every
declared threshold; missing/invalid metrics and data-integrity violations reject
the candidate. Rejection preserves the deployed model.

Gate thresholds are selected from a preregistered validation grid using separate
audit outcomes, then frozen before test scenarios are constructed. The pilot
keeps its initial parent fixed. It is not the closed-loop experimental result.

## Registry, evolution history and rollback (Modules 12–14)

The file-based MVP stores versioned model artifacts and a flushed, hash-chained,
append-only journal. Events contain parent/version IDs, training snapshots and
IDs, parameters, drift context, evaluations, decisions and active-pointer changes.
Artifact checksums, stale-parent checks and an exclusive owner lease protect
lifecycle integrity. Replay reconstructs the deployed version; malformed or
incomplete journals fail closed without silent repair.

After promotion the in-memory holder changes atomically. Independent monitoring
compares fresh labelled traffic and historical retention against the frozen
parent, then restores it if necessary. Tests and the lifecycle demo demonstrate
promotion, rejection and prediction-identical rollback, including registry reopen.

## Comparative methodology and evidence

Static/fixed policies, inspired baselines, EVO-X and its ablations share initial
models, source groups, label budgets and sentinel alarms. Each owns a continually
updated deployed model. Fixed policies deploy valid candidates unconditionally;
EVO-X variants use post-deployment monitoring and, except no-gate, pre-deployment
safety checks. Inspired policies are approximations rather than exact SSF/HOIDS
reproductions.

Final metrics use reporting source groups excluded from decisions. They include
detection, historical forgetting/backward transfer, a binary novel-class forward
transfer proxy, censored recovery time, resource profiles and lifecycle counts.
Finite-horizon frozen-parent counterfactuals quantify harmful/unnecessary
promotions and harmful rejections. Their limits must be disclosed.

Scenarios are aggregated within each seed before paired tests. Holm-adjusted
Wilcoxon comparisons and a predeclared 0.02 non-inferiority margin support the
hypothesis assessment. Quick/synthetic checks remain explicitly inconclusive.

Insert actual results from the selected run's protocol, `per_scenario.csv`,
`summary.csv`, `paired_statistics.csv`, `hypotheses.json`, `counterfactuals.csv`
and `comparison.png`. Record hardware, algorithm, seeds, gate/controller configs,
code/source hashes and memory-measurement scope. Report negative results,
unsupported updates and censored outcomes. Full real-data benchmarks and final
H1–H4 conclusions are not claimed by the synthetic verification runs.
