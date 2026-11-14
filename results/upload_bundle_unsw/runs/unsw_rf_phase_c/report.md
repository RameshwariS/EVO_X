# Phase C (REAL)

Algorithm: random_forest. Test scenarios: 114. Policies: 10.

| Policy | Macro-F1 | Forgetting | Updates | Train seconds | Proxy seconds |
|---|---:|---:|---:|---:|---:|
| active_incremental | 0.5682 | 0.0919 | 1.1140 | 0.2012 | 0.0000 |
| always_incremental | 0.5685 | 0.1019 | 1.1140 | 0.2064 | 0.0000 |
| always_replay | 0.5595 | 0.0448 | 1.1140 | 0.4829 | 0.0000 |
| always_retrain | 0.5335 | 0.2673 | 1.1140 | 0.3268 | 0.0000 |
| evox | 0.5236 | 0.0000 | 0.0000 | 0.0000 | 1.2962 |
| evox_no_gate | 0.5236 | 0.0000 | 0.0000 | 0.0000 | 1.2987 |
| evox_rules_only | 0.5238 | 0.0009 | 0.0439 | 0.2674 | 1.2961 |
| evox_utility_only | 0.5236 | 0.0000 | 0.0000 | 0.0000 | 1.2951 |
| ssf_inspired | 0.4054 | 0.3695 | 1.1140 | 0.3313 | 0.0000 |
| static | 0.5236 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Hypotheses

- H1: **inconclusive**. Seed-paired macro-F1 comparisons with Holm correction; see paired_statistics.csv.
- H2: **inconclusive**. Descriptive F1 tolerance: False; Holm-adjusted non-inferiority: False; significant update reduction: False.
- H3: **inconclusive**. Finite-horizon frozen-parent counterfactual harmful-promotion counts; not a full alternate-policy replay.
- H4: **inconclusive**. Paired historical-anchor forgetting versus recent-only retraining.

All policies share source-flow partitions, label access and sentinel drift triggers. Final scores use independent reporting rows; gate and monitoring data are separate. Fixed policies deploy valid candidates unconditionally and do not use rollback; EVO-X variants use the monitor. SSF-inspired and active-incremental policies are approximations, not reproductions of SSF/HOIDS. Counterfactuals compare a frozen candidate with its frozen parent for ten future batches. Resources include training/proxy CPU time, Python allocation peaks, serialized model sizes and 10ms sampled process RSS; RSS includes the surrounding process and may miss shorter peaks. Full real-data multi-seed runs are required before research claims.
