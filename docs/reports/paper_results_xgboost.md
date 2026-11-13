## 5. Results

### 5.1 Detection

Detection rate, delay (batches after the true change point) and false alarms per 1000 batches on test scenarios (tolerance window as in §4; data tag: REAL).

| Detector | Detection rate | Detected/events | Mean delay | Median delay | FA / 1000 |
|---|---|---|---|---|---|
| adwin_err | 0.849 | 107/126 | 8.38 | 4.0 | 1.40 |
| ph_err | 0.730 | 92/126 | 9.32 | 4.0 | 5.24 |
| psi | 0.921 | 116/126 | 0.45 | 0.0 | 0.22 |
| mmd | 0.921 | 116/126 | 4.86 | 1.0 | 2.29 |
| adwin_conf | 0.929 | 117/126 | 1.03 | 0.0 | 0.37 |
| fused | 1.000 | 126/126 | 3.57 | 1.0 | 0.15 |

Fusion versus single detectors (paired Wilcoxon over seeds):

  fused_vs        metric  fused_mean  other_mean  wilcoxon_p  n_pairs
 adwin_err detected_frac      1.0000      0.8657      0.0002      108
 adwin_err  false_alarms      0.0185      0.1667      0.0008      108
    ph_err detected_frac      1.0000      0.7315      0.0000      108
    ph_err  false_alarms      0.0185      0.6481      0.0000      108
       psi detected_frac      1.0000      0.9074      0.0016      108
       psi  false_alarms      0.0185      0.0278      0.6547      108
       mmd detected_frac      1.0000      0.9074      0.0016      108
       mmd  false_alarms      0.0185      0.2870      0.1138      108
adwin_conf detected_frac      1.0000      0.9167      0.0027      108
adwin_conf  false_alarms      0.0185      0.0463      0.2568      108

### 5.2 Main comparison

Mean ± std over 114 test scenarios (6 test seeds, algorithm xgboost). Each system maintains its own continually updated deployed model; label budget and data access are identical.

| System | Macro-F1 | Binary F1 | FPR | Forgetting | Updates | Rejected | Rollbacks | Harmful promotions | Train s |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.731 ± 0.082 | 0.938 ± 0.048 | 0.032 ± 0.009 | 0.000 ± 0.000 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.000 ± 0.000 |
| always_retrain | 0.759 ± 0.061 | 0.953 ± 0.018 | 0.027 ± 0.011 | 0.211 ± 0.079 | 1.22 ± 0.58 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.09 ± 0.45 | 0.842 ± 0.426 |
| always_incremental | 0.751 ± 0.090 | 0.942 ± 0.049 | 0.030 ± 0.008 | 0.007 ± 0.011 | 0.57 ± 0.62 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.07 ± 0.26 | 0.153 ± 0.166 |
| always_replay | 0.777 ± 0.066 | 0.956 ± 0.022 | 0.026 ± 0.007 | 0.014 ± 0.009 | 1.22 ± 0.58 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.10 ± 0.30 | 1.755 ± 0.860 |
| ssf_inspired | 0.614 ± 0.104 | 0.846 ± 0.107 | 0.068 ± 0.090 | 0.421 ± 0.156 | 1.22 ± 0.58 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.13 ± 0.51 | 0.673 ± 0.265 |
| active_incremental | 0.739 ± 0.087 | 0.939 ± 0.049 | 0.031 ± 0.009 | 0.003 ± 0.009 | 0.19 ± 0.42 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.04 ± 0.21 | 0.047 ± 0.102 |
| evox | 0.735 ± 0.082 | 0.938 ± 0.049 | 0.032 ± 0.008 | 0.001 ± 0.005 | 0.14 ± 0.37 | 0.06 ± 0.24 | 0.10 ± 0.30 | 0.04 ± 0.23 | 0.095 ± 0.297 |
| evox_no_gate | 0.736 ± 0.083 | 0.939 ± 0.049 | 0.032 ± 0.008 | 0.002 ± 0.005 | 0.20 ± 0.44 | 0.00 ± 0.00 | 0.14 ± 0.37 | 0.04 ± 0.28 | 0.094 ± 0.292 |
| evox_rules_only | 0.734 ± 0.084 | 0.938 ± 0.049 | 0.032 ± 0.008 | 0.002 ± 0.005 | 0.16 ± 0.37 | 0.80 ± 0.57 | 0.11 ± 0.31 | 0.03 ± 0.16 | 0.696 ± 0.661 |
| evox_utility_only | 0.740 ± 0.080 | 0.940 ± 0.048 | 0.031 ± 0.008 | 0.003 ± 0.007 | 0.37 ± 0.50 | 0.11 ± 0.31 | 0.30 ± 0.46 | 0.04 ± 0.24 | 0.204 ± 0.366 |

**Per drift type (macro-F1, mean over scenarios of that type)**

| System | abrupt | gradual | incremental | novel_class | recurring | stationary | temporary |
|---|---|---|---|---|---|---|---|
| static | 0.733 | 0.757 | 0.774 | 0.601 | 0.806 | 0.674 | 0.733 |
| always_retrain | 0.785 | 0.769 | 0.789 | 0.695 | 0.808 | 0.674 | 0.735 |
| always_incremental | 0.774 | 0.789 | 0.801 | 0.602 | 0.819 | 0.674 | 0.748 |
| always_replay | 0.799 | 0.796 | 0.821 | 0.708 | 0.831 | 0.674 | 0.743 |
| ssf_inspired | 0.657 | 0.695 | 0.631 | 0.553 | 0.562 | 0.674 | 0.564 |
| active_incremental | 0.744 | 0.762 | 0.799 | 0.601 | 0.812 | 0.674 | 0.738 |
| evox | 0.749 | 0.757 | 0.774 | 0.603 | 0.811 | 0.674 | 0.734 |
| evox_no_gate | 0.750 | 0.757 | 0.774 | 0.603 | 0.812 | 0.674 | 0.738 |
| evox_rules_only | 0.748 | 0.757 | 0.774 | 0.601 | 0.812 | 0.674 | 0.734 |
| evox_utility_only | 0.755 | 0.763 | 0.781 | 0.616 | 0.815 | 0.674 | 0.734 |

**EVO-X minus baseline, paired (Wilcoxon signed-rank, Holm-adjusted)**

| Baseline | Metric | Pairs | Mean diff. | p | Holm p |
|---|---|---|---|---|---|
| active_incremental | macro_f1 | 6 | -0.0045 | 0.0625 | 0.2812 |
| active_incremental | updates | 6 | -0.0526 | 0.3125 | 0.6250 |
| active_incremental | forgetting | 6 | -0.0018 | 0.1562 | 0.3125 |
| active_incremental | harmful_promotions | 6 | -0.0088 | 1.0000 | 1.0000 |
| active_incremental | training_seconds | 6 | 0.0481 | 0.2188 | 0.4688 |
| active_incremental | macro_f1_noninferiority | 6 | -0.0045 | 0.0156 | 0.1406 |
| active_incremental | macro_f1_below_margin | 6 | -0.0045 | 1.0000 | 1.0000 |
| always_incremental | macro_f1 | 6 | -0.0167 | 0.0312 | 0.2812 |
| always_incremental | updates | 6 | -0.4298 | 0.0312 | 0.2812 |
| always_incremental | forgetting | 6 | -0.0060 | 0.0312 | 0.2812 |
| always_incremental | harmful_promotions | 6 | -0.0351 | 0.2500 | 1.0000 |
| always_incremental | training_seconds | 6 | -0.0576 | 0.1562 | 0.4688 |
| always_incremental | macro_f1_noninferiority | 6 | -0.0167 | 0.1562 | 0.4688 |
| always_incremental | macro_f1_below_margin | 6 | -0.0167 | 0.8906 | 1.0000 |
| always_replay | macro_f1 | 6 | -0.0426 | 0.0312 | 0.2812 |
| always_replay | updates | 6 | -1.0789 | 0.0312 | 0.2812 |
| always_replay | forgetting | 6 | -0.0122 | 0.0312 | 0.2812 |
| always_replay | harmful_promotions | 6 | -0.0614 | 0.1250 | 0.8750 |
| always_replay | training_seconds | 6 | -1.6595 | 0.0312 | 0.2812 |
| always_replay | macro_f1_noninferiority | 6 | -0.0426 | 1.0000 | 1.0000 |
| always_replay | macro_f1_below_margin | 6 | -0.0426 | 0.0156 | 0.1406 |
| always_retrain | macro_f1 | 6 | -0.0241 | 0.0312 | 0.2812 |
| always_retrain | updates | 6 | -1.0789 | 0.0312 | 0.2812 |
| always_retrain | forgetting | 6 | -0.2097 | 0.0312 | 0.2812 |
| always_retrain | harmful_promotions | 6 | -1.0526 | 0.0312 | 0.2812 |
| always_retrain | training_seconds | 6 | -0.7469 | 0.0312 | 0.2812 |
| always_retrain | macro_f1_noninferiority | 6 | -0.0241 | 0.7188 | 1.0000 |
| always_retrain | macro_f1_below_margin | 6 | -0.0241 | 0.3438 | 1.0000 |
| evox_no_gate | macro_f1 | 6 | -0.0011 | 0.0625 | 0.2812 |
| evox_no_gate | updates | 6 | -0.0614 | 0.0625 | 0.2812 |
| evox_no_gate | forgetting | 6 | -0.0003 | 0.0625 | 0.2812 |
| evox_no_gate | harmful_promotions | 6 | -0.0088 | 1.0000 | 1.0000 |
| evox_no_gate | training_seconds | 6 | 0.0010 | 0.5625 | 0.5625 |
| evox_no_gate | macro_f1_noninferiority | 6 | -0.0011 | 0.0156 | 0.1406 |
| evox_no_gate | macro_f1_below_margin | 6 | -0.0011 | 1.0000 | 1.0000 |
| evox_rules_only | macro_f1 | 6 | 0.0002 | 0.6875 | 0.6875 |
| evox_rules_only | updates | 6 | -0.0175 | 0.7500 | 0.7500 |
| evox_rules_only | forgetting | 6 | -0.0005 | 0.2188 | 0.3125 |
| evox_rules_only | harmful_promotions | 6 | 0.0088 | 1.0000 | 1.0000 |
| evox_rules_only | training_seconds | 6 | -0.6003 | 0.0312 | 0.2812 |
| evox_rules_only | macro_f1_noninferiority | 6 | 0.0002 | 0.0156 | 0.1406 |
| evox_rules_only | macro_f1_below_margin | 6 | 0.0002 | 1.0000 | 1.0000 |
| evox_utility_only | macro_f1 | 6 | -0.0057 | 0.0312 | 0.2812 |
| evox_utility_only | updates | 6 | -0.2281 | 0.0312 | 0.2812 |
| evox_utility_only | forgetting | 6 | -0.0020 | 0.0312 | 0.2812 |
| evox_utility_only | harmful_promotions | 6 | -0.0088 | 1.0000 | 1.0000 |
| evox_utility_only | training_seconds | 6 | -0.1091 | 0.0312 | 0.2812 |
| evox_utility_only | macro_f1_noninferiority | 6 | -0.0057 | 0.0156 | 0.1406 |
| evox_utility_only | macro_f1_below_margin | 6 | -0.0057 | 1.0000 | 1.0000 |
| ssf_inspired | macro_f1 | 6 | 0.1208 | 0.0312 | 0.2812 |
| ssf_inspired | updates | 6 | -1.0789 | 0.0312 | 0.2812 |
| ssf_inspired | forgetting | 6 | -0.4194 | 0.0312 | 0.2812 |
| ssf_inspired | harmful_promotions | 6 | -1.0965 | 0.0312 | 0.2812 |
| ssf_inspired | training_seconds | 6 | -0.5777 | 0.0312 | 0.2812 |
| ssf_inspired | macro_f1_noninferiority | 6 | 0.1208 | 0.0156 | 0.1406 |
| ssf_inspired | macro_f1_below_margin | 6 | 0.1208 | 1.0000 | 1.0000 |
| static | macro_f1 | 6 | 0.0037 | 0.0312 | 0.2812 |
| static | updates | 6 | 0.1404 | 0.0312 | 0.2812 |
| static | forgetting | 6 | 0.0014 | 0.0625 | 0.2812 |
| static | harmful_promotions | 6 | 0.0351 | 0.5000 | 1.0000 |
| static | training_seconds | 6 | 0.0953 | 0.0312 | 0.2812 |
| static | macro_f1_noninferiority | 6 | 0.0037 | 0.0156 | 0.1406 |
| static | macro_f1_below_margin | 6 | 0.0037 | 1.0000 | 1.0000 |

Figure: `comparison.png` in the run directory.

### 5.3 Hypotheses H1–H4

|  | Claim | Status | Evidence |
|---|---|---|---|
| H1 | Strategy selection improves detection relative to fixed policies | inconclusive | Seed-paired macro-F1 comparisons with Holm correction; see paired_statistics.csv. |
| H2 | Fewer updates while F1 stays within 0.02 of the best fixed policy | inconclusive | Descriptive F1 tolerance: False; Holm-adjusted non-inferiority: False; significant update reduction: False. |
| H3 | The gate reduces harmful promotions versus the no-gate ablation | inconclusive | Finite-horizon frozen-parent counterfactual harmful-promotion counts; not a full alternate-policy replay. |
| H4 | Less historical forgetting than recent-only full retraining | inconclusive | Paired historical-anchor forgetting versus recent-only retraining. |

Statuses are taken verbatim from the run's `hypotheses.json`; negative and inconclusive outcomes are reported as findings.

### 5.4 Ablations

| Variant | Macro-F1 | Updates | Harmful promotions | Unnecessary updates |
|---|---|---|---|---|
| evox | 0.735 ± 0.082 | 0.14 ± 0.37 | 0.04 ± 0.23 | 0.04 ± 0.18 |
| evox_no_gate | 0.736 ± 0.083 | 0.20 ± 0.44 | 0.04 ± 0.28 | 0.05 ± 0.26 |
| evox_rules_only | 0.734 ± 0.084 | 0.16 ± 0.37 | 0.03 ± 0.16 | 0.04 ± 0.18 |
| evox_utility_only | 0.740 ± 0.080 | 0.37 ± 0.50 | 0.04 ± 0.24 | 0.06 ± 0.24 |

Counterfactual replay of candidates (harmful = would have lowered F1 beyond tolerance if deployed):

| system | candidates | harmful | censored |
|---|---|---|---|
| evox | 23 | 5 | 0 |
| evox_no_gate | 23 | 5 | 0 |
| evox_rules_only | 109 | 83 | 0 |
| evox_utility_only | 54 | 7 | 0 |

Gate threshold sensitivity: 48 validation configurations evaluated (accepted candidates 63–91; harmful accepted 5–6). Full grid: `gate_sensitivity.csv`.

### 5.5 Cross-algorithm transfer (H5)

```json
{
  "data_tag": "REAL",
  "quick": false,
  "macro_f1": {
    "static": 0.7308550494567846,
    "evox_native": 0.7345677119830594,
    "evox_transfer": 0.7324883286675893
  },
  "transfer_ratio_pooled": null,
  "action_agreement": 0.9043209876543209,
  "H5": {
    "status": "inconclusive",
    "evidence": "The native controller does not beat the static IDS, so there is no gain to transfer."
  }
}
```

### 5.6 Cross-dataset portability (RQ7)

[RESULT: pass `--portability-dir` with the Phase D portability run]

