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

Mean ± std over 114 test scenarios (6 test seeds, algorithm random_forest). Each system maintains its own continually updated deployed model; label budget and data access are identical.

| System | Macro-F1 | Binary F1 | FPR | Forgetting | Updates | Rejected | Rollbacks | Harmful promotions | Train s |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.700 ± 0.089 | 0.907 ± 0.056 | 0.051 ± 0.009 | 0.000 ± 0.000 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.000 ± 0.000 |
| always_retrain | 0.720 ± 0.069 | 0.925 ± 0.025 | 0.048 ± 0.017 | 0.208 ± 0.083 | 1.27 ± 0.60 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.09 ± 0.39 | 0.413 ± 0.190 |
| always_incremental | 0.744 ± 0.065 | 0.933 ± 0.026 | 0.041 ± 0.012 | 0.080 ± 0.075 | 1.27 ± 0.60 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.85 ± 0.73 | 0.255 ± 0.117 |
| always_replay | 0.728 ± 0.075 | 0.922 ± 0.035 | 0.048 ± 0.009 | 0.013 ± 0.009 | 1.27 ± 0.60 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.21 ± 0.41 | 0.573 ± 0.261 |
| ssf_inspired | 0.588 ± 0.102 | 0.807 ± 0.108 | 0.085 ± 0.087 | 0.449 ± 0.167 | 1.27 ± 0.60 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.16 ± 0.47 | 0.377 ± 0.176 |
| active_incremental | 0.744 ± 0.069 | 0.936 ± 0.026 | 0.036 ± 0.011 | 0.116 ± 0.092 | 1.27 ± 0.60 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.58 | 0.237 ± 0.109 |
| evox | 0.701 ± 0.089 | 0.908 ± 0.056 | 0.051 ± 0.009 | 0.000 ± 0.002 | 0.04 ± 0.18 | 0.04 ± 0.21 | 0.04 ± 0.18 | 0.01 ± 0.09 | 0.017 ± 0.059 |
| evox_no_gate | 0.701 ± 0.090 | 0.908 ± 0.056 | 0.051 ± 0.009 | 0.001 ± 0.003 | 0.08 ± 0.27 | 0.00 ± 0.00 | 0.08 ± 0.27 | 0.04 ± 0.21 | 0.017 ± 0.058 |
| evox_rules_only | 0.708 ± 0.083 | 0.915 ± 0.041 | 0.050 ± 0.010 | 0.003 ± 0.009 | 0.23 ± 0.46 | 0.70 ± 0.62 | 0.18 ± 0.40 | 0.10 ± 0.33 | 0.286 ± 0.230 |
| evox_utility_only | 0.703 ± 0.088 | 0.909 ± 0.055 | 0.050 ± 0.009 | 0.001 ± 0.006 | 0.12 ± 0.38 | 0.11 ± 0.31 | 0.10 ± 0.33 | 0.07 ± 0.26 | 0.046 ± 0.093 |

**Per drift type (macro-F1, mean over scenarios of that type)**

| System | abrupt | gradual | incremental | novel_class | recurring | stationary | temporary |
|---|---|---|---|---|---|---|---|
| static | 0.743 | 0.748 | 0.744 | 0.554 | 0.771 | 0.606 | 0.672 |
| always_retrain | 0.765 | 0.738 | 0.748 | 0.641 | 0.772 | 0.611 | 0.694 |
| always_incremental | 0.780 | 0.759 | 0.770 | 0.680 | 0.796 | 0.622 | 0.720 |
| always_replay | 0.779 | 0.757 | 0.770 | 0.638 | 0.785 | 0.607 | 0.678 |
| ssf_inspired | 0.633 | 0.677 | 0.621 | 0.514 | 0.556 | 0.554 | 0.539 |
| active_incremental | 0.778 | 0.761 | 0.769 | 0.670 | 0.800 | 0.619 | 0.726 |
| evox | 0.743 | 0.748 | 0.744 | 0.554 | 0.773 | 0.606 | 0.674 |
| evox_no_gate | 0.743 | 0.748 | 0.744 | 0.554 | 0.773 | 0.606 | 0.674 |
| evox_rules_only | 0.752 | 0.748 | 0.744 | 0.585 | 0.774 | 0.606 | 0.680 |
| evox_utility_only | 0.743 | 0.748 | 0.744 | 0.563 | 0.773 | 0.606 | 0.676 |

**EVO-X minus baseline, paired (Wilcoxon signed-rank, Holm-adjusted)**

| Baseline | Metric | Pairs | Mean diff. | p | Holm p |
|---|---|---|---|---|---|
| active_incremental | macro_f1 | 6 | -0.0431 | 0.0312 | 0.2812 |
| active_incremental | updates | 6 | -1.2368 | 0.0312 | 0.2812 |
| active_incremental | forgetting | 6 | -0.1157 | 0.0312 | 0.2812 |
| active_incremental | harmful_promotions | 6 | -0.9912 | 0.0312 | 0.2812 |
| active_incremental | training_seconds | 6 | -0.2202 | 0.0312 | 0.2812 |
| active_incremental | macro_f1_noninferiority | 6 | -0.0431 | 1.0000 | 1.0000 |
| active_incremental | macro_f1_below_margin | 6 | -0.0431 | 0.0156 | 0.1406 |
| always_incremental | macro_f1 | 6 | -0.0433 | 0.0312 | 0.2812 |
| always_incremental | updates | 6 | -1.2368 | 0.0312 | 0.2812 |
| always_incremental | forgetting | 6 | -0.0795 | 0.0312 | 0.2812 |
| always_incremental | harmful_promotions | 6 | -0.8421 | 0.0312 | 0.2812 |
| always_incremental | training_seconds | 6 | -0.2381 | 0.0312 | 0.2812 |
| always_incremental | macro_f1_noninferiority | 6 | -0.0433 | 1.0000 | 1.0000 |
| always_incremental | macro_f1_below_margin | 6 | -0.0433 | 0.0156 | 0.1406 |
| always_replay | macro_f1 | 6 | -0.0271 | 0.0312 | 0.2812 |
| always_replay | updates | 6 | -1.2368 | 0.0312 | 0.2812 |
| always_replay | forgetting | 6 | -0.0130 | 0.0312 | 0.2812 |
| always_replay | harmful_promotions | 6 | -0.2018 | 0.0312 | 0.2812 |
| always_replay | training_seconds | 6 | -0.5561 | 0.0312 | 0.2812 |
| always_replay | macro_f1_noninferiority | 6 | -0.0271 | 0.9844 | 1.0000 |
| always_replay | macro_f1_below_margin | 6 | -0.0271 | 0.0312 | 0.2188 |
| always_retrain | macro_f1 | 6 | -0.0194 | 0.0312 | 0.2812 |
| always_retrain | updates | 6 | -1.2368 | 0.0312 | 0.2812 |
| always_retrain | forgetting | 6 | -0.2078 | 0.0312 | 0.2812 |
| always_retrain | harmful_promotions | 6 | -1.0789 | 0.0312 | 0.2812 |
| always_retrain | training_seconds | 6 | -0.3955 | 0.0312 | 0.2812 |
| always_retrain | macro_f1_noninferiority | 6 | -0.0194 | 0.5000 | 1.0000 |
| always_retrain | macro_f1_below_margin | 6 | -0.0194 | 0.5781 | 1.0000 |
| evox_no_gate | macro_f1 | 6 | -0.0001 | 0.2500 | 0.3750 |
| evox_no_gate | updates | 6 | -0.0439 | 0.2500 | 0.3750 |
| evox_no_gate | forgetting | 6 | -0.0005 | 0.2500 | 0.3750 |
| evox_no_gate | harmful_promotions | 6 | -0.0351 | 0.2500 | 0.5000 |
| evox_no_gate | training_seconds | 6 | 0.0002 | 0.3750 | 0.3750 |
| evox_no_gate | macro_f1_noninferiority | 6 | -0.0001 | 0.0156 | 0.1406 |
| evox_no_gate | macro_f1_below_margin | 6 | -0.0001 | 1.0000 | 1.0000 |
| evox_rules_only | macro_f1 | 6 | -0.0075 | 0.0312 | 0.2812 |
| evox_rules_only | updates | 6 | -0.1930 | 0.0312 | 0.2812 |
| evox_rules_only | forgetting | 6 | -0.0029 | 0.0312 | 0.2812 |
| evox_rules_only | harmful_promotions | 6 | -0.0877 | 0.0625 | 0.2812 |
| evox_rules_only | training_seconds | 6 | -0.2691 | 0.0312 | 0.2812 |
| evox_rules_only | macro_f1_noninferiority | 6 | -0.0075 | 0.0156 | 0.1406 |
| evox_rules_only | macro_f1_below_margin | 6 | -0.0075 | 1.0000 | 1.0000 |
| evox_utility_only | macro_f1 | 6 | -0.0019 | 0.1250 | 0.3750 |
| evox_utility_only | updates | 6 | -0.0877 | 0.1250 | 0.3750 |
| evox_utility_only | forgetting | 6 | -0.0012 | 0.1250 | 0.3750 |
| evox_utility_only | harmful_promotions | 6 | -0.0614 | 0.1250 | 0.3750 |
| evox_utility_only | training_seconds | 6 | -0.0290 | 0.0312 | 0.2812 |
| evox_utility_only | macro_f1_noninferiority | 6 | -0.0019 | 0.0156 | 0.1406 |
| evox_utility_only | macro_f1_below_margin | 6 | -0.0019 | 1.0000 | 1.0000 |
| ssf_inspired | macro_f1 | 6 | 0.1126 | 0.0312 | 0.2812 |
| ssf_inspired | updates | 6 | -1.2368 | 0.0312 | 0.2812 |
| ssf_inspired | forgetting | 6 | -0.4488 | 0.0312 | 0.2812 |
| ssf_inspired | harmful_promotions | 6 | -1.1491 | 0.0312 | 0.2812 |
| ssf_inspired | training_seconds | 6 | -0.3596 | 0.0312 | 0.2812 |
| ssf_inspired | macro_f1_noninferiority | 6 | 0.1126 | 0.0156 | 0.1406 |
| ssf_inspired | macro_f1_below_margin | 6 | 0.1126 | 1.0000 | 1.0000 |
| static | macro_f1 | 6 | 0.0007 | 0.1250 | 0.3750 |
| static | updates | 6 | 0.0351 | 0.1250 | 0.3750 |
| static | forgetting | 6 | 0.0003 | 0.1250 | 0.3750 |
| static | harmful_promotions | 6 | 0.0088 | 1.0000 | 1.0000 |
| static | training_seconds | 6 | 0.0171 | 0.0625 | 0.2812 |
| static | macro_f1_noninferiority | 6 | 0.0007 | 0.0156 | 0.1406 |
| static | macro_f1_below_margin | 6 | 0.0007 | 1.0000 | 1.0000 |

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
| evox | 0.701 ± 0.089 | 0.04 ± 0.18 | 0.01 ± 0.09 | 0.01 ± 0.09 |
| evox_no_gate | 0.701 ± 0.090 | 0.08 ± 0.27 | 0.04 ± 0.21 | 0.03 ± 0.16 |
| evox_rules_only | 0.708 ± 0.083 | 0.23 ± 0.46 | 0.10 ± 0.33 | 0.05 ± 0.22 |
| evox_utility_only | 0.703 ± 0.088 | 0.12 ± 0.38 | 0.07 ± 0.26 | 0.04 ± 0.18 |

Counterfactual replay of candidates (harmful = would have lowered F1 beyond tolerance if deployed):

| system | candidates | harmful | censored |
|---|---|---|---|
| evox | 9 | 5 | 0 |
| evox_no_gate | 9 | 5 | 0 |
| evox_rules_only | 106 | 83 | 0 |
| evox_utility_only | 26 | 16 | 1 |

Gate threshold sensitivity: 48 validation configurations evaluated (accepted candidates 37–103; harmful accepted 1–17). Full grid: `gate_sensitivity.csv`.

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

