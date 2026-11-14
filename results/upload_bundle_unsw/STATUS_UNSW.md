# UNSW-NB15 portability run status

## Commands and timing (host Asia/Kolkata)

| Step | Exact command | Start | End | Duration | Exit | Output |
|---|---|---|---|---:|---:|---|
| Initial tests | `.venv/bin/python -m pytest tests -q` | 2026-11-13 23:20:06 | 2026-11-13 23:20:38 | 32.37s | 0 | `tests` |
| Adapter check | `.venv/bin/python -m evox generalize adapter-check --dataset unsw-nb15 --raw-dir data/raw_unsw --json results/unsw_workspace/adapter_check.json` | 23:20:52 | 23:21:04 | 12s | 0 | `results/unsw_workspace/adapter_check.json` |
| Build stream | `.venv/bin/python -m evox generalize build-stream --dataset unsw-nb15 --raw-dir data/raw_unsw` | 23:21:28 | 23:22:06 | 38s | 0 | `data/processed/unsw_nb15_stream` |
| Detect (redirected) | `EVOX_RESULTS_DIR=$PWD/results/unsw_workspace .venv/bin/python -m evox detect --stream-dir data/processed/unsw_nb15_stream` | 23:22:21 | 23:28:02 | 341s | 0 | `results/unsw_workspace/metrics`, `figures` |
| RF Phase B | `.venv/bin/python -m evox evolve --stream-dir data/processed/unsw_nb15_stream --detector-config results/unsw_workspace/metrics/phase6_tuned_config.json --model random_forest --output-dir results/runs/unsw_rf_phase_b` | 2026-11-13T23:28:15+05:30 | 2026-11-13T23:42:07+05:30 | 832.0s | 0 | `results/runs/unsw_rf_phase_b` |
| RF Phase C | `.venv/bin/python -m evox compare --stream-dir data/processed/unsw_nb15_stream --model random_forest --controller-config results/runs/unsw_rf_phase_b/frozen_controller.json --output-dir results/runs/unsw_rf_phase_c` | 2026-11-13T23:50:56+05:30 | 2026-11-14T00:33:44+05:30 | 2568.0s | 0 | `results/runs/unsw_rf_phase_c` |
| Portability report | `.venv/bin/python -m evox generalize report --run CICIDS2017/random_forest=results/runs/rf_phase_c --run UNSW-NB15/random_forest=results/runs/unsw_rf_phase_c --output-dir results/runs/phase_d_portability` | 2026-11-14T00:34:15+05:30 | 2026-11-14T00:34:17+05:30 | 2.0s | 0 | `results/runs/phase_d_portability` |

Initial and final tests: 258 passed, 1 skipped. Adapter fix: none. Reference class labels in stream metadata: BENIGN, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms.

## Phase B / C run facts

- Phase B summary: data_tag `REAL`, quick `False`, algorithm `random_forest`, test seeds `[2001, 2002, 2003, 2004, 2005, 2006]`, test scenarios `114`, systems `['active_incremental', 'always_incremental', 'always_replay', 'always_retrain', 'evox', 'evox_no_gate', 'evox_rules_only', 'evox_utility_only', 'ssf_inspired', 'static']`, action counts `{'no_op': 193}`.
- Phase C summary: data_tag `REAL`, quick `False`, algorithm `random_forest`, test seeds `[2001, 2002, 2003, 2004, 2005, 2006]`, scenarios `114`, systems `['active_incremental', 'always_incremental', 'always_replay', 'always_retrain', 'evox', 'evox_no_gate', 'evox_rules_only', 'evox_utility_only', 'ssf_inspired', 'static']`.
- H1: `inconclusive`
- H2: `inconclusive`
- H3: `inconclusive`
- H4: `inconclusive`

## Portability report headline (verbatim)

UNSW-NB15 row: EVO-X F1 `0.5236`; Best-fixed F1 `0.5774`; Advantage `-0.0537`; Reference `-0.0494`; Verdict `none`; Cost ratio `0.00`.

## Spot checks

- Mean macro_f1 per_scenario: EVO-X `0.523641`, static `0.523641` (report headline EVO-X `0.5236`).
- paired_statistics static/macro_f1 row: `[{'baseline': 'static', 'metric': 'macro_f1', 'n_pairs': 6, 'mean_difference': 0.0, 'p': 1.0, 'holm_p': 1.0}]`.
- summary promotions/rejections: `640` / `109`.

## Red flags

- Zero-update totals by system: `{'active_incremental': 127, 'always_incremental': 127, 'always_replay': 127, 'always_retrain': 127, 'evox': 0, 'evox_no_gate': 0, 'evox_rules_only': 5, 'evox_utility_only': 0, 'ssf_inspired': 127, 'static': 0}`; this includes static and all four EVO-X ablations at 0 updates. EVO-X identical to static across all scenarios: `True`.
- Controller action selection: only `no_op` in Phase B.
- Gate rejection totals: `109` of 114 test scenarios (not zero or every scenario).
- Harmful promotions total for EVO-X: 0; counterfactual censoring records are present in the per-scenario file.
- Maximum numeric-column NaN share in per_scenario: 84.2%.
- Reference train has 10/10 encoded classes but is highly imbalanced (smallest class: 24 rows); reference validation has only class 0 (121,480 rows), so all 9 attack classes are absent from validation.

## Integrity

Pre-existing `results/metrics/` and `results/figures/`: all 27 copied files match their SHA256 baseline; no differences found. CICIDS processed stream SHA256 before/after build also matched. No CICIDS files restored because none differed.


Bundle size control: removed `runs/unsw_rf_phase_c/registries/` model-history JSONL files to meet the 25 MB upload limit; run outputs remain preserved in the original results directory. The required scenario CSVs, paired statistics, hypotheses and summary are included.
