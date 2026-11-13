# EVO-X real experiment run log

## Current status

Stopped during Step 2 (Phase A detector run). The data preflight passed and a real timestamp ordered stream was built. The full detector process was killed with exit code -9 while preparing validation scenarios, before producing detector comparison results or a tuned config. No Phase B/C/D experiments or paper fill were run.

## Commands

All times are Asia/Kolkata. Duration/exit information is taken from the per-command logs where recorded. One real-stream build log did not capture its start time or duration; its completion timestamp is the output file modification time.

| Step | Command | Start | End | Duration | Exit | Output / notes | data_tag | quick | test seeds |
|---|---|---|---|---:|---:|---|---|---|---|
| 0 | `.venv/bin/python -m pytest tests -q` | 2026-11-13 18:20:40 | 2026-11-13 18:21:18 | 38 s | 0 | `results/logs/step0_pytest.txt`; 257 passed, 1 skipped | N/A | N/A | N/A |
| 0 | Capture environment and `git rev-parse HEAD` | 2026-11-13 18:21:30 | 2026-11-13 18:21:30 | 0 s | 128 | `results/logs/environment.txt`; Python 3.12.14, 12 CPUs, 15.26 GiB RAM, Ubuntu 22.04; no Git metadata in workspace | N/A | N/A | N/A |
| 1 | Initial `check-data` with nested raw-dir | 2026-11-13 18:22:48 | 2026-11-13 18:22:49 | 1 s | 0 | `results/logs/step1_check_data_nested.txt`; old supplied CSVs had no Timestamp | N/A | N/A | N/A |
| 1 | `check-data --raw-dir data/raw/TrafficLabelling` after dataset was added | 2026-11-13 19:17:18 | 2026-11-13 19:17:19 | 0.7 s | 0 | `results/logs/step1_check_data_after_upload.txt`; all 8 CSVs parse and have Timestamp | REAL source | N/A | N/A |
| 1 | `build-stream --out-dir data/processed/cicids2017_stream` | 2026-11-13 19:12:22 | 2026-11-13 19:12:23 | 1 s | 0 | `results/logs/step1_build_stream.txt`; CLI searched top-level `data/raw/*.csv`, found none, and fell back to SYNTHETIC despite no `--synthetic`; created synthetic stream. Its fixed Phase 2 report paths under `results/metrics/` already existed and were written by this command. Preserved this stream after user selected a new real-stream path. | SYNTHETIC | N/A | N/A |
| 1 | `build-stream --out-dir data/processed/cicids2017_stream_real` with `EVOX_RESULTS_DIR=results/real_rebuild` | Not captured | 2026-11-13 19:14:53 | Not captured | 0 | `results/logs/step1_build_stream_real.txt`; copied the 8 input CSVs from `data/raw/TrafficLabelling/` to top-level `data/raw/` because the builder has no raw-dir option. 2,234,468 rows after dedup; all seven validations PASS; ordering timestamp. Reports isolated under `results/real_rebuild/`. | REAL | N/A | N/A |
| 2 | `detect --stream-dir data/processed/cicids2017_stream_real` with `EVOX_RESULTS_DIR=results/real_experiment_artifacts` | 2026-11-13 19:15:10 | 2026-11-13 19:16:30 | 80.1 s | -9 | `results/logs/step2_detect_real.txt`; process killed after preparing 25/56 validation scenarios. No phase6 detector comparison or tuned config was produced. | REAL | false | N/A |

## Hypotheses

H1-H5 statuses are unavailable: no qualifying hypothesis run files were generated. Phase A did not complete, so Phase B/C and transfer were not run.

## Red flags

Not evaluated. Phase A detector outputs were not produced, so gate behavior, updates, promotion outcomes, FPR, and NaN checks are unavailable.

## Skipped steps

- Steps 3-7 were not run because Phase A terminated with exit code -9 before producing the detector config needed by Phase B.
- RQ7 was not run. The workflow stopped before the cross-dataset stage; no `data/raw_unsw` input was present.
- Step 8 spot checks and final pytest rerun were not run because no paper results were produced.
- The workspace is not a Git checkout, so the requested commit hash is unavailable.

## Continuation: REAL run completed (2026-11-13)

The earlier log above is superseded by these completed outputs. Real stream: `data/processed/cicids2017_stream_real` (2,234,468 deduplicated rows; timestamp order; seven checks pass). Phase A OOM retries and the genuine memory fix are described above in the session; regression added to `tests/unit/drift/test_scenarios.py`.

| Command | Start (+05:30) | End (+05:30) | Duration | Exit | Output | Tag / quick / test seeds |
|---|---:|---:|---:|---:|---|---|
| Phase A detect, retry2 | 20:01:34 | 20:07:30 | 356.6s | 0 | `results/real_experiment_artifacts_retry2` | REAL / false / validation 3, test 6 |
| RF Phase B evolve | 20:07:55 | 20:25:55 | 1080.1s | 0 | `results/runs/rf_phase_b` | REAL / false / validation 3, test 6 |
| XGB Phase B evolve | 20:26:04 | 20:50:11 | 1446.8s | 0 | `results/runs/xgb_phase_b` | REAL / false / validation 3, test 6 |
| RF Phase C compare | 20:50:29 | 21:39:34 | 2944.9s | 0 | `results/runs/rf_phase_c` | REAL / false / test seeds 2001–2006 |
| XGB Phase C compare | 21:40:02 | 22:38:30 | 3508.1s | 0 | `results/runs/xgb_phase_c` | REAL / false / test seeds 2001–2006 |
| RF→XGB transfer | 22:39:05 | 23:08:44 | 1780s | 0 | `results/runs/phase_d_transfer_rf_to_xgb` | REAL / false / 6 seeds |
| paper-fill RF and XGB | after transfer | completed | <2s | 0 | `docs/reports/paper_results.md`, `paper_results_filled.md`, `paper_results_xgboost.md` | REAL / false / 6 seeds |
| pytest tests -q | 23:08:?? | 23:09:?? | 50.46s | 0 | test output | 258 passed, 1 skipped |

The Phase B actions were RF `no_op=200, incremental=18`; XGB `no_op=164, incremental=31, replay=7`. Neither was all No-Op. Both Phase C summaries report 114 test scenarios, 10 systems, 1,140 runs. RF promotions/rejections/rollbacks: 778/97/44. XGB: 603/110/73. H1, H2, H3, H4 are each `inconclusive` in both Phase C hypothesis files. H5 is `inconclusive`; evidence: “The native controller does not beat the static IDS, so there is no gain to transfer.” Transfer summary: REAL, quick false, 6 seeds, 114 scenarios.

## Paper fill and limitations

Both paper-fill commands exited 0. Section 5.6 retains its `[RESULT: pass --portability-dir with the Phase D portability run]` marker because UNSW-NB15 is absent (`data/raw_unsw` contains no files); RQ7 not run. The fill generated the requested RF filled copy and XGBoost results. Do not treat the remaining 5.6 marker as a completed portability result.

## Red flags / verification status

The prescribed end-to-end pytest passed (258 passed, 1 skipped). Phase C summaries have nonzero promotions and rejections. Transfer found EVO-X native macro-F1 0.73457 and static 0.73086, while H5 evidence states no native beat; report these as saved without reinterpretation. The required three paper-to-CSV spot-checks and comprehensive red-flag scan were not completed in this continuation; this is an outstanding verification limitation. Initial accidental synthetic build and prior `results/metrics` writes are documented above; real run artifacts use isolated paths.

### Step 8 spot checks and red flags (completed)

- `paper_results.md` RF summary agrees with `per_scenario.csv`: static macro-F1 0.700 (raw mean 0.700046); EVO-X 0.701 (0.700702).
- The RF paired-statistics table is generated from `paired_statistics.csv`; for EVO-X vs static, the exact pairwise row is not present because the file reports the registered baselines. The paired row `evox_no_gate` versus EVO-X has macro-F1 p=0.2500, Holm p=0.3750; the paper table reflects these registered comparisons.
- RF summary promotions/rejections agree: 778/97.
- Red flags across Phase C: no system has zero updates everywhere except static (expected comparator); EVO-X is not identical to static overall; numeric columns are NaN-heavy for recovery/censoring metrics (maximum numeric NaN fraction 84.2%, expected where events are censored/not observed; inspect before using those metrics); FPR values in paper are around 0.05, not near 0 or 1. Promotions and rejections are nonzero. Harmful promotions are present in summaries/tables, so the “all zero with no candidates” condition does not apply.
- UNSW absent: RQ7 not run. Its marker remains in the filled copy as a valid not-run section.
