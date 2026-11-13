# Producing real results and filling the paper (run on the machine that has the CICIDS2017 CSVs)

The synthetic/quick runs only prove the pipeline works (all hypotheses come out "inconclusive"). The paper needs REAL, full-mode runs
with ≥ 6 test seeds. `evox paper-fill` refuses anything less. Use fresh, empty output directories.

```powershell
# 0. data in data/raw/ (TrafficLabelling release with Timestamp)
python -m evox check-data
python -m evox build-stream                                   # -> data/processed/cicids2017_stream

# 1. Phase A: detectors tuned on validation scenarios, evaluated on test scenarios  -> results/metrics/phase6_*.csv
python -m evox detect --stream-dir data/processed/cicids2017_stream

# 2. Phase B: controller calibration, per algorithm
python -m evox evolve --stream-dir data/processed/cicids2017_stream --detector-config results/metrics/phase6_tuned_config.json --model random_forest --output-dir results/runs/rf_phase_b
python -m evox evolve --stream-dir data/processed/cicids2017_stream --detector-config results/metrics/phase6_tuned_config.json --model xgboost       --output-dir results/runs/xgb_phase_b

# 3. Phase C: the main comparison (full mode; hours) -- 114 test scenarios x 10 policies per algorithm
python -m evox compare --stream-dir data/processed/cicids2017_stream --model random_forest --controller-config results/runs/rf_phase_b/frozen_controller.json --output-dir results/runs/rf_phase_c
python -m evox compare --stream-dir data/processed/cicids2017_stream --model xgboost       --controller-config results/runs/xgb_phase_b/frozen_controller.json --output-dir results/runs/xgb_phase_c

# 4. Phase D: transfer (H5)
python -m evox generalize transfer --source-controller results/runs/rf_phase_c/frozen_controller.json --source-model random_forest --target-run results/runs/xgb_phase_c --model xgboost --stream-dir data/processed/cicids2017_stream --output-dir results/runs/phase_d_transfer_rf_to_xgb

# 5. Phase D: portability (UNSW-NB15; adapter-check FIRST -- adapters have only been tested on synthetic frames)
python -m evox generalize adapter-check --dataset unsw-nb15 --raw-dir data/raw_unsw
python -m evox generalize build-stream  --dataset unsw-nb15 --raw-dir data/raw_unsw
#   then detect / evolve / compare on data/processed/unsw_nb15_stream, then:
python -m evox generalize report --run CICIDS2017/random_forest=results/runs/rf_phase_c --run UNSW-NB15/random_forest=<unsw compare dir> --output-dir results/runs/phase_d_portability

# 6. Fill the paper (refuses non-REAL / quick / <6-seed inputs)
python -m evox paper-fill --compare-dir results/runs/rf_phase_c --detect-dir results/metrics --transfer-dir results/runs/phase_d_transfer_rf_to_xgb --portability-dir results/runs/phase_d_portability --paper docs/reports/paper_draft.md --output docs/reports/paper_results.md
```

Before pasting anything: open `paper_results.md`, confirm there is no "NOT REPORTABLE" banner, read the H1–H4 statuses as they are
(negative/inconclusive results are findings), and re-verify the literature gap and reference [3] ("HOIDS", unverified).
Check `generalize report` flags and the `--help` of each command if an option name differs on your version.
