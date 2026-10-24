# EVO-X — Phase 3: Static IDS baseline (Synopsis §7.9 Phase 3, §8.3 Experiment 1)

## Part A — built and tested (this repo)
| File | Purpose |
|---|---|
| `src/evox/models/ids.py` | `IDSModel` wrapper for Random Forest / XGBoost: fit, predict (global class ids), predict_with_confidence, clone, save/load. Remaps labels internally (XGBoost needs contiguous 0..k-1). `update()` is a Phase 7 stub |
| `src/evox/evaluation/metrics.py` | binary (P/R/F1/FPR/FNR with support), known-class macro-F1, novel-class flag rate. No accuracy (§8.4) |
| `src/evox/experiments/static_ids.py` | trains once on `ref_train`, replays the stream, writes batch log, window metrics, summary table (mean ± std over seeds), per-flow prediction/confidence logs (input to Phase 6) and the degradation figure |
| `tests/unit/models/test_phase3.py` | 8 tests incl. hand-computed metric cases; 30 tests pass in total |

```bash
pytest tests -v
python -m evox static-ids --synthetic --seeds 2       # pipeline check only
python -m evox static-ids                             # uses data/processed/cicids2017_stream
python -m evox static-ids --seeds 5 --max-train-per-class 100000   # real data, bounded training time
```

### Three evaluation views (why)
A static model can never name a class absent from its training data, so one number would hide
what is going on:
1. **Binary** attack-vs-benign (does it at least flag novel attacks?)
2. **Known-class macro-F1** (only flows of classes the model was trained on — isolates covariate/concept shift)
3. **Novel flag rate** (share of novel-class flows flagged as attack)

### Gotchas found while building this
* `ref_val` is chronologically *after* `ref_train`, so it can contain classes the model never saw. Then
  `refval_bin_f1` is legitimately low. Use `refval_n_attack` and `n_known_classes` in the summary to interpret it,
  and consider a protocol where the model is trained on the whole reference (train+val) for final runs
  (decide with your guide; record it).
* A metric over a slice with no attacks is reported as NaN with its support, never as 0 or 1.
* Synthetic-stream numbers are pipeline checks only. Never put them in the report.

## Part B — what you do (needs the real stream from Phase 2 Part B)
1. Finish Phase 2 Part B first (real stream built, reference window decided).
2. `python -m evox static-ids --seeds 5 --max-train-per-class 100000` (drop the cap if RAM/time allow;
   RF 200 trees × 5 seeds on ~1M rows is slow — expect hours without the cap).
3. Open `results/figures/phase3_degradation.png` and `results/metrics/phase3_summary.csv`. Write down, for each model:
   pre-drift reference score, stream score, stream known-only score, novel flag rate, fit time, latency per 1k flows.
   The *known-only vs. whole-stream* gap is your honest answer to "how much of the drop is novelty vs. shift".
4. Report: Experiment 1 section (use figure + table), mention seeds and that results are relative, not absolute.
5. Commit code and `phase3_summary*.csv`; keep large `.npz` flow logs out of git (they are regenerable).
6. Do the three side tasks in `docs/design/controller.md` (rules pseudocode, utility-weight pre-registration, HOIDS citation).

## Definition of done
Real-data run for both models × 5 seeds · figure + table in report · flow logs saved · side tasks drafted · tests green.

# Next phase — Phase 4: Kafka streaming simulation (1.5 weeks) — see `docs/archive/phases/PHASE4_PLAN.md`
