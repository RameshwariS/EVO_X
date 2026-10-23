# Next phase — Phase 3: Static IDS (Synopsis §7.9 Phase 3, §8.3 Experiment 1) — 1.5 weeks

**Goal:** train Random Forest and XGBoost once on `ref_train`, never update, replay the live stream, and
measure how performance decays batch by batch. This is the "magnitude of the problem" baseline and also
produces the per-batch error/confidence logs that Phase 6 drift detectors will consume.

Prerequisite: Phase 2 Part B decision (reference fraction) is made.

## Tasks
1. **Model wrapper (do first; Phases 7–10 depend on it).** `src/ids_model.py` with a common interface:
   `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `clone()`, `save/load`. Keep incremental update (`update(X, y)`)
   as a stub now; XGBoost can continue from an existing booster via `xgb_model=`, RF cannot (use warm-start
   trees or the River alternative) — decide this now, it shapes Phase 7.
2. **Label handling gotcha.** XGBoost needs contiguous labels 0..k-1. Train only on classes present in
   `ref_train` (remap ids), and map predictions back. Anything a static model cannot name is, by design, wrong
   on novel classes — evaluate this explicitly (next point) rather than hiding it.
3. **Three evaluation views, per replay batch and as a rolling mean (e.g. 10 batches)** — many single batches
   contain one class, so per-batch macro-F1 is noisy:
   - *Binary* attack-vs-benign: precision, recall, F1, FPR, FNR (works for novel attacks: does the model at
     least flag them?).
   - *Multi-class on reference-known classes*: macro-F1, per-class recall.
   - *Novel-class detection rate*: share of novel-class flows flagged as attack.
   Use zero_division=0 and never report accuracy as the headline (§8.4).
4. **Log per-batch signals for Phase 6:** error indicator stream, max-probability (confidence), predicted class
   histogram. Save as `results/metrics/phase3_<model>_batch_log.csv`.
5. **Run:** RF and XGBoost × 5 seeds (`N_SEEDS` in config); report mean ± std. Outputs: degradation-curve figure
   (rolling F1 vs batch, with class-composition strip from Phase 2 underneath), summary table
   (pre-drift reference-val F1 vs stream F1), training time and inference latency (needed for the
   Evolution Controller's cost terms, §7.4).
6. **Tests:** wrapper round-trip; label remapping; metric functions on hand-computed toy cases.
7. **Report:** Experiment 1 section with the degradation figure and the honest statement of what part of the
   drop is novel-class vs covariate shift.

## Alongside (small, but flagged by your reviewer and in your README)
- Write the **Evolution Controller rule set as pseudocode** (drift magnitude/persistence thresholds → which of the
  5 strategies is a candidate, before utility scoring). Needed before Phase 8.
- **Pre-register the utility-weight procedure** (grid on a held-out validation stream, fixed before looking at
  test scenarios).
- Resolve the **HOIDS citation** (authors/venue/year) — it is a baseline in Phase 11.
- Reviewer's request: benchmark against SSF and HOIDS; plan their re-implementation effort now (Phase 7).

## Definition of done
Both models trained on `ref_train`; degradation curves and tables for 3 views × 2 models × 5 seeds; batch logs
saved; wrapper interface agreed; tests green. Then **Phase 4: Kafka replay** — producer reads `EvoStream.iter_batches()`
(topic `network-events`); consumer scores with the static model (topic `predictions`).
