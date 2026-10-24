# Phase A — Streaming & Drift Lab (synopsis phases 4, 5, 6)

**Status: code + tests complete; every number below comes from the SYNTHETIC stream and is a pipeline check, not a result.**
Tests: `pytest tests` → 105 passed, 1 skipped (the Kafka integration test, needs a broker).

## Definition-of-done checklist (from the plan)
| Item | State |
|---|---|
| Kafka replay reproduces Phase 3 predictions exactly | **Equivalence proven for the in-memory transport** (same wire format, same producer/scorer code, bit-identical predictions + confidences, also under injected duplicate delivery). **The real-Kafka run is NOT verified**: no Docker/broker in my sandbox. Run the 3 commands in "Run it" on your machine; the integration test is ready. |
| Latency table | `results/metrics/phase4_latency_memory.csv` (in-memory). Kafka table = run with `--transport kafka --bench`. |
| All six scenario types reproducible with ground-truth files | Yes (+ a `stationary` control). `data/processed/scenarios/*/drift_truth.json`, 166 files. |
| Detector comparison table (delay, false alarms, misses) per scenario, individual vs fused | Yes: `phase6_detector_comparison.csv`, `phase6_detector_by_scenario.csv`. |
| `DriftSignature` per detected event | Yes: `phase6_signatures.csv`. |
| Report sections for Modules 2, 4, 5, 6 | `docs/reports/PhaseA_report_sections.md` (draft; fill numbers from the real run). |

## What was built
| File | Module |
|---|---|
| `infra/docker/compose.yaml`, `src/evox/streaming/transport.py` (InMemory + Kafka), `stream_codec.py`, `stream_producer.py`, `stream_consumer.py` (`ModelHolder`, idempotent `StreamScorer`), `stream_pipeline.py`, `src/evox/experiments/replay.py` | 4 |
| `src/evox/drift/scenarios.py`, `src/evox/experiments/scenarios.py` | 2 |
| `src/evox/drift/detectors.py` (ADWIN, Page–Hinkley, PSI, MMD, ADWIN-on-confidence, fusion), `src/evox/evaluation/drift.py`, `src/evox/experiments/detect.py` | 5 |
| `src/evox/drift/analyzer.py` (`DriftSignature`, `SignatureDB`) | 6 |
| `src/evox/config.py` additions: topics, `LABEL_DELAY_BATCHES=2`, `VALIDATION_SEEDS`, `TEST_SEEDS` | — |

## Run it
```bash
pip install -r requirements.txt && pytest tests
python -m evox replay run --transport memory --synthetic --check-equivalence --bench     # no Docker
docker compose -f infra/docker/compose.yaml up -d
python -m evox replay run --transport kafka --create-topics --check-equivalence --bench  # exit code 1 if not bit-identical
pytest -m integration                                                                       # needs the broker
python -m evox scenarios          # real stream if data/processed/cicids2017_stream exists, else add --synthetic
python -m evox detect              # ~4 min on synthetic
```
For clean Kafka re-runs use `docker compose -f infra/docker/compose.yaml down -v` first (topics keep old messages).

## Decisions I made (change if Prof. Patil disagrees)
* **Label policy:** labels for batch *t* arrive when batch *t+2* is scored. Error detectors (ADWIN-err, PH) pay that delay; PSI/MMD/ADWIN-conf do not.
* **Scenario source:** flows pooled and split *per class* (random within class → clean ground truth) into train / val / two **disjoint** pools (validation vs test). Features re-standardised on lab-train only. This is *not* chronological; a `within_class="chronological"` option keeps natural drift.
* **Concept choice independent of magnitude** (low/med/high = same drift, different strength).
* **ADWIN is fed per-flow error bits** — fed batch means it never fired (see `test_detectors.py` docstring).
* **Tuning protocol pre-registered** in `src/evox/experiments/detect.py` header; frozen config written to `phase6_tuned_config.json` *before* test scenarios are built. Criterion `J = det_rate − 0.02·FA/1000 − 0.002·delay`.
* Scoring: tolerance 10 batches; return-to-A events are unscored (alarms there are neutral).

## Synthetic-run findings worth carrying to the real run (and what they do NOT show)
1. **Fusion looks perfect but the tuned fusion is just PH ∨ PSI** (weights 0.25/0.25/0/0, θ=0.5 → either member at threshold alarms). 126/126 events, 0 false alarms, p<0.05 vs every single detector — on easy synthetic data tuned near ceiling. Do not claim "fusion reduces false alarms" from this; it is a hypothesis for the real run. Also note PSI alone has 0 false alarms and delay 1.3.
2. **Detectors are complementary:** PSI/MMD miss many novel-class events (0.67 detection) because an 8% new class barely moves marginals; error-based ADWIN catches them (1.00) but only after labels arrive.
3. **Label delay costs error detectors ~1 batch per batch of delay** (mean delay 2.6→7.2 for k=0→5).
4. **Analyzer weaknesses (honest):** type-guess accuracy 0.67; **all 18 temporary events guessed "abrupt"** (the `temporary` rule needs the revert to be visible within the 8-batch look-ahead and it usually is not); gradual recall 50%. Magnitude ranks well (Spearman 0.90–0.94); onset estimate median error 0; recurrence 1.00 vs 0.00 (trivially separable here: same B reappears).
5. **`novelty` (low-confidence share) is not novelty-specific:** covariate shift raises it more than a new class does (0.85 vs 0.03). Phase B rules must not read high novelty as "new class".
6. **Synthetic data is too easy:** RF error stays <1% pre-drift and PSI is 10× threshold post-drift; magnitude calibration (`phase5_magnitude_calibration.csv`) must be redone on real CICIDS2017 before the levels are trusted. MMD false alarms ≈2.5/1000 batches at the default multiplier (expected from its null-quantile threshold).

## Only you can do
1. Run Phases 2–3 on real CICIDS2017 and settle the reference window so it contains Benign + ≥2 attack families (plan Appendix 2).
2. Run the Kafka commands above and paste the latency table.
3. Re-run `src/evox/experiments/scenarios.py` and `src/evox/experiments/detect.py` on the real stream; re-read finding 6 and the calibration table; only then quote numbers.
4. Agree the label policy (k) and the tuning criterion with Prof. Patil.

## Next: Phase B (adaptation engine + Evolution Controller). Inputs it will consume: `DriftSignature`, `ModelHolder`, the frozen detector config, scenario library.
