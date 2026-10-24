# EVO-X — Remaining Work in 4 Phases (Detailed Plan)

**Project:** EVO-X: An Autonomous AI Model Evolution Framework for Continual Intrusion Detection under Concept Drift
**Basis:** Project Synopsis §4 (objectives, RQs, hypotheses), §6.1 (MVP vs extensions), §7 (methodology), §8 (experiments), §7.9 (timeline)
**Status:** Synopsis Phases 1–3 are built (code + tests). The four phases below group synopsis Phases **4–14**.

> Naming: to avoid confusion with the synopsis numbering, the four groups are called **Phase A, B, C, D** here.

---

## 0. Overview

| Phase | Synopsis phases | Planned time | Theme | Ends with something you can show |
|---|---|---|---|---|
| **A — Streaming & drift lab** | 4, 5, 6 | 5.5 wk | Replay the stream, create controlled drift, detect and characterise it | A reproducible drift lab with measured detector performance |
| **B — Adaptation & decision core** | 7, 8 | 4.5 wk | Produce candidate models and decide which strategy to use | A controller that picks a strategy and logs why |
| **C — Safety, lifecycle & experiments** | 9, 10, 11 | 6 wk | Evaluate candidates, gate them, version them, then run the main comparison | The main experimental result (EVO-X vs baselines) |
| **D — Generalisation, demo & write-up** | 12, 13, 14 | 5.5 wk | Test portability, build the dashboard, write everything up | Final report, paper draft, demo |

Total ≈ 21.5 weeks. Dependencies run strictly A → B → C → D (B needs A's detectors/scenarios; C needs B's candidates; D needs C's results).

### What already exists (Phases 1–3)
| Piece | Where | What it gives the later phases |
|---|---|---|
| Time-ordered stream (`EvoStream`): reference window + batches, leakage-free scaling, novel-class mask | `src/evox/data/stream.py` | The base data every scenario is built from |
| Stream validation + diagnostics | `src/evox/evaluation/stream.py` | Integrity checks, class timeline |
| Dataset adapters (UNSW-NB15, CSE-CIC-IDS2018) | `src/evox/data/adapters.py` | Cross-dataset runs (RQ7) — *unverified on real files* |
| `IDSModel` wrapper (RF, XGBoost), metrics, static experiment | `src/evox/models/ids.py`, `src/evox/evaluation/metrics.py`, `src/evox/experiments/static_ids.py` | The "current IDS", baseline numbers, per-flow error/confidence logs |
| Controller spec draft, citation notes | `docs/` | Starting point for Phase B |

### Carry-over items that must be done by you before Phase A–D can use real results
1. Run Phase 2 on **real CICIDS2017** (TrafficLabelling release with timestamps), decide the **reference window**.
2. Run Phase 3 on real data, decide the **training protocol** (train on `ref_train` only vs full reference).
3. Resolve the **"HOIDS" reference** (no paper by that name was found; see `docs/research/hoids-citation.md`).

### Rules that apply to every phase
- **No leakage:** statistics, thresholds and weights are fit on reference/validation data only; test scenarios are never used for tuning.
- **Two seed families:** a *validation* seed set (for choosing thresholds/weights) and a *test* seed set (for reporting). Never mix them.
- **Seeds & statistics:** ≥ 5 seeds, report mean ± std; paired Wilcoxon signed-rank for superiority claims (§8.6). With exactly 5 seeds the smallest possible two-sided p-value is 0.0625, so use ≥ 6 seeds (or pool across scenarios) if you want a significance claim at 0.05.
- **Honest reporting:** results where EVO-X does not beat a baseline are reported as findings (§8.5, §9.1). Synthetic-pipeline numbers are never reported as results.
- **Metrics:** never accuracy as a headline (§8.4). Report precision, recall, F1, FPR/FNR, forgetting, cost, latency, update counts.
- **Label policy (decide once, write it down):** adaptation needs labels. Choose and document: (a) oracle labels immediately, (b) labels delayed by *k* batches, or (c) a labelled fraction *f*. All methods, including baselines, get the same label budget.
- **Every phase ships:** code, unit tests, a short report section, and a "definition of done" check.

---

# PHASE A — Streaming & Drift Lab (5.5 weeks)

**Synopsis phases:** 4 (Kafka streaming), 5 (synthetic drift generator), 6 (drift detection) · **Modules:** 4, 2, 5, 6 · **Objectives:** 1, 2, 3
**Answers:** the groundwork for RQ1 (can measured drift characteristics guide strategy choice?)

## A1. Kafka streaming simulation (≈1.5 wk)
**Goal:** turn the in-memory replay loop into a decoupled producer/consumer pipeline *without changing any numbers*.

Tasks
1. `infra/docker/compose.yaml` with a single-node Kafka (KRaft mode). Topics: `network-events`, `predictions` now; `drift-events`, `evolution-events`, `model-events` as stubs.
2. **Producer** (`src/evox/streaming/producer.py`): publishes each replay batch from `EvoStream` (compressed float32 + metadata: batch index, row range, timestamps). Labels travel in a **separate field** so a consumer can withhold them (realistic label delay; used in Phase B). Options: `--rate` (batches/s), `--seed`.
3. **Consumer/scorer** (`src/evox/streaming/consumer.py`): scores with the deployed `IDSModel`, publishes predictions + confidence. A small `ModelHolder` lets the deployed model be swapped atomically later (Phase C).
4. **One partition** for `network-events` (ordering is the point); at-least-once delivery, idempotent on batch index.
5. Measure throughput (batches/s) and end-to-end latency for RF and XGBoost — these become the *Latency* term of the controller utility (§7.4).

Acceptance test (critical): **predictions through Kafka are bit-identical to the in-memory Phase 3 run with the same seed.** If not, find out why before building on it.

Outputs: compose file, producer/consumer, latency table, architecture figure update.
Risks: serialization dtype drift, partition reordering, Docker/RAM limits on student machines → keep an in-memory fallback so later phases never block on Kafka.

## A2. Synthetic concept-drift generator (≈2 wk)
**Goal:** create controlled drift scenarios with **known ground truth**, because the natural CICIDS2017 stream is almost entirely *novel-class* drift (attacks are weekday-scheduled — see Phase 2 findings). Recurring, gradual, incremental and temporary drift of *known* classes must come from here.

Six scenario types (Module 2):
| Scenario | Construction (from existing flows) | Ground truth recorded |
|---|---|---|
| **Abrupt** | Instant switch of class mix / feature regime at a change point | change point |
| **Gradual** | Probabilistic blend old→new over a transition window | start, end |
| **Recurring** | A previously seen mix reappears (A→B→A) | each change point + which concept |
| **Temporary** | Short-lived shift that reverts | start, end, revert point |
| **Incremental** | Small cumulative shifts (e.g. feature scaling/offset growing per batch) | slope, start |
| **Novel-class** | An attack family first appears at a known point | class id, first batch |

Tasks
1. Scenario specification as a dataclass/YAML: segments, mixing weights schedule, seed, length.
2. Generator that re-sequences/re-weights existing flows by class and (for incremental/covariate drift) applies bounded feature perturbations; writes an `EvoStream`-compatible object plus **`drift_truth.json`** (type, start, end, magnitude, affected classes/features).
3. **Be explicit about what each scenario is:** re-sequencing class proportions is *prior/label-distribution shift*; relabelling or feature perturbation is *synthetic* concept/covariate drift. Document this as a limitation (§9.3).
4. Separate **validation scenarios** (different seeds/draws) from **test scenarios** — the controller weights and gate thresholds are tuned only on the former.
5. Scenario library: each of the 6 types × several magnitudes/durations (e.g. low/med/high) × seeds.
6. Tests: truth files consistent with the stream; recurring scenario really repeats; reproducible from seed.

Outputs: generator, scenario library, truth files, a figure per scenario (class composition + shift curve).
Risks: generator makes drift that is trivially detectable or undetectable → calibrate magnitude levels against the static model's measured degradation.

## A3. Drift detection and drift analysis (≈2 wk)
**Goal:** detect drift reliably and characterise it for the controller (Modules 5, 6).

Detectors
| Detector | Signal | Needs labels? | Strength |
|---|---|---|---|
| **ADWIN** (River) | model error stream | **yes** | adaptive window, theoretical false-positive guarantee |
| **Page–Hinkley** | error stream or a summary statistic | yes (error mode) | lightweight, good at abrupt change |
| **PSI** | per-feature distribution vs reference (bins from reference quantiles; conventional 0.1 / 0.25 cut-offs) | **no** | interpretable, feature-level |
| **MMD** | multivariate kernel two-sample test (RBF, median-heuristic bandwidth, subsampled; permutation or calibrated threshold) | **no** | multivariate shift |

Fusion: `D_fused = Σ wᵢ dᵢ, Σ wᵢ = 1` on normalised scores; drift declared above a threshold. **Whether fusion reduces false alarms is a measured result, not an assumption** (§7.4).
Use the Phase 3 per-flow logs (error, confidence) as the detector input.

Drift Analyzer (Module 6) — output a `DriftSignature` per event:
`magnitude`, `duration`, `persistence` (consecutive batches above threshold), `type guess` (abrupt/gradual/recurring), `novelty` (share of low-confidence / unseen-class-like flows), `per-feature PSI vector`, and `recurrence` (similarity to stored past signatures).

Evaluation (on Phase A2 scenarios, using ground truth): **detection delay**, **false-alarm rate** (per 1000 batches), **missed detections**, with an explicit tolerance window; each detector alone vs fusion.

Tests: detectors on synthetic step/ramp signals with known change points; analyzer on scenario truth; no detector sees labels it shouldn't.
Risks: error-based detectors need labels (state the label policy); thresholds tuned on test scenarios (forbidden — tune on validation scenarios).

### Phase A — Definition of done
- Kafka replay reproduces Phase 3 predictions exactly; latency table recorded.
- All six scenario types generate reproducibly with ground-truth files.
- Detector comparison table (delay, false alarms, misses) per scenario, individual vs fused.
- `DriftSignature` produced for each detected event.
- Tests green; report sections for Modules 2, 4, 5, 6 drafted.

### Phase A — MVP vs extension
MVP: in-memory fallback + minimal Kafka, all six scenarios, ADWIN + Page–Hinkley + PSI, simple analyzer. Extension: MMD, full fusion, recurrence matching by signature database.

---

# PHASE B — Adaptation & Decision Core (4.5 weeks)

**Synopsis phases:** 7 (adaptation strategies), 8 (Evolution Controller) · **Modules:** 8, 9, 7 · **Objectives:** 4 (and 7 partially)
**Answers:** RQ1 (strategy selection vs fixed), RQ2 (fewer unnecessary updates), RQ4/RQ5 (forgetting and cost)

## B1. Adaptation Engine and candidate models (≈2.5 wk)
**Goal:** given a drift event and an action, produce a **candidate model** — never deployed automatically.

Strategies (action set A)
| Action | What it does | Notes |
|---|---|---|
| **No-Op** | no update | for noise / temporary shifts |
| **Incremental** | fine-tune on the latest batch/window | XGBoost: continue boosting from the existing booster; RF has no true `partial_fit` → warm-start added trees or prune-and-grow; implement `IDSModel.update()` per algorithm |
| **Memory Replay** | recent data mixed with a bounded historical buffer | buffer policy (reservoir or class-balanced), buffer size is a parameter |
| **Strategic Sample Selection** | prioritise informative drifted samples, discard stale buffer entries on drift (SSF-inspired) | also the basis of the SSF-inspired baseline; drop to extension if short on time |
| **Full Retrain** | train from scratch on a recent window (reserved for major, persistent drift) | the *naive* variant (recent data only) is what H4 compares against |

Tasks
1. `AdaptationStrategy` interface: `apply(base_model, data, label_policy) -> CandidateModel` (with `parent_version`, training-data snapshot id, metrics slot).
2. Implement the five strategies behind that interface; apply the **label policy** consistently (no hidden oracle labels).
3. **Cost profiling:** measure training time, memory, inference latency per strategy (and per algorithm) → a profile table used by the controller's Cost/Latency terms.
4. Unit tests: candidate has correct parent; Incremental ≠ Full Retrain; Replay buffer respects size and class balance; No-Op returns the parent unchanged.
Risks: RF incremental update is the awkward one — decide early and document; buffer growth; accidental use of future labels.

## B2. Evolution Controller (≈2 wk)
**Goal:** pick an action from the drift signature, model performance and past outcomes — the project's central, least-specified module (your external reviewer flagged this).

Two stages (see `docs/design/controller.md`)
1. **Rule-based eligibility filter** — which strategies are allowed given magnitude, persistence, novelty, recurrence and remaining budget (e.g. low magnitude or short persistence → No-Op only; recurrence high → Replay/Incremental; high novelty + high magnitude → Full Retrain / Strategic Selection; budget too low → drop Full Retrain). All thresholds are `[τ]` parameters.
2. **Utility ranking** among eligible actions (§7.4):
   `Utility(a) = w₁·PerformanceGain(a) − w₂·Forgetting(a) − w₃·AdaptationCost(a) − w₄·ΔFPR(a) − w₅·Latency(a)`, `a* = argmax`.
   Gain/forgetting/ΔFPR are estimated by a **cheap proxy evaluation on a small held-out sample**; cost/latency come from the profile table.

Tasks
1. Implement the controller with a transparent decision record (inputs, eligible set, each utility term, chosen action) written to `evolution-events`.
2. **Pre-register** the weight-selection procedure *before* looking at test scenarios: fixed grid (e.g. each wᵢ ∈ {0, 0.25, 0.5, 1}, normalised), chosen on **validation scenarios** by a stated criterion (e.g. macro-F1 − λ·normalised cost); report the full sensitivity grid.
3. **Ablations planned now:** rules-only, utility-only, rules + utility.
4. **Table-driven unit tests** for rule routing (every branch reachable and tested).
5. Define operationally what an **"unnecessary update"** is (e.g. candidate rejected by the gate, or ΔF1_new below τ₁ after promotion) — needed for RQ2.
6. Meta-controller / RL is **out of scope** for the MVP (§7.7).
Risks: proxy estimates are noisy and mislead the utility; weights look tuned to favour EVO-X (hence pre-registration).

### Phase B — Definition of done
- All strategies produce candidates for drifted batches from the Phase A scenarios.
- Strategy cost/latency profile table exists.
- Controller picks an action for each detected drift event and logs the full reasoning.
- Weight-selection procedure written down and executed on validation scenarios only.
- Tests green; report sections for Modules 7–9 drafted.

### Phase B — MVP vs extension
MVP: No-Op, Incremental, Replay, Full Retrain; rules + utility; one algorithm (RF or XGBoost). Extension: Strategic Selection, gradient-projection forgetting mitigation (RepShield-inspired), meta-controller.

---

# PHASE C — Safety, Lifecycle & Experiments (6 weeks)

**Synopsis phases:** 9 (candidate evaluation + Safety Gate), 10 (registry + rollback), 11 (comparative experiments) · **Modules:** 10, 11, 12, 13, 14 · **Objectives:** 5, 6, 7
**Answers:** RQ2, RQ3 (harmful updates), RQ4, RQ5, and hypotheses H1–H4

## C1. Candidate evaluation — Evolution Evaluator (≈0.75 wk)
Evaluate every candidate **before** it can be deployed:
- On **new/drifted held-out data** (ΔF1_new) and on **historical held-out data** (ΔF1_historical, per-class).
- Continual-learning metrics: **forgetting** (historical-class F1 drop), backward/forward transfer, recovery time.
- Safety/system metrics: **ΔFPR**, training cost, inference latency.
- Hold-out rule: evaluation samples are excluded from the candidate's training data; the historical set is a class-stratified memory so rare classes are not lost.
Tests: metric functions on hand-computed cases; no overlap between train and evaluation indices.

## C2. Safety Gate (≈0.75 wk)
Promotion policy (§7.4): promote only if **all** hold
`ΔF1_new ≥ τ₁`, `ΔF1_historical ≥ −τ₂`, `Forgetting ≤ τ₃`, `ΔFPR ≤ τ₄`, `Cost ≤ τ₅`; otherwise reject (and optionally let the controller try one alternative strategy, logging both).
Tasks
1. Configurable thresholds τ₁…τ₅ (parameters, not constants); selected on **validation scenarios**, with a sensitivity sweep reported.
2. **Counterfactual replay:** shadow-deploy rejected candidates on the stream to measure what *would* have happened — this is how **harmful promotions avoided** (H3) is measured.
3. Define **harmful update** precisely (e.g. a candidate that, if deployed, lowers overall or historical F1 by more than a tolerance).
Tests: each threshold failing alone → reject; all passing → promote; boundary values.

## C3. Model registry, versioning and rollback (≈1.5 wk)
- **Versioned registry:** v1 → v2 → … with status `DEPLOYED`, `REJECTED`, `ROLLED_BACK`; stores hyperparameters, training-data snapshot, evaluation metrics.
- **Rollback:** the active pointer reverts to the last known-good version; a post-promotion monitor triggers rollback if the promoted model underperforms (respecting the label policy).
- **Evolution history:** every event stored (timestamp, drift signature, strategy, candidate id, metrics, gate decision) — this also feeds the dashboard and future meta-learning.
- **MVP:** file-based registry (JSON + joblib) with the same interface. **Extension:** MLflow + PostgreSQL (+ Docker) behind that interface.
Tests: rollback restores *bit-identical predictions* of the previous version; history is append-only and complete.

## C4. Comparative experiments (≈3 wk) — the actual research result
**Goal:** answer "under what drift conditions, if any, does strategy selection beat a fixed strategy, and at what cost?" (§8.1).

Systems compared (§8.2)
Static IDS · Always Retrain · Always Incremental · Always Replay · SSF-inspired (strategic selection + forgetting, applied uniformly) · the "online + active learning" baseline (**HOIDS-inspired — rename/cite once the reference is resolved**) · **EVO-X (full pipeline)**.

Experiment grid: systems × 6 drift scenarios × magnitude levels × (RF, XGBoost) × seeds (test seed family) — start with CICIDS2017; other datasets are Phase D.

Metrics (§8.4)
| Category | Metrics |
|---|---|
| Detection | precision, recall, F1, FPR, FNR (overall and per class) |
| Continual learning | forgetting, forward/backward transfer, recovery time |
| System efficiency | training time, CPU/memory, inference latency, number of updates |
| Evolution quality | unnecessary updates, harmful updates, rejected candidates, rollbacks, successful-adaptation rate |

Required analyses
1. **Per-scenario results**, not only aggregates (H1 predicts the advantage is scenario-dependent, e.g. larger under recurring/mixed drift, small under simple abrupt drift).
2. **Ablations:** no safety gate; rules-only; utility-only; weight sensitivity; threshold sensitivity.
3. **Fairness:** identical label budget and data access for all systems; equal tuning effort for baselines.
4. **Statistics:** mean ± std over seeds, paired Wilcoxon signed-rank.
5. **Negative results** reported as findings.

Minimum viable experiment set (§6.1): static, always-retrain, always-incremental vs EVO-X, one dataset, one algorithm.

### Phase C — Definition of done
- Every candidate flows through evaluator → gate → registry; all decisions are logged.
- Rollback demonstrated and tested.
- Full result tables and figures for all systems × scenarios, with statistics and ablations.
- Hypotheses H1–H4 each marked supported / not supported / inconclusive with evidence.
- Report sections for Modules 10–14 and Chapter 8 results drafted.

### Phase C — Risks
Experiment grid too large (cut magnitudes/seeds before cutting scenarios); SSF/online-AL baseline re-implementations are approximate (call them "-inspired" and say so); the gate rejects everything or nothing (check the threshold sensitivity sweep early on validation scenarios).

---

# PHASE D — Generalisation, Demo & Write-up (5.5 weeks)

**Synopsis phases:** 12 (dashboard + end-to-end demo), 13 (cross-dataset / cross-model), 14 (report, paper, demo prep) · **Objectives:** 8, 9
**Answers:** RQ6 (algorithm portability), RQ7 (dataset portability), hypothesis H5

## D1. Generalisation experiments (≈1.5 wk)
- **Cross-algorithm (RQ6 / H5):** tune the controller on one algorithm, test on the other *without algorithm-specific tuning of the decision logic*; measure decision-quality transfer (utility-score correlation or downstream F1).
- **Cross-dataset (RQ7):** repeat the core comparison on **UNSW-NB15** and, if time permits, **CSE-CIC-IDS2018** through the existing adapters (verify the adapters on the real files first; categorical columns are dropped in the current UNSW adapter). Datasets differ in features and attack families, so this is a genuine generalisation test.
- Report where the advantage holds, shrinks or disappears; do not average it away.

## D2. Dashboard and end-to-end demo (≈2 wk)
Purpose: let a non-specialist reviewer *see* the system work (§10.3).
- **Backend:** FastAPI reading the evolution history/registry.
- **Frontend:** React (+ Plotly) showing: current model and version, live drift state, controller decisions with the computed utility scores, candidate evaluation results, safety-gate outcome, version/rollback timeline.
- **MVP:** a replay-driven demo reading stored results (no live Kafka dependency). **Extension:** live updates over WebSocket from Kafka topics, full Docker multi-service deployment.
- **Demo script:** one scripted scenario that shows drift → controller decision → candidate → gate reject/promote → rollback.

## D3. Report, paper draft and final preparation (≈2 wk)
- **Report:** finalise Chapters 3, 7, 8, 9 from the real results; keep the novelty claim *defensible* (§9.2) and **re-verify the literature gap** with a fresh search (the synopsis requires this); fix reference [3].
- **Limitations (§9.3):** synthetic drift construction, dataset artefacts/age, small set of base models, threshold sensitivity, no live deployment.
- **Paper draft:** problem, EVO-X architecture, experimental protocol, results with statistics, honest discussion.
- **Reproducibility package:** configs, seeds, one-command scripts, README, pinned requirements, Docker where used; archive result CSVs.
- **Final demo rehearsal** and viva Q&A preparation (expect: why rule-based and not RL; why these thresholds; what if the gate rejects everything; how do you know the drift is real).

### Phase D — Definition of done
Cross-model and cross-dataset results (or an explicit statement of what was not completed) · working demo · final report and paper draft consistent with the numbers · reproducibility package.

### Phase D — Where to cut if time is short (in this order)
1. Live Kafka-fed dashboard → replay-driven dashboard
2. MLflow/PostgreSQL/Docker → file-based registry
3. CSE-CIC-IDS2018 → UNSW-NB15 only
4. Neural-network IDS (never required)
5. Strategic Sample Selection / RepShield-style extensions

---

# Appendix 1 — Traceability: research questions and hypotheses → phase

| Item | Statement (short) | Measured in |
|---|---|---|
| RQ1 / H1 | Drift characteristics guide strategy better than a fixed one; advantage is scenario-dependent | A (signatures) → B (controller) → C4 per-scenario results |
| RQ2 / H2 | Fewer unnecessary updates at comparable F1 | B2 definition → C4 update counts vs always-retrain |
| RQ3 / H3 | Safety gate reduces harmful promotions | C2 counterfactual replay → C4 |
| RQ4 / H4 | Replay + gated evolution reduces forgetting vs naive retraining | B1 strategies → C1 forgetting metric → C4 |
| RQ5 | Lower cumulative cost than always-retrain | B1 cost profiles → C4 |
| RQ6 / H5 | Controller transfers across algorithms | D1 |
| RQ7 | Advantage holds across datasets | D1 |

# Appendix 2 — Open decisions to settle with Prof. Patil (and when)

| Decision | Needed before | Default proposal |
|---|---|---|
| Reference window size / protocol (natural stream is mostly novel-class drift) | A2 | Reference contains Benign + ≥ 2 attack families; known-class drift comes from the generator |
| Training protocol (`ref_train` only vs full reference) | A3 | One protocol everywhere, written down |
| Label policy (oracle / delayed / fraction) | A3, B1 | Delayed by k batches, same for all systems |
| Controller thresholds: fixed from theory or tuned on validation? | B2 | Tuned on validation scenarios only |
| Definition of "unnecessary" and "harmful" update | B2, C2 | As defined in B2.5 and C2.3 |
| Baseline naming/citation ("HOIDS") | C4 | Rename to the cited paper, or drop the name |
| Which extensions are in scope | start of each phase | MVP list in §6.1 of the synopsis |

# Appendix 3 — Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| Real-data runs take hours (RF × seeds × 1M+ rows) | all | per-class training cap, fewer seeds on validation, run overnight |
| Synthetic drift judged unrealistic | A2, D3 | state limitations; calibrate magnitude to measured degradation; keep natural novel-class scenario too |
| Controller proxy estimates are unreliable | B2 | evaluate proxy vs real candidate scores; ablate utility-only |
| Gate rejects everything / nothing | C2 | sensitivity sweep on validation scenarios first |
| EVO-X shows no advantage in some scenarios | C4 | that is a result; analyse *why* per scenario |
| Time overrun | all | cut list in Phase D; protect C4's 3 weeks |
| Baseline re-implementations differ from the original papers | C4 | label as "-inspired", document differences |
