# Phase B — Adaptation and decision core

Implements synopsis Phases 7–8 (Modules 7–9). Phase A remains the detector/scenario
foundation. Phase B runs in **shadow mode**: it creates candidates and decision
records; the deployed parent is never swapped. Promotion, rejection, rollback,
and the final comparative experiment belong to Phase C.

## Verification in this checkout

- Initial Phase B verification: **145 passed, 1 skipped**. After the package-layout
  migration, the complete suite is **158 passed, 1 skipped** (the real-Kafka
  integration test needs a broker).
- Random Forest quick run: `results/phase_b_verified/`; 14 scenarios, 14 detected
  events, 781 validation-only weight vectors. The selected policy chose No-Op
  for all events. This is recorded as a limitation, not a positive adaptation result.
- XGBoost quick run: `results/phase_b_demo/`; the same 14-scenario scope and 781
  weight vectors. It chose nine No-Ops, four strategic-selection candidates and
  one incremental candidate; five unpromoted model files are saved.
- Both runs cover all six drift types plus stationary controls, and freeze the
  controller before test scenarios are created. Every event remains `SHADOW_ONLY`.

These are **synthetic smoke checks using one seed per family**, not full-scale
benchmarks. The RF run's selected weights put all utility weight on latency;
the complete sensitivity table preserves that degenerate result. Real-data
validation and any policy revisions must be followed by a fresh frozen run,
without selecting changes based on held-out test performance.

## Run

PowerShell, from the repository root:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m pytest tests
.venv/Scripts/python.exe -m evox evolve --synthetic --quick --output-dir results/runs/phase_b_next_run
```

The quick run covers all six drift types and a stationary control, with one
validation seed and one test seed at medium magnitude. It preserves the default
500-flow batch size used by the Phase A scenario lab. It is a **pipeline check**,
not evidence for a research claim. Full runs use all three validation seeds, six
test seeds, and three magnitudes:

```powershell
.venv/Scripts/python.exe -m evox evolve --synthetic --output-dir results/runs/phase_b_synthetic
.venv/Scripts/python.exe -m evox evolve --stream-dir data/processed/cicids2017_stream --detector-config results/metrics/phase6_tuned_config.json --output-dir results/runs/phase_b_real --save-candidates
```

Real runs require an existing processed stream and a matching `REAL` Phase A
configuration. The runner refuses mismatched source tags/label delays and never
silently substitutes synthetic traffic. Existing output folders must be empty;
use a new output directory to preserve each experiment.

The `.venv` in this checkout is already installed. Earlier development smoke-run
folders remain in `results/`; cleanup was blocked by the execution policy. Use
`phase_b_verified/` and `phase_b_demo/` for the verified results described above.

Use `--model xgboost` for the second algorithm and `--transport kafka` to publish
decision records to the existing `evolution-events` topic (broker required).
In-memory event serialization/roundtrip is exercised by default.

## Components

| Component | File | Contract |
|---|---|---|
| Samples, labels, memory | `src/evox/adaptation/data.py` | Stable IDs, release times, deterministic label fraction, bounded class-balanced memory |
| Strategies and candidates | `src/evox/adaptation/strategies.py` | `apply(base_model, data, label_policy) -> CandidateModel` |
| Model update | `src/evox/models/ids.py` | RF prune-and-grow; XGBoost continuation |
| Proxy evaluation | `src/evox/evolution/proxy.py` | Separate training, proxy and tuning-audit holdouts |
| Controller | `src/evox/evolution/controller.py` | Eligibility rules, normalized utility, ablations, audit record |
| Experiment runner | `src/evox/experiments/evolve.py` | Phase A alarms/signatures -> estimates -> frozen controller -> shadow candidates |

Every non-No-Op candidate has a parent version, content-hashed training snapshot,
training IDs, algorithm/configuration, label policy, measured training time,
Python memory peak, serialized model size, latency and an initially empty metrics
slot. Candidate persistence is optional (`--save-candidates`).

## Adaptation semantics

- **No-Op:** returns the exact parent object with `NO_OP` status and no training.
- **Incremental RF:** scores old trees using released recent training labels,
  retains the best half minus one (at least one), and grows replacements on recent data. The tree count
  stays fixed. Each tree has its own global class map, so updates can add attack
  classes without misinterpreting old probability columns. This is a
  prune-and-grow approximation, not `partial_fit`.
- **Incremental XGBoost:** adds 20 boosting rounds to the existing booster.
  The complete existing class set must be represented in the update window.
  New/missing classes make this action unavailable; the controller logs the
  reason and can choose replay or a fresh fit. We do not pretend continuation
  can change the classifier's output dimension safely.
- **Replay:** fits a fresh model on recent labelled data mixed with bounded,
  class-balanced historical memory.
- **Strategic selection:** excludes stale streaming memory entries from its
  candidate's training pool (initial reference memory is retained), prioritizes error/uncertainty, preserves a
  representative of each available class, and fits a fresh candidate. This is
  **SSF-inspired**, not an exact reproduction of SSF.
- **Full retrain:** fits from scratch on the recent labelled window only.

The engine never mutates the parent. A rejected or discarded candidate therefore
does not need rollback to restore the parent; post-promotion rollback is Phase C.

## Data and label protocol

Default label delay is the Phase A value, two batches. A stream sample from batch
`t` may train or be evaluated only at `now >= t + delay`. Optional fractional
labels use deterministic sample-ID hashing and apply equally to every strategy.
Reference labels are initially available.

Phase A draws source flows with replacement. The scenario generator now also
returns source-pool IDs, without changing random draws or generated features.
All repeated appearances of an original source flow go into one role: 60%
training, 20% proxy holdout, 20% independent audit holdout. This grouping also
holds across offsets/regimes. Historical validation samples are split into
disjoint proxy/audit sets. Neither is used in candidate training or memory.
Only already-released samples in the recent window enter recent holdouts.

## Controller policy and pre-registration

The complete run protocol is saved **before scenario evaluation**. Initial rules
use threshold-normalized magnitude (low 1, high 3), persistence (short 2, long 10),
recurrence 0.9 and confidence-based unfamiliarity 0.2. These are declared policy
parameters, not claimed optimal thresholds. Confidence alone does not establish
novel-class drift; it only opens candidate eligibility.

Rules route insignificant/ended drift to No-Op, recurrence to replay/incremental,
high unfamiliarity plus high magnitude to retrain/selection, major persistent
drift to retrain/replay, and moderate drift to incremental/selection/replay.
No-Op remains a fallback. Unsupported candidates and candidates exceeding the
per-event training budget are excluded with explicit reasons.

Utility uses recent macro-F1 gain minus weighted historical macro-F1 loss,
normalized training cost, false-positive-rate increase and normalized added
inference latency. Equal utility prefers No-Op, then lower cost, then action name.
Rules-only chooses the first viable action in declared rule order; utility-only
removes the rule filter. All three ablations are recorded on identical episodes.

The weight procedure is fixed in advance:

1. Enumerate `{0, 0.25, 0.5, 1}^5`, omit the zero vector, normalize and deduplicate.
2. Train cheap candidates on capped training data. Estimate gain/forgetting/FPR
   on proxy holdouts; measure strategy cost and added latency on validation only.
3. Estimate full-candidate cost as median validation proxy seconds/row times the
   final training-row count. This is a linear approximation; actual candidate
   timing is reported separately. Freeze utility normalization scales too.
4. Select weights on validation seeds 1001–1003 by maximizing mean
   `0.5*(audit_recent_macro_f1 + audit_historical_macro_f1) - 0.05*normalized_cost`.
   Ties prefer lower estimated cost, then the first declared grid vector.
5. Save every grid result and freeze weights/profiles **before constructing test
   scenarios** (seeds 2001–2006). Test timing never updates the cost profile.

The tuning function refuses test-family episodes and test seeds, including a
test seed relabelled as validation. Quick mode uses the first seed of each family
and is explicitly recorded in every run's protocol/summary.

## Outputs

Under the selected output directory:

- `preregistered_protocol.json`, `detector_config.json`: exact protocol and detector snapshot.
- `frozen_controller.json`, `weight_sensitivity.csv`: selected policy, profiles and full grid.
- `evolution_events.jsonl`: append-only decisions with signatures, input context,
  estimates, exclusions, eligible actions, every utility contribution and candidate lineage.
- `proxy_profiles.csv`, `candidate_profiles.csv`: per-strategy/per-algorithm proxy
  measurements and actual chosen-candidate costs.
- `ablations.csv`: rules-only, utility-only and combined decisions, with independent
  audit scores of cheap proxy candidates.
- `scenario_coverage.csv`: includes scenarios with zero alarms.
- `summary.json`, `report.md`: run scope and results, explicitly tagged synthetic/real.
- `candidates/*.joblib`: optional full candidates, still unpromoted.

An **unnecessary update** is pre-defined as a candidate whose independent new-data
macro-F1 gain is less than 0.01. Phase C will evaluate full candidates on independent
data and also report gate-rejected attempts. Phase B proxy/audit scores alone are
not counts of harmful or unnecessary full-model deployments.

## Limits carried into Phase C

The cost model is an approximation; proxy search itself has overhead. Memory
profiles measure Python allocations (`tracemalloc`) and serialized model bytes,
not native-library peak RSS. Latency is a warm, three-repeat median on capped
batches and is machine-dependent. Scenario episodes keep the initial deployed
model fixed: candidates are generated in shadow mode rather than fed back into
the next drift event. Final evaluator, safety thresholds, promotion, rollback,
and end-to-end fixed-baseline comparisons remain Phase C work.
