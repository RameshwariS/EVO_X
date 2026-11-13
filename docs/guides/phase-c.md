# Phase C — Safety, lifecycle and comparative experiments

Implements synopsis Phases 9–11 (Modules 10–14): independent candidate evaluation,
a configurable Safety Gate, durable versioning/history, post-deployment rollback
and a continual comparison against fixed policies. Accepted models replace the
active model used on later batches. The MVP uses a file registry as permitted by
the roadmap; MLflow/PostgreSQL and live enterprise deployment remain extensions.

## Verification

- **207 tests passed, 1 Kafka integration test skipped.**
- `results/runs/phase_c_lifecycle_demo/`: promote v2, reject v3, restore v1;
  restored-parent predictions and durable active-pointer recovery verified.
- `results/runs/phase_c_verified/`: XGBoost, seven test scenarios × ten policies,
  48 validation gate configurations; 40 promotions, 5 rejections, 6 rollbacks.
- `results/runs/phase_c_rf_verified/`: Random Forest, the same scope; 35 promotions,
  6 rejections and no rollbacks. EVO-X's frozen RF policy selected No-Op; this
  negative/degenerate outcome is preserved rather than tuned away on test data.

Both comparisons are **synthetic quick checks**, using one seed per family and
medium drift magnitude. They demonstrate execution, not a research advantage.
The four research hypotheses are explicitly inconclusive in these runs.

## Run

Update the local package and run the tests:

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[dev,kafka]"
.venv/Scripts/python.exe -m pytest tests
.venv/Scripts/python.exe -m evox lifecycle-demo --output-dir results/runs/phase_c_demo_next
```

For a fresh comparison, first create a matching Phase B calibration:

```powershell
.venv/Scripts/python.exe -m evox evolve --synthetic --quick --model random_forest --output-dir results/runs/phase_b_for_c
.venv/Scripts/python.exe -m evox compare --synthetic --quick --model random_forest --controller-config results/runs/phase_b_for_c/frozen_controller.json --output-dir results/runs/phase_c_next
```

The controller's sibling `preregistered_protocol.json` and `detector_config.json`
provide the label policy and detector snapshot. New Phase B checkpoints also
embed their algorithm and label-policy metadata. Source tags, algorithms, batch
sizes and label delays must match. Existing output directories must be empty.

For full real-data experiments, use a real processed stream and matching REAL
Phase A/B calibration:

```powershell
.venv/Scripts/python.exe -m evox compare --model random_forest --stream-dir data/processed/cicids2017_stream --controller-config results/runs/phase_b_real/frozen_controller.json --output-dir results/runs/phase_c_real
```

There is no synthetic fallback. Full mode uses three validation seeds, six test
seeds and three magnitudes plus controls: 114 test scenarios × ten policies.
Use `--systems` for a subset, `--model xgboost` with its matching calibration,
and additional disjoint `--test-seeds` for more statistical power. Quick mode
uses one seed per family, medium magnitude and seven scenarios.

## Modules

| Component | Source |
|---|---|
| Full evaluation | `src/evox/evaluation/candidate.py` |
| Safety Gate and validation selection | `src/evox/evolution/safety.py` |
| Registry and append-only history | `src/evox/lifecycle/registry.py` |
| Promotion and rollback monitor | `src/evox/lifecycle/manager.py` |
| Continual policies | `src/evox/experiments/continual.py` |
| Comparative CLI | `src/evox/experiments/compare.py` |
| Paired statistics/hypotheses | `src/evox/evaluation/comparison.py` |

## Isolation and fairness

Repeated appearances of an original source flow stay in one role, even under
different drift offsets: 50% train, 15% proxy, 15% gate, 10% monitor and 10%
report. Historical validation samples are stratified into the four holdout roles.
Training, proxy/gate evaluation and monitoring respect delayed/fractional labels.
The evaluator rejects candidate/ancestor-training overlap and missing/unreleased
holdouts. Monitoring also excludes proxy-selection and gate-evaluation IDs.

All methods share the initial model, source roles, label access, windows and
alarms from a frozen initial sentinel IDS. Each method evolves its own active
model. Report labels are used retrospectively on independent reporting rows;
they never influence training, strategy selection, gates or rollback.

## Gate and registry

Full evaluation measures recent/historical macro-F1, per-class precision/recall,
F1/FPR/FNR, forgetting versus best historical scores and inference latency.
Promotion requires sufficient recent gain, bounded historical loss, bounded
forgetting/FPR increase, and training/latency budgets. Missing, nonfinite and
physically invalid metrics fail closed. Rejection leaves the active pointer
unchanged. The optional alternative-strategy retry is not enabled in this MVP.

The 48-point gate grid varies gain, historical-loss, forgetting, FPR and training
cost thresholds; latency uses the declared budget. Initial-parent validation
pilots and independent audit outcomes select thresholds by gain/retention minus
cost. Pilots are one-step estimates, not deployed results. The selected gate is
saved before constructing test scenarios. The Phase B controller stays frozen.

Each policy owns a file registry with versioned joblib models and an append-only,
flushed/fsynced, sequence-checked and hash-chained `history.jsonl`. Events record
snapshots, IDs, parameters, drift context, evaluations, decisions and pointer
changes. Model writes are atomic and artifacts are checksum-verified on load.
Journal replay reconstructs the active model; an exclusive lifetime OS lease
rejects competing owners. States include CANDIDATE, DEPLOYED, SUPERSEDED,
REJECTED and ROLLED_BACK. Damaged/incomplete journals fail closed and remain
intact for recovery; automatic journal repair is not implemented.

After promotion, the monitor compares the model to its frozen parent using new,
released monitor rows and an independent historical anchor. It requires two
batches and 32 rows by default, checks performance/FPR degradation, ignores
duplicate batches and restores the parent through an atomic `ModelHolder` swap.

## Policies and metrics

The default comparison includes static, always-retrain, always-incremental,
always-replay, SSF-inspired selection, active-incremental, EVO-X, no-gate,
rules-only and utility-only. Fixed policies deploy valid candidates without
gating/rollback. EVO-X variants use monitoring; no-gate removes only the
pre-deployment gate. All policies enforce data/label integrity. Unsupported
XGBoost class-expansion updates are recorded, not silently replaced by retraining.
Active-incremental selects the least-confident half of already-released labels;
it receives no extra oracle labels. Inspired baselines are approximations, not
exact implementations of SSF or the unresolved HOIDS reference.

Predictions are made before updates. Metrics cover detection, historical-anchor
forgetting/backward transfer, a first-reporting-appearance binary forward-transfer
proxy, recovery, latency, training/proxy wall and CPU time, sampled process RSS,
model size and lifecycle counts. The forward-transfer proxy is not a full
task-matrix FWT score. Recovery needs two attack-supported batches within 0.02
of pre-drift binary F1; unrecovered events are censored.

Frozen candidates and parents are shadow-scored on ten future reporting batches
after labels arrive. A future or historical macro-F1 loss over 0.03 defines harm;
a future gain below 0.01 defines an unnecessary promoted update. Empty horizons
are censored and truncated horizons marked. This is not a complete alternate
trajectory of all future policy decisions.

Statistics pair seed-level means rather than individual scenario rows. Wilcoxon
tests use Holm correction; H2 additionally tests non-inferiority with a predeclared
0.02 margin. Six seeds can be underpowered after correction. Synthetic, sparse
and quick experiments never establish H1–H4. Negative outcomes remain explicit.

Memory profiles include Python allocation peaks, model bytes and process RSS
sampled every 10 ms. RSS includes native/resident state of the surrounding
process and can miss shorter peaks; it is not isolated strategy allocation.

## Outputs and remaining research work

The run directory contains the protocol, code/source-data hashes, source snapshot
ZIP, frozen configs, validation pilots and full gate sensitivity grid, followed by:

- `per_scenario.csv`, `summary.csv`, `per_class.csv`, `batch_metrics.csv`;
- `candidate_events.csv`, `counterfactuals.csv`, `scenario_coverage.csv`;
- `paired_statistics.csv`, `hypotheses.json`, `summary.json`, `report.md`;
- `comparison.png` and per-scenario/per-policy `registries/` with full histories.

The implementation is verified with tests and synthetic checks. Full real-data,
multi-seed benchmarks and dissertation conclusions remain outstanding until the
matching real calibration and experiment matrix have been executed.
