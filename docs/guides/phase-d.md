# Phase D: generalisation, dashboard, demo and reproducibility

Phase D adds four commands. None of them changes the behaviour of Phases A–C; they read Phase B/C outputs or call
the existing Phase C simulation code unchanged.

| Command | Purpose | Code |
|---|---|---|
| `evox generalize adapter-check / build-stream` | Verify a dataset adapter on real files; build a stream for UNSW-NB15 or CSE-CIC-IDS2018 | `experiments/generalize.py` |
| `evox generalize transfer` | Cross-algorithm controller transfer (RQ6 / H5) | `experiments/generalize.py`, `evaluation/generalization.py` |
| `evox generalize report` | Cross-dataset / cross-algorithm portability tables (RQ7) | same |
| `evox dashboard serve / export` | Read-only dashboard over stored runs | `dashboard/` |
| `evox demo` | Scripted drift → decision → candidate → gate → rollback demo | `experiments/demo.py` |
| `evox reproduce` | One-command pipeline check with a manifest of versions, seeds and checksums | `experiments/reproduce.py` |

## D1. Cross-dataset portability (RQ7)

Run the existing Phase A–C workflow once per dataset, then combine the Phase C runs.

```powershell
# 0. Verify the adapter on the REAL files first (fails with exit code 1 and names the failing check)
python -m evox generalize adapter-check --dataset unsw-nb15 --raw-dir data/raw_unsw

# 1. Build the processed stream (writes phase_d_unsw_nb15_* reports; never touches the CICIDS2017 stream or phase2_* files)
python -m evox generalize build-stream --dataset unsw-nb15 --raw-dir data/raw_unsw

# 2. The normal pipeline on that stream (validation seeds tune, test seeds report)
python -m evox detect  --stream-dir data/processed/unsw_nb15_stream
python -m evox evolve  --stream-dir data/processed/unsw_nb15_stream --detector-config <detector.json> --output-dir results/runs/unsw_phase_b
python -m evox compare --stream-dir data/processed/unsw_nb15_stream --controller-config results/runs/unsw_phase_b/frozen_controller.json --output-dir results/runs/unsw_phase_c

# 3. Combine runs. The first --run (or --reference-dataset) is the reference dataset.
python -m evox generalize report `
  --run CICIDS2017/random_forest=results/runs/cicids_phase_c `
  --run UNSW-NB15/random_forest=results/runs/unsw_phase_c `
  --output-dir results/runs/phase_d_generalization
```

`DATASET` in `--run` is a free label you choose; the algorithm is checked against the run's protocol.

**What the report computes.** For every dataset × algorithm × drift kind (and pooled as `ALL`) it forms
EVO-X minus each baseline (each fixed policy, `static`, and `best_fixed` = the best fixed policy in hindsight for
each scenario, the strictest comparison), with seeds as the unit of replication, a paired Wilcoxon p-value only when
at least six seeds exist, the update and training-time ratio, and a non-inferiority flag (margin 0.02, as in Phase C).

**Verdicts** are relative to the reference dataset's advantage (margin 0.01, retain 0.5):

| Verdict | Meaning |
|---|---|
| `holds` | same direction, at least 50% of the reference size |
| `shrinks` | same direction, below 50% of the reference size |
| `absent` | reference had an advantage; here within ±margin |
| `reversed` | reference had an advantage; here a baseline is better by more than the margin |
| `emerges` | no advantage on the reference, one here |
| `none` | no advantage on either dataset (not evidence of portability) |
| `unknown` | a number is missing |

The advantage is never averaged across datasets. If any included run is synthetic, `--quick`, or has fewer than six
seeds the report is stamped **PIPELINE CHECK ONLY**.

**Known adapter limits.** The UNSW-NB15 adapter drops categorical columns (`proto`, `state`, `service`); the
datasets differ in features and attack families. Both adapters were exercised only on synthetic frames until you run
`adapter-check` on the real files, so treat that command as a prerequisite. `build-stream` refuses to build if a
check fails unless `--force` is given. CICIDS2017's 12-hour timestamp repair is switched off for other datasets.

## D1. Cross-algorithm transfer (RQ6 / H5)

Tune the controller on algorithm A (Phase B/C on A), then run it **unchanged** on algorithm B.

```powershell
python -m evox generalize transfer `
  --source-controller results/runs/rf_phase_c/frozen_controller.json --source-model random_forest `
  --target-run results/runs/xgb_phase_c --model xgboost `
  --stream-dir data/processed/cicids2017_stream --output-dir results/runs/phase_d_transfer_rf_to_xgb
```

* **Transferred:** weights, eligibility thresholds, cost/latency scales and cost profiles of A's frozen controller.
* **Target's own:** the Safety Gate thresholds, detector configuration, label policy, seeds and scenarios, all taken
  from B's finished Phase C run. The gate is a safety constraint selected on B's validation scenarios, not controller
  decision logic. The label policies of the two controllers must match or the command refuses.
* **Systems run** on identical scenarios: `static`, `evox_native` (B's own controller), `evox_transfer`, plus any
  `--systems` fixed policies (default `always_retrain`).
* **Decision-quality transfer** is reported two ways. *Downstream*: the transfer ratio, the share of the native
  controller's macro-F1 gain over static that the transferred controller keeps. *At the decision*: both controllers
  decide on identical proxy estimates at the same decision points (initial parent model, so trajectories cannot
  diverge) and we report action agreement and the Spearman correlation of utility scores.
* **H5 verdict.** `inconclusive` unless data is REAL, not quick, and at least six paired seeds. Then `supported` if
  the transfer ratio is at least 0.8 and the F1 gap to native is no worse than 0.02; `not_supported` if the ratio is
  below 0.5; otherwise `inconclusive`. If the native controller does not beat static there is no gain to transfer
  and the ratio is `null`.
* Source = target algorithm is accepted but flagged as a self-transfer sanity check, not RQ6.

## D2. Dashboard and demo

```powershell
python -m evox demo                        # runs the scripted scenario, writes results/runs/phase_d_demo/dashboard.html
python -m evox demo --serve                # ...and serves it on http://127.0.0.1:8000
python -m evox dashboard serve --runs-dir results/runs
python -m evox dashboard export --runs-dir results/runs --output results/runs/dashboard.html
```

The dashboard is **replay-driven**: it only reads finished runs (no live Kafka dependency). It is read-only (GET only),
binds to loopback by default, resolves run ids only from the discovered set, and escapes everything it displays. The
export is one HTML file with embedded data that opens from disk and sends no network requests.

Tabs: *Overview* (hypotheses, report, scripted-demo story), *Drift & decisions* (drift-state meters against the
controller thresholds, rule fired, eligible/excluded actions, utility terms per action, chosen action),
*Candidates & gate* (each version's gate checks with value vs bound), *Versions & rollback* (event timeline plus
journal, with a recomputed hash-chain verification badge), *Results* (per-system means, portability tables,
per-batch F1 with version-change markers). Synthetic or quick runs show a **PIPELINE CHECK ONLY** banner.

`evox demo` is honest about what is scripted: the registry events (promotion, rejection, rollback, exact
restoration) come from the real Phase C lifecycle code; the controller decision is made by the real controller but on
**scripted** drift-signature and proxy-estimate inputs, labelled as such in its record.

## D3. Reproducibility

```powershell
python -m evox reproduce --dry-run         # print the plan
python -m evox reproduce                   # synthetic quick pipeline check -> results/runs/phase_d_reproduce/
python -m evox reproduce --stream-dir data/processed/cicids2017_stream --detector-config results/metrics/phase6_tuned_config.json --output-dir results/runs/phase_d_reproduce_real
```

It writes `MANIFEST.json` (Python and package versions, git commit and dirty flag when available, seeds, arguments,
per-step status and wall time) and `SHA256SUMS`. Wall-clock columns (training seconds, latency) legitimately vary
between runs, so checksums of files containing them will differ; F1, update counts and decisions are deterministic for
fixed seeds, versions and hardware. Synthetic output never counts as a result.

See also: [limitations](../limitations.md), [viva preparation](../viva-prep.md), the
[Phase D report sections](../reports/PhaseD_report_sections.md) and the [paper skeleton](../reports/paper_draft.md).


## Filling the paper's Results section (`evox paper-fill`)

After the REAL, full-mode runs exist (Phase C comparison, Phase A detector tables, Phase D transfer/portability):

```powershell
.venv/Scripts/python.exe -m evox paper-fill `
  --compare-dir results/runs/phase_c_real --detect-dir results/metrics `
  --transfer-dir results/runs/phase_d_transfer_real --portability-dir results/runs/phase_d_portability_real `
  --paper docs/reports/paper_draft.md --output docs/reports/paper_results.md
```

Every number is read from the run files. The command **refuses to write** unless the Phase C run has `data_tag == REAL`,
was not `--quick`, and has at least six distinct test seeds (the detector, transfer and portability inputs are checked the same way).
It writes `paper_results.md` and a filled **copy** of the draft (`paper_results_filled.md`); the original draft is never edited.
`--allow-synthetic` previews the layout on a pipeline-check run: the output is stamped NOT REPORTABLE at the top and on
every subsection and must never be copied into the paper. Statuses of H1-H4 are copied verbatim from `hypotheses.json`.
