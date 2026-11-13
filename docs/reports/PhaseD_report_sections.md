# Phase D report draft — generalisation, demonstration and reproducibility

Source of truth for every number below: the run directories named in each section. No figure in this draft is a
result until the cited run is REAL, not `quick`, and has at least six test seeds.

## Generalisation across algorithms and datasets (RQ6, RQ7, H5)

**Cross-algorithm (RQ6 / H5).** A controller tuned on one base model (weights, eligibility thresholds, cost scales and
cost profiles, frozen on validation scenarios) is run unchanged on the other base model over the identical test
scenarios and seeds. The target model keeps its own Safety Gate thresholds and detector configuration, because these are
safety constraints selected on its validation scenarios rather than controller decision logic. Three systems are
compared per scenario: the static IDS, the target's native controller and the transferred controller. Decision-quality
transfer is reported as (i) the share of the native controller's macro-F1 gain over the static IDS that the transferred
controller retains and (ii) the agreement of the two controllers' chosen actions, and the rank correlation of their
utility scores, when both decide on identical proxy estimates at the same decision points. H5 is judged only on REAL
data with at least six paired seeds; below that the verdict is *inconclusive* by construction.
*Results:* `results/runs/<phase_d_transfer_*>/summary.json` and `report.md`. **[to be filled from the real run]**

**Cross-dataset (RQ7).** The Phase A–C workflow is repeated on UNSW-NB15 (and CSE-CIC-IDS2018 if time allows) through
the existing adapters, after `evox generalize adapter-check` passes on the real files. For each dataset × algorithm ×
drift kind, EVO-X's macro-F1 advantage over each fixed policy, over the static IDS and over the best fixed policy in
hindsight is tabulated with seed-level dispersion, and classified relative to the reference dataset as *holds*,
*shrinks*, *absent*, *reversed*, *emerges* or *none*. The advantage is not averaged across datasets; where it shrinks or
disappears this is reported as a finding. Datasets differ in features, attack families and, for UNSW-NB15, in the
categorical columns that the adapter drops.
*Results:* `results/runs/phase_d_generalization/portability.csv`, `report.md`, `portability.png`. **[to be filled]**

## Dashboard and end-to-end demonstration (Module 15)

A read-only, replay-driven dashboard presents a stored run to a non-specialist reviewer: current and historical model
versions, the drift state of each detected event against the controller's thresholds, the eligibility rule that fired,
each action's utility terms and the chosen action, every candidate's Safety Gate checks (value against bound), the
promotion / rejection / rollback timeline, and per-system results. The registry journal's hash chain is re-verified
when it is displayed, so a tampered history is shown as unverified. The dashboard reads finished runs only and has no
dependency on a live Kafka broker. It listens on the loopback interface by default, accepts only GET requests, and
displays all file content as text.

The scripted demonstration (`evox demo`) walks one scenario through drift, controller decision, candidate creation,
a gate promotion, a gate rejection and a post-promotion rollback, ending with a check that the restored model's
predictions are identical to the original. The lifecycle events are produced by the real Phase C code; the controller's
inputs are scripted and labelled as such.

## Reproducibility (Module 16)

`evox reproduce` chains the Phase B, C and D workflows into one fresh directory and writes a manifest of Python and
package versions, git commit, seeds, exact arguments, per-step status and wall time, and SHA-256 checksums of the key
artefacts. The default run is synthetic and quick and demonstrates that the pipeline executes end to end; it is never
a research result. Wall-clock quantities differ between machines and are excluded from equality claims.

## Limitations

See `docs/limitations.md`: synthetic drift construction; dataset age and artefacts; two tree-ensemble base models;
threshold and weight sensitivity; a single label policy; approximate "-inspired" baselines; a gate and detector that
belong to the target algorithm in the transfer experiment; no live deployment.
