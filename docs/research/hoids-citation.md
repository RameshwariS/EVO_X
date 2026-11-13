# Reference [3] "HOIDS" — status (Phase 3 Part B, task 5)

**Result of a web search (Oct 2026): I could not find a paper named "HOIDS: Concept drift aware hybrid online intrusion
detection system".** I did not invent a citation. Do not leave "Authors to be confirmed" in the final report.

## Closest match found (verify it is what you meant)
F. Camarda, A. De Paola, S. Drago, P. Ferraro, G. Lo Re, **"Managing Concept Drift in Online Intrusion Detection Systems
with Active Learning"**, Joint National Conference on Cybersecurity (ITASEC & SERICS 2025), Bologna, Feb 2025,
CEUR Workshop Proceedings vol. 3962, paper 42. https://ceur-ws.org/Vol-3962/paper42.pdf

Why it fits the synopsis description ("drift detection + online and active learning to reduce unnecessary labelling and
retraining"): incremental Random Forest (AIRF), uncertainty-based active learning (label ~0.5 % of records), error-rate
drift detector with accuracy threshold α, adaptation only when drift is detected (prune to best trees, warm-start new trees).
Why it may NOT be the source: it never uses the name "HOIDS", it is not described as "hybrid", and it evaluates only on KDD'99.

## What to do (in this order)
1. Ask where "HOIDS" came from (your notes / an AI-generated summary / a paper you read). If you cannot produce the source,
   treat the name as unverified.
2. Ask Prof. Patil which prior work she expects as the "online + active learning" baseline.
3. Recommended wording if no source appears: rename the baseline **"Drift-triggered incremental RF with active learning
   (after Camarda et al., 2025)"**, cite that paper, and remove the name "HOIDS" from §3, §8.2 and the references. Update
   Chapter 3 text accordingly (this paper is closest to the "reduce unnecessary adaptation" point).

## How it maps to a Phase 7 / Phase 11 baseline (if you adopt it)
| Their component | EVO-X implementation |
|---|---|
| AIRF incremental RF | `IDSModel("random_forest")` + `update()`: keep top half-minus-one trees by accuracy on the retraining window, warm-start to n_trees |
| Informative active learning (1 − max proba) | already available: `predict_with_confidence` (Phase 3 logs confidence) |
| Error-rate drift detector (accuracy < α on labelled sample) | fixed-rule trigger on the Phase 3 flow log; needs labels for ~0.5 % of the batch |
| Retraining window RW | replay buffer of the last window + selected old records |
Note the label-budget assumption: it needs a labelled sample per batch. Decide whether your baselines get the same
label budget as EVO-X (fairness; say so in §8).
