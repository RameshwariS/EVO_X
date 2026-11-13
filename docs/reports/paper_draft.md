# EVO-X: autonomous model evolution for continual intrusion detection under concept drift — paper skeleton

> Status: **skeleton**. Each `[RESULT: …]` marks a number or table that must come from a REAL, non-quick run with
> at least six test seeds. Nothing here is a finding yet. Do not paste synthetic or quick-run numbers.
> Generate §5 with `python -m evox paper-fill ...` (see `docs/guides/phase-d.md`); it refuses non-REAL / quick / <6-seed runs.

## Abstract
Problem (IDS accuracy decays under concept drift), approach (drift characterisation → controller → gated candidate →
versioned registry with rollback), evidence (`[RESULT: one sentence per supported/unsupported hypothesis]`), caveat
(replayed benchmarks, synthetic drift for known classes).

## 1. Introduction
Contribution list, each tied to evidence: (1) decoupled streaming + drift laboratory with ground truth;
(2) transparent two-stage controller (eligibility rules + utility ranking); (3) independent evaluation, Safety Gate,
counterfactual harm measurement; (4) versioned registry with exact rollback; (5) portability study.
Keep the novelty claim defensible and **re-verify the literature gap with a fresh search** before submission.

## 2. Related work
Concept-drift detection; continual / incremental learning for IDS; replay and forgetting mitigation; model
lifecycle management. Cite the baseline sources; call re-implementations "-inspired". Resolve the "HOIDS" reference
(see `docs/research/hoids-citation.md`) and fix reference [3].

## 3. EVO-X architecture
Figure: stream → scorer → detectors → analyzer (signature) → controller → adaptation engine → evaluator → Safety Gate →
registry / rollback. Describe the label policy and the leakage rules.

## 4. Experimental protocol
Datasets and adapters; synthetic drift taxonomy and what each type actually is (prior shift vs synthetic
concept/covariate shift); validation vs test seed families; pre-registered weight and threshold selection; systems
compared; metrics (never accuracy as a headline); statistics (paired Wilcoxon, ≥ 6 seeds); fairness (identical label
budget and data access).

## 5. Results
5.1 Detection (delay, false alarms) — `[RESULT: phase6 detector tables]`.
5.2 Main comparison per drift scenario — `[RESULT: per_scenario.csv, paired_statistics.csv]`.
5.3 Hypotheses H1–H4 — `[RESULT: hypotheses.json, marked supported / not supported / inconclusive]`.
5.4 Ablations: no gate, rules-only, utility-only, weight and threshold sensitivity.
5.5 Cross-algorithm transfer (H5) — `[RESULT: phase_d_transfer summary.json]`.
5.6 Cross-dataset portability (RQ7) — `[RESULT: portability.csv]`.
Report negative results as findings.

## 6. Discussion
Where the advantage holds, shrinks or disappears and why; cost versus benefit; what the gate prevented.

## 7. Limitations
Use `docs/limitations.md` verbatim as the starting point.

## 8. Conclusion and future work
Meta-controller / RL, neural base models, live deployment, naturally recorded drift.

## Reproducibility statement
Reference `MANIFEST.json`, seeds, code snapshot (`source_snapshot.zip`) and the commands in `docs/guides/phase-d.md`.
