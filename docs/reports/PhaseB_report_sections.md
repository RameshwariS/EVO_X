# Phase B report draft — Modules 7–9

## Evolution Controller (Module 7)

The controller implements two transparent stages. An eligibility filter maps
drift magnitude, persistence, recurrence, ended state and confidence-based
unfamiliarity to admissible adaptation actions. A utility scorer then balances
held-out performance gain against forgetting, adaptation cost, false positives
and added latency. No-Op is always available as a conservative fallback. Actions
that violate capability or training-budget constraints are explicitly excluded.
Each decision records the full signature, estimates, weighted utility terms,
eligible set, exclusions and selected candidate.

Weights are selected from a pre-declared normalized grid on validation scenarios
only. Cheap candidates use training samples disjoint from both the proxy
evaluation set and the independent validation-audit set used to select weights.
Cost and latency profiles are frozen before test scenario generation. Rules-only,
utility-only and combined-controller ablations are logged for the same events.
The framework does not assume that the combined policy will outperform either
ablation; comparative claims require Phase C experiments.

## Adaptation Engine (Module 8)

Five actions are implemented: No-Op, incremental update, memory replay, strategic
sample selection and full retraining. Random Forest incremental adaptation uses
a bounded prune-and-grow ensemble with explicit probability-column alignment
across changing class sets. XGBoost uses boosting continuation only when the
existing class set is unchanged and fully represented; otherwise this action is
marked unavailable. Replay and strategic selection train fresh candidates with
historical memory; recent-only full retraining is retained as the naive policy.
Strategic selection is SSF-inspired rather than an exact reproduction.

Label availability follows the configured delay/fraction uniformly across
strategies and evaluation. Memory is bounded and class-balanced. Repeated source
flows in synthetic scenarios are grouped into one data role, preventing overlap
between training and holdouts even when the same source flow is replayed with
different drift offsets.

## Candidate Model (Module 9)

Adaptation never changes the active model. A candidate contains its parent
version, strategy, content-hashed training snapshot, training IDs, label policy,
configuration and measured resource profile. No-Op returns the exact parent.
Candidates remain unpromoted pending the Phase C evaluator and Safety Gate.
Python allocation peak and serialized model size are reported with their scope;
they must not be described as total process/native peak memory.

## Experimental evidence to insert

Use the selected run directory's `summary.json`, `weight_sensitivity.csv`,
`candidate_profiles.csv`, `proxy_profiles.csv`, `ablations.csv` and `report.md`.
Record the dataset/source tag, model, seed families, quick/full scope, label
policy, thresholds, selected weights and hardware. Synthetic or quick runs
validate the pipeline and do not establish CICIDS2017 superiority. Full-model
safety, forgetting across promoted versions and harmful-update counts require
Phase C and must not be inferred from Phase B proxy metrics.
