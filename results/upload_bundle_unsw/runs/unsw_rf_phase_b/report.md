# Phase B run (REAL)

Algorithm: random_forest. Quick mode: False. Scenarios: 171. Detected events: 193. Weight vectors: 781.

Selected actions: {"no_op": 193}. All candidates remain SHADOW_ONLY.

The source-flow grouped holdouts prevent resampled flows crossing training/proxy/audit roles. Labels obey the recorded delay and fraction. Weights and cost profiles use validation scenarios only. Ablation metrics are proxy-candidate estimates, not deployed continual-learning results. Training memory is Python allocation peak plus serialized model size; native allocation/RSS is not measured. Phase C must independently evaluate and gate full candidates before deployment.
