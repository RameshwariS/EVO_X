# Evolution Controller — specification DRAFT (Phase 3 side task; needed before Phase 8)
> Phase B implementation and the executable selection protocol are now documented
> in [PHASE_B.md](../guides/phase-b.md). This file preserves the original design draft.

Your external reviewer (Anish Shobith) flagged that the selection logic is the core and least-specified part.
Everything below is a starting point to review with your guide. Threshold values are placeholders `[τ]`
to be set from validation-stream experiments, NOT chosen by looking at test scenarios.

## 1. Inputs (from the Drift Analyzer)
`magnitude m` (fused drift score), `persistence p` (consecutive batches above threshold), `novelty n`
(share of flows predicted with low confidence / flagged as unseen-class), `recurrence r` (similarity to a stored past
drift signature, 0–1), `perf_drop d` (recent F1 drop vs reference-val), plus budget `B` (remaining adaptation cost).

## 2. Stage 1 — rule-based candidate filter (which strategies are *eligible*)
```
if m < [τ_low] or p < [τ_p_short]:                  eligible = {No-Op}               # noise / temporary shift
elif r >= [τ_rec]:                                   eligible = {Replay, Incremental}  # seen before: reuse memory
elif n >= [τ_nov] and m >= [τ_high]:                 eligible = {Full Retrain, Strategic Selection}
elif m >= [τ_high] and p >= [τ_p_long]:              eligible = {Full Retrain, Replay}
else:                                                eligible = {Incremental, Strategic Selection, Replay}
if B < cost(Full Retrain): eligible -= {Full Retrain}
```
## 3. Stage 2 — utility scoring among eligible actions (§7.4)
`Utility(a) = w1·ΔF1_est(a) − w2·Forgetting_est(a) − w3·Cost(a) − w4·ΔFPR_est(a) − w5·Latency(a)`; `a* = argmax`.
Estimates come from a cheap proxy evaluation on a small held-out sample (stated in §7.4); `Cost`/`Latency` from
the measured profiles (Phase 3 `fit_seconds`, `latency_ms_per_1k_flows`; Phase 4 end-to-end latency).
Stage 1 + Stage 2 together answer the reviewer's point: rules constrain, utility ranks.

## 3. Stage 3 — always passes through Candidate Evaluation + Safety Gate (§7.4, thresholds τ1..τ5)
If the gate rejects `a*`: roll back, then optionally try the next-best eligible action (max 1 retry; log both).

## 4. Pre-registration of the weights (write this down BEFORE Phase 11)
- Weight grid: each `w_i ∈ {0, 0.25, 0.5, 1}` normalised to sum 1 (list the exact grid).
- Selection data: a **validation stream** built from a different seed/scenario draw than any reported test scenario.
- Selection criterion: pick the weight vector maximising [macro-F1 − λ·normalised cost] on the validation stream; ties → lower cost.
- Report the sensitivity (all grid points) in an appendix; do not re-tune after seeing test results.
- Ablations to plan: rules-only, utility-only, rules+utility, with/without safety gate.

## 5. Open questions for your guide
1. Are rule thresholds learned on the validation stream, or fixed from drift-detector theory?
2. What counts as "unnecessary update" operationally (RQ2)? Suggest: candidate rejected by gate, or ΔF1_new < τ1 after promotion.
3. How is `recurrence` computed (distance between per-feature PSI vectors?) and what is the storage policy?

## 6. HOIDS citation
See `docs/research/hoids-citation.md` — no paper by that name could be found; closest match and next steps are listed there.
