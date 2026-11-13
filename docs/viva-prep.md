# Viva preparation: likely questions and where the evidence is

| Question | Short answer | Evidence in the repo |
|---|---|---|
| Why rules + utility and not RL? | A transparent, auditable decision record was required; RL needs far more data than a replayed benchmark provides and would be hard to pre-register. Meta-control is out of scope (§7.7). | `docs/design/controller.md`; `evolution_events.jsonl`; rules-only / utility-only ablations in Phase C |
| Why these thresholds and weights? | They are not hand-picked: a pre-registered grid, chosen on validation scenarios by a stated criterion, frozen before test scenarios were built. | `preregistered_protocol.json`, `weight_sensitivity.csv`, `gate_sensitivity.csv` |
| What if the gate rejects everything? | Then EVO-X degenerates to the static model and that is reported. The sensitivity sweep and rejected-candidate counts show how often it happens. | `gate_sensitivity.csv`, `counterfactuals.csv`, dashboard *Candidates & gate* |
| How do you know the drift is real? | Synthetic drift has ground truth by construction; detectors are scored against it. Natural CICIDS2017 drift is mostly novel-class and is kept as its own scenario. | scenario ground truth (`experiments/scenarios.py`), Phase A detector tables, `docs/limitations.md` |
| How do you know a rejected update was really harmful? | Counterfactual replay shadow-deploys rejected candidates and measures what would have happened. | `counterfactuals.csv`, hypothesis H3 in `hypotheses.json` |
| Does the advantage generalise? | Reported per dataset and per drift kind with a verdict (holds / shrinks / absent / reversed), never averaged. | `evox generalize report`, `portability.csv` |
| Did the controller transfer across algorithms? | Measured as the share of native gain kept, plus agreement at identical decision points. | `evox generalize transfer`, `summary.json` (H5) |
| Is rollback really exact? | Tests and the demo check bit-identical predictions after rollback and after reopening the registry; the hash-chained journal is re-verified by the dashboard. | `tests/unit/lifecycle`, `evox demo` |
| Are your quick/synthetic numbers results? | No. They validate the pipeline, and every report and dashboard view says so. | run protocols (`data_tag`, `quick`), dashboard banner |
| Where does EVO-X lose? | Wherever the tables say so; negative results are findings (§8.5). | `per_scenario.csv`, `paired_statistics.csv` |
