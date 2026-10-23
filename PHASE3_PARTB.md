# Phase 3 — Part B

## Done for you (this zip)
| Item | Where |
|---|---|
| Training protocol B (train on full reference), `--dry-run` size check | `phase3_static_ids.py` |
| One-command report section generated from your real results (refuses to look "real" on synthetic data) | `make_phase3_report.py` → `docs/Phase3_report_section.md` |
| HOIDS citation investigated — **could not be verified**, closest paper + options | `docs/hoids_citation_resolution.md` |
| Controller rules/weights pre-registration draft (updated) | `docs/controller_spec_draft.md` |
| 31 tests green (report generator included) | `tests/` |

## What only you can do (needs the real stream)
1. Finish Phase 2 Part B (real stream built, reference window decided).
2. `python phase3_static_ids.py --dry-run --max-train-per-class 100000` — check the training sizes look sane.
3. **Choose the protocol with Prof. Patil and write it down:** A = train on `ref_train`, keep `ref_val` held out (default);
   B = `--train-on-full-reference` (final models use all reference data; `ref_val` becomes in-sample and is not reported).
   Suggested: run A once to see the pre-deployment check, then use B for the numbers you cite — or just A everywhere; but be consistent.
4. `python phase3_static_ids.py --seeds 5 --max-train-per-class 100000` (hours without the cap).
5. `python make_phase3_report.py` → open `docs/Phase3_report_section.md` (must NOT start with "NOT REPORTABLE") and paste into Chapter 8
   together with `results/figures/phase3_degradation.png`.
6. Sanity-check before trusting the numbers: stream F1 should be clearly below the pre-deployment score; confidence should not be ~1.0 on
   novel flows without comment (point it out if it is — XGBoost is often over-confident); look at the figure for the class composition strip.
7. Commit code + `phase3_summary*.csv` (flow-log `.npz` files are git-ignored).
8. Settle the HOIDS question (see doc) before editing Chapter 3 / §8.2 / reference [3].

## Definition of done
Real run, both models × 5 seeds · protocol written down · report section generated and pasted · HOIDS decision made · tests green.

# Next phase: Phase 4 — Kafka streaming (see `PHASE4_PLAN.md`)
