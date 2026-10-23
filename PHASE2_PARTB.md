# Phase 2 — Part B

## Already done for you (in this zip)
| Item | File |
|---|---|
| Preflight check of the raw download | `check_raw_data.py` |
| Evidence for the reference-window decision | `scripts_compare_reference_fractions.py` |
| UNSW-NB15 and CSE-CIC-IDS2018 adapters (RQ7) | `src/dataset_adapters.py` |
| Draft of the report section (fill the blanks) | `docs/Phase2_report_section.md` |
| Plan for the next phase | `PHASE3_PLAN.md` |
| Tests (22 pass; adapters tested on synthetic frames only) | `tests/test_stream.py` |

## What only you can do (needs the real data) — in this order
1. **Download** CICIDS2017 `GeneratedLabelledFlows.zip` (TrafficLabelling, has `Timestamp`) from
   https://www.unb.ca/cic/datasets/ids-2017.html into `data/raw/` (unzip the 8 CSVs). ~8 GB RAM needed.
2. `python check_raw_data.py` → must print `PREFLIGHT: OK`. If timestamps are "<99% parseable",
   send me the sample timestamp it prints.
3. `python phase2_build_stream.py` → all checks PASS, log shows `ordering: timestamp` and the
   "12-hour quirk" warning. Write down: rows before/after dedup, split sizes.
4. `python scripts_compare_reference_fractions.py` → look at `results/metrics/phase2_reference_sweep.csv`.
5. **Meet Prof. Patil and decide the reference window** (record the decision in the report).
   What the synthetic week already shows, and what you should confirm on real data: because each attack
   family lives in one weekday, "novel share of stream attacks" stays ≈100 % for most reference
   fractions (it is not monotone). So the natural stream is essentially a **novel-class scenario**; recurring /
   gradual / incremental / temporary drift of *known* classes will have to come from the Phase 5
   resequencing generator, as the synopsis (Module 2) already says. My recommendation: keep
   `REFERENCE_FRACTION` where the reference contains Benign + ≥2 attack families (on real data likely
   ~0.35–0.45, i.e. through Tuesday/Wednesday morning); confirm from the sweep table.
6. Fill in `docs/Phase2_report_section.md` with the real numbers + the figure; commit code (not data).
7. Optional (RQ7): download UNSW-NB15 (`UNSW-NB15_1..4.csv`) and/or CSE-CIC-IDS2018 (use
   `subsample_every=10`), run through the adapters and `build_stream`. The adapters follow the public
   documentation but are unverified on the real files.
8. Delete the nested duplicate `EVO-X-Phase1/` folder from your repo.

## Definition of done for Phase 2
Stream built from real CICIDS2017 with all validation checks passing · reference-window decision
written down and justified · report section filled · tests green in CI/locally.
