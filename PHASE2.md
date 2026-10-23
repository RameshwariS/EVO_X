# EVO-X — Phase 2: Dataset preprocessing & time-ordered stream construction

Synopsis §7.9 Phase 2 (1.5 weeks) · §7.6 (time-ordered streams, no random split).
Split used here: **Part A** = code, tests, synthetic validation (done, in this repo).
**Part B** = things that need the real dataset / your guide (below).

## Part A — what was built

| File | Purpose |
|---|---|
| `src/stream_builder.py` | cleaning, timestamp parsing, ordering, reference/stream split, leakage-free scaling, `EvoStream` (+ `iter_batches`, save/load) |
| `src/stream_report.py` | `validate_stream` integrity checks, class timeline (novel-class table), per-batch composition, overview figure |
| `src/synthetic_data.py` | + `generate_synthetic_week` — weekday-scheduled attacks and CICIDS2017 CSV quirks, to test real-data paths offline |
| `src/data_loader.py` | **bug fix**: files were sorted alphabetically (Friday→Monday→Thursday…); now capture order. Optional provenance columns |
| `phase2_build_stream.py` | entry point |
| `tests/test_stream.py` | 11 new tests (18 total pass) |

Stream layout: `[ ref_train | ref_val ] [ live stream → batches of 1000 ]`.

Leakage rules: chronological only; imputation medians + scaler fit on `ref_train` only; class ids fixed
(BENIGN=0) but `reference_class_ids` recorded so classes first appearing in the stream are identifiable as novel.

Real-data quirks handled: leading-space headers, duplicate `Fwd Header Length`, `Infinity`/NaN,
`Web Attack \x96 Brute Force` label, d/m/Y timestamps on a 12-hour clock with no AM/PM
(13:00 is written "1:00"), and CSVs with no Timestamp (file-order fallback).

Usage:
```bash
pip install -r requirements.txt
pytest tests -v
python phase2_build_stream.py --synthetic     # pipeline check only
python phase2_build_stream.py                 # real CSVs in data/raw/
```
Outputs: `data/processed/cicids2017_stream/{stream.npz,stream_meta.json}`,
`results/metrics/phase2_*.{csv,json}`, `results/figures/phase2_stream_overview.png`.

Using the stream in later phases:
```python
from src.stream_builder import EvoStream
s = EvoStream.load("data/processed/cicids2017_stream")
model.fit(s.X_ref_train, s.y_ref_train)          # Phase 3
for batch in s.iter_batches():                   # Phase 4/5/6
    preds = model.predict(batch.X)
```

## Part B — what you do next

1. **Get the right CICIDS2017 files.** Download **`GeneratedLabelledFlows.zip` (folder "TrafficLabelling")**
   from https://www.unb.ca/cic/datasets/ids-2017.html, NOT `MachineLearningCSV.zip`. Only the former has
   `Timestamp`. (The Phase 1 README recommended MachineLearningCSV; it still works via file-order fallback,
   but ordering inside a day is then only approximate.) Unzip the 8 CSVs into `data/raw/`.
   Needs ~8 GB RAM (2.8 M rows).
2. **Run:** `python phase2_build_stream.py`. All validation lines must say PASS, and the log should say
   `ordering: timestamp` and print the "12-hour quirk" warning (that confirms the fix triggered).
3. **Inspect the outputs** and eyeball the figure: Monday benign only; Wed DoS block; Fri DDoS/PortScan.
   Open `phase2_class_timeline.csv`.
4. **Decide the reference-window protocol with Prof. Patil.** With the default 30 % reference, expect most
   attack families to be *novel* in the stream (attacks are weekday-scheduled), i.e. the natural stream is
   mostly novel-class drift, not drift of known classes. Options to discuss: (a) keep 30 % and treat natural
   stream as the "novel-class" scenario; (b) larger reference (e.g. 0.6, run `--reference-fraction 0.6`);
   (c) use the stream only as a base and create abrupt/gradual/recurring scenarios in Phase 5 by
   resequencing (what the synopsis Module 2 describes). Record the decision in the report.
5. **Record in your report (Ch. 7/8):** row counts before/after dedup, split sizes, ordering source,
   novel-class table, the figure, and the leakage rules above. Cite the timestamp quirk as a dataset
   artefact (§9.3).
6. **Commit** code (not data: `data/raw` and `data/processed` are git-ignored). Delete the duplicate
   nested `EVO-X-Phase1/` folder from the repo (root already contains the same code).
7. **Optional, if time:** write a small loader for UNSW-NB15 / CSE-CIC-IDS2018 producing the same
   `EvoStream` (needed for RQ7). Do not start Phase 3 until step 4 is decided, since Phase 3 trains on `ref_train`.

Caveat: Part A was validated only on synthetic data; real-data behaviour (timestamp format variants,
memory, runtime) is unconfirmed until step 2. If the parser logs "<99% parseable", send me the first few
raw `Timestamp` values and I'll adjust.

> Part B materials: see [PHASE2_PARTB.md](PHASE2_PARTB.md) and [PHASE3_PLAN.md](PHASE3_PLAN.md).
