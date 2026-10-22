# EVO-X — Phase 1: Static Baseline IDS on CICIDS2017

**Project:** EVO-X: An Autonomous AI Model Evolution Framework for Continual
Intrusion Detection under Concept Drift
**Team:** Prashil Suryawanshi, Rameshwari Satpute, Pooja Bhosale, Himanshu
Bangalkar — Walchand College of Engineering, Sangli
**This deliverable:** Synopsis §7.9, **Phase 1** — "Literature review;
baseline static IDS on CICIDS2017" and Synopsis §8.3, **Experiment 1**
("Static baseline: trained once on initial data; never adapts").

This is intentionally the *simplest* piece of the full EVO-X architecture
(Figure 7.1): everything up to and including the "Current IDS" box, with no
drift detection, controller, or adaptation logic yet — those are Phase 5+.
Its job is to (a) stand up a clean, reusable data/preprocessing pipeline the
rest of the project will build on, and (b) produce the pre-drift reference
numbers that later experiments (Chapter 8) will compare against.

## What this does

```
Data Sources (CICIDS2017 CSVs, or synthetic fallback)
        │
        ▼
Preprocessing: cleaning, encoding, scaling   (src/preprocessing.py)
        │  — chronological (time-ordered) train/val/test split, NOT random
        ▼
Current IDS: Random Forest + XGBoost          (src/baseline_ids.py)
        │
        ▼
Evaluation: Precision/Recall/F1/FPR/FNR,      (src/evaluate.py)
            per-class + overall, confusion matrix
```

This matches the left-hand column of Figure 7.1 in the synopsis
(Data Sources → Stream Simulator/Preprocessing → Current IDS), minus the
Kafka streaming layer, which is Phase 4.

## Getting the real dataset

> https://www.unb.ca/cic/datasets/ids-2017.html

Download the `MachineLearningCSV.zip` (or equivalent per-day CSVs), unzip,
and place the CSV files in `data/raw/`, e.g.:
```
data/raw/Monday-WorkingHours.pcap_ISCX.csv
data/raw/Tuesday-WorkingHours.pcap_ISCX.csv
data/raw/Wednesday-workingHours.pcap_ISCX.csv
...
```

`main.py` auto-detects any `*.csv` in `data/raw/` and uses them. **No code
changes are needed** — `src/data_loader.py` handles the well-known
CICIDS2017 CSV quirks (leading-space column names, `inf`/`NaN` in
`Flow Bytes/s`, etc.).

## Running without the real dataset

Because CICIDS2017 is a multi-gigabyte manual download, this repo ships a
**synthetic, schema-compatible generator** (`src/synthetic_data.py`) so the
full pipeline can be run, read, and unit-tested immediately:

```bash
pip install -r requirements.txt
python main.py --synthetic --rows 50000
```

If real CSVs are present in `data/raw/`, just run:

```bash
python main.py
```

**Do not report synthetic-data numbers as CICIDS2017 results** — the
synthetic generator exists only to validate that the pipeline runs
end-to-end; it is not a substitute for real traffic statistics.

## Outputs

- `results/models/*.joblib` — trained Random Forest / XGBoost baselines
- `results/metrics/*_metrics.json` — full per-class + overall metrics
- `results/metrics/model_comparison.csv` — side-by-side comparison table
- `results/figures/*_confusion_matrix.png` — normalized confusion matrices

## Design decisions worth flagging to your guide

1. **Chronological split, not random split.** Train/val/test are split by
   time order (70/10/20), not `sklearn.train_test_split`. This is required
   by Synopsis §7.6 ("avoiding a random split relabelled as continual
   learning") and is what later phases' streaming/drift-injection modules
   will build on.
2. **Class imbalance is handled at training time**, not just measured at
   evaluation time: Random Forest uses `class_weight="balanced"`; XGBoost
   uses inverse-frequency sample weights. This directly addresses the
   evaluation-metric caveat in Synopsis §8.4 ("accuracy... is a poor
   indicator under class imbalance").
3. **Label encoder is fit on the union of train/val/test classes**, not
   train alone — a closed-world assumption appropriate for a *static*
   baseline, since some CICIDS2017 attack classes (e.g. Heartbleed) have
   so few flows that a strict chronological split can otherwise put every
   instance in the test fold. The open-world case (a genuinely novel class
   appearing after deployment) is deliberately left to the "novel-class"
   drift scenario in later phases — see the docstring in
   `src/preprocessing.py` for the full reasoning.
4. **Accuracy is reported but not treated as the headline metric** — see
   the confusion matrix output, where rare classes (Bot, Infiltration) are
   effectively invisible under raw accuracy but visible in macro-F1 and
   per-class FPR/FNR. This is the empirical justification, not just an
   assertion, for why Chapter 8's evaluation plan avoids accuracy.

## Project layout

```
EVO-X-Phase1/
├── main.py                  # Phase 1 entry point
├── requirements.txt
├── README.md
├── src/
│   ├── config.py             # paths, hyperparameters, split ratios
│   ├── data_loader.py        # Module 1: Data Acquisition
│   ├── synthetic_data.py     # CICIDS2017-schema synthetic fallback
│   ├── preprocessing.py      # cleaning, time-ordering, split, scaling
│   ├── baseline_ids.py       # Module 3: Base IDS (RF, XGBoost)
│   └── evaluate.py           # Synopsis §8.4 metrics + confusion matrix
├── tests/
│   └── test_pipeline.py      # pytest smoke tests for every stage
└── data/
    └── raw/                  # place real CICIDS2017 CSVs here
```

## Next phases (not part of this deliverable)

Per Synopsis §7.9: Phase 2 (stream construction), Phase 3 (this baseline,
extended), Phase 4 (Kafka streaming), Phase 5 (synthetic drift generator),
Phase 6 (ADWIN/Page-Hinkley/PSI/MMD drift detection), Phase 7 (adaptation
strategies), Phase 8 (Evolution Controller), Phase 9 (Safety Gate), Phase
10 (model registry/rollback), Phase 11 (comparative experiments).

## Known gaps to resolve before your next review

- **Reference [3] (HOIDS) is missing full citation details** ("Authors to
  be confirmed" in the synopsis references list) — resolve before this is
  used as a comparison baseline in Phase 11.
- **Evolution Controller's exact rule set** (which magnitude/persistence
  thresholds route to which of the 5 strategies, before the utility score
  is computed) needs to be written out as pseudocode ahead of Phase 8 —
  this was also flagged directly by your external reviewer (Anish Shobith).
- **Utility-function weight selection procedure** should be pre-registered
  (e.g., grid search on a held-out validation stream) before Phase 8, so
  the weights don't look tuned post-hoc to favor EVO-X.
