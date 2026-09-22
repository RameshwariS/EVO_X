"""
EVO-X Phase 1 — Configuration
==============================
Central place for paths, column conventions, and constants used across the
Phase 1 pipeline (static baseline IDS on CICIDS2017), corresponding to:

    Synopsis §1.1  — conventional pipeline: Historical Dataset -> Train ML
                      Model -> Deploy -> Predict Attacks
    Synopsis §7.9  — Phase 1: "Literature review; baseline static IDS on
                      CICIDS2017"
    Synopsis §8.3  — Experiment 1 (Static baseline)

If real CICIDS2017 CSVs (from the CIC/UNB distribution, e.g.
"Monday-WorkingHours.pcap_ISCX.csv", "Wednesday-workingHours...csv", etc.)
are placed in DATA_DIR, the pipeline will use them. Otherwise it falls back
to a synthetic CICIDS2017-like generator (src/synthetic_data.py) so the
pipeline is runnable and testable without the ~7GB real dataset.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

for d in [RAW_DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset conventions (CICIDS2017 / CSE-CIC-IDS2018 style)
# ---------------------------------------------------------------------------
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"
TIMESTAMP_COLUMN = "Timestamp"

# Columns that are identifiers / leak information and must never be used
# as model features (kept for traceability but dropped before training).
NON_FEATURE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Source Port",
    "Src Port",
    "Destination Port",
    "Dst Port",
    TIMESTAMP_COLUMN,
    LABEL_COLUMN,
]

# ---------------------------------------------------------------------------
# Experiment / reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SEEDS = 5  # matches Synopsis §8.6: repeated across multiple random seeds

# Time-ordered split (NOT a random split — Synopsis §7.6 explicitly
# rejects "random split relabelled as continual learning")
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10
TEST_FRACTION = 0.20

# ---------------------------------------------------------------------------
# Baseline model hyperparameters (Synopsis §7.3, Module 3 — Base IDS)
# ---------------------------------------------------------------------------
RANDOM_FOREST_PARAMS = dict(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    class_weight="balanced",  # addresses class imbalance (Synopsis §8.4 note)
    random_state=RANDOM_SEED,
)

XGBOOST_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=RANDOM_SEED,
)
