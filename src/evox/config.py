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

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    if configured := os.getenv("EVOX_PROJECT_ROOT"):
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "evox").is_dir():
            return parent
    # An installed wheel uses its caller's workspace unless configured explicitly.
    return Path.cwd().resolve()


PROJECT_ROOT = _project_root()
DATA_DIR = Path(os.getenv("EVOX_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = Path(os.getenv("EVOX_RESULTS_DIR", str(PROJECT_ROOT / "results"))).expanduser().resolve()
PROCESSED_DIR = DATA_DIR / "processed"
STREAM_DIR = PROCESSED_DIR / "cicids2017_stream"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

def ensure_output_dirs() -> None:
    """Create output paths explicitly; importing configuration is read-only."""
    for directory in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

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

# ---------------------------------------------------------------------------
# Phase 2 — time-ordered stream construction (Synopsis §7.6, §7.9 Phase 2)
# ---------------------------------------------------------------------------
# The stream is: [ reference window | live stream ].  The reference window is
# the "historical dataset" the first IDS is trained on; everything after it
# is replayed batch-by-batch (later, through Kafka — Phase 4).
#
# DESIGN DECISION TO CONFIRM WITH GUIDE: on real CICIDS2017, attacks are
# scheduled by weekday, so a chronological cut places most attack families
# AFTER the reference window.  See results/metrics/phase2_class_timeline.csv
# after running on real data, then pick the value.
REFERENCE_FRACTION = 0.30        # share of the time-ordered data used as reference
REFERENCE_VAL_FRACTION = 0.20    # last 20% of the reference window = validation
STREAM_BATCH_SIZE = 1000         # flows per replay batch

# ---------------------------------------------------------------------------
# Phase A (synopsis phases 4-6) -- streaming, drift lab, detectors
# ---------------------------------------------------------------------------
# Kafka topics (Module 4).  Only the first two carry data in Phase A; the other
# three are created as stubs and filled in Phases B/C.
KAFKA_BOOTSTRAP = os.getenv("EVOX_KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_EVENTS = "network-events"        # 1 partition: ordering is the point
TOPIC_PREDICTIONS = "predictions"
TOPIC_DRIFT = "drift-events"           # stub (written from Phase A3 onwards)
TOPIC_EVOLUTION = "evolution-events"   # stub (Phase B2)
TOPIC_MODEL = "model-events"           # stub (Phase C3)

# LABEL POLICY (decide once, apply to every method including baselines).
# Default proposal from the plan, Appendix 2: labels for batch t become
# available when batch t+LABEL_DELAY_BATCHES has been scored.  Label-free
# detectors (PSI, MMD, confidence-ADWIN) ignore this entirely.
LABEL_DELAY_BATCHES = 2

# Two seed families that must never be mixed (plan, "Rules that apply to every phase").
VALIDATION_SEEDS = (1001, 1002, 1003)            # choose thresholds / weights here
TEST_SEEDS = (2001, 2002, 2003, 2004, 2005, 2006)  # report here (>=6 so Wilcoxon can reach p<0.05)

SCENARIO_DIR = PROCESSED_DIR / "scenarios"
