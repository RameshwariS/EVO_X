"""
EVO-X Phase 1 — Baseline IDS Models (Synopsis §7.3, Module 3 — Base IDS)
===========================================================================
Trains the static, never-adapting IDS classifiers that:
  (a) serve as the "Current IDS" block in Figure 7.1, and
  (b) constitute Experiment 1 ("Static baseline") in Synopsis §8.3 — the
      reference point that demonstrates the magnitude of the concept-drift
      problem that EVO-X (Phase 2+) is designed to address.

Two algorithms, matching the synopsis's stated base-classifier choices:
  - Random Forest (interpretable, fast baseline)
  - XGBoost        (stronger, industry-realistic baseline)

The classifier itself is explicitly NOT this project's novelty (Synopsis
§4.1, §7.3) — it is a replaceable substrate for the Evolution Controller
built in later phases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without xgboost installed
    _XGBOOST_AVAILABLE = False

from evox.config import RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, MODELS_DIR

logger = logging.getLogger(__name__)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    logger.info("Training Random Forest baseline (%d samples, %d features)...", *X_train.shape)
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray):
    if not _XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Run: pip install xgboost "
            "(or `pip install -r requirements.txt`)."
        )
    logger.info("Training XGBoost baseline (%d samples, %d features)...", *X_train.shape)
    n_classes = len(np.unique(y_train))
    params = dict(XGBOOST_PARAMS)
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    else:
        params["objective"] = "binary:logistic"
        params.pop("eval_metric", None)
        params["eval_metric"] = "logloss"

    # Sample weighting for imbalance (XGBoost has no native class_weight
    # param like sklearn's RandomForest — Synopsis §9.3 imbalance note).
    class_counts = np.bincount(y_train)
    weights_per_class = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
    sample_weight = weights_per_class[y_train]

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def save_model(model, name: str, models_dir: Path = MODELS_DIR) -> Path:
    path = models_dir / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(name: str, models_dir: Path = MODELS_DIR):
    return joblib.load(models_dir / f"{name}.joblib")
