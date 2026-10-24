"""
EVO-X Phase 1 — Preprocessing (Synopsis §7.3, mapped to the "Preprocessing:
Cleaning, Encoding, Scaling" stage in Figure 7.1 architecture diagram)
=============================================================================
Steps:
  1. Clean:   drop identifier/leak columns, handle inf/NaN artefacts
              (documented CICIDS2017 quirk, Synopsis §9.3), drop duplicates.
  2. Time-order: sort by Timestamp (NOT randomly split — Synopsis §7.6).
  3. Split:   chronological train/val/test split (also NOT random —
              preserves the streaming/continual-learning setup used later
              in Phase 5+; a random split would be "random split relabelled
              as continual learning", which the synopsis explicitly rejects).
  4. Encode:  label-encode the multi-class target.
  5. Scale:   fit StandardScaler on TRAIN ONLY, apply to val/test (avoids
              train/test leakage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from evox.config import (
    LABEL_COLUMN,
    TIMESTAMP_COLUMN,
    NON_FEATURE_COLUMNS,
    TRAIN_FRACTION,
    VAL_FRACTION,
    TEST_FRACTION,
)

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list
    label_encoder: LabelEncoder
    scaler: StandardScaler


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Replace inf/-inf (a documented CICIDS2017 artefact, e.g. Flow Bytes/s
    # divides by a duration that can be ~0) with NaN, then drop rows that
    # are unrecoverable in the numeric feature columns.
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    before = len(df)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))
    df = df.drop_duplicates()
    logger.info("Cleaning: %d -> %d rows (dropped %d duplicates)", before, len(df), before - len(df))

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    return df


def time_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by timestamp if available; otherwise assume the input is
    already in acquisition order (Synopsis §7.6: 'flows are re-ordered by
    timestamp ... to construct meaningful streams')."""
    if TIMESTAMP_COLUMN in df.columns:
        df = df.copy()
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    else:
        logger.warning("No '%s' column found — assuming rows are already time-ordered.", TIMESTAMP_COLUMN)
    return df


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    logger.info(
        "Chronological split -> train=%d (%.0f%%) val=%d (%.0f%%) test=%d (%.0f%%)",
        len(train), 100 * len(train) / n,
        len(val), 100 * len(val) / n,
        len(test), 100 * len(test) / n,
    )
    return train, val, test


def prepare_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> PreparedData:
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in train.columns]
    feature_names = [c for c in train.columns if c not in drop_cols and c != LABEL_COLUMN]
    # Keep only numeric features for the baseline model (categorical/port
    # engineering is left for later phases if needed).
    feature_names = [c for c in feature_names if pd.api.types.is_numeric_dtype(train[c])]

    # NOTE: The label encoder is fit on the union of classes across all
    # three chronological splits (train+val+test), not on train alone.
    # Phase 1's static baseline assumes a CLOSED attack-class taxonomy —
    # every attack category a classifier will ever need to name is known
    # up front from the dataset's label set. This is a deliberate
    # simplifying assumption for the static baseline only: some rare
    # attack classes (e.g. Heartbleed, Infiltration) have so few flows in
    # CICIDS2017 that a strict chronological split can otherwise put every
    # instance of a class in the test fold, making it "unseen" at fit
    # time. The OPEN-world case — a genuinely novel attack class
    # appearing only after deployment — is deliberately deferred to the
    # "novel-class" drift scenario in later phases (Synopsis §7, Module 2),
    # where it is a first-class research question, not a preprocessing
    # artefact to be avoided here.
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train[LABEL_COLUMN], val[LABEL_COLUMN], test[LABEL_COLUMN]]))
    y_train = label_encoder.transform(train[LABEL_COLUMN])
    y_val = label_encoder.transform(val[LABEL_COLUMN])
    y_test = label_encoder.transform(test[LABEL_COLUMN])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names])
    X_val = scaler.transform(val[feature_names])
    X_test = scaler.transform(test[feature_names])

    logger.info("Prepared feature matrix: %d features, %d classes", len(feature_names), len(label_encoder.classes_))
    logger.info("Classes: %s", list(label_encoder.classes_))

    return PreparedData(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=feature_names, label_encoder=label_encoder, scaler=scaler,
    )


def run_preprocessing_pipeline(raw_df: pd.DataFrame) -> PreparedData:
    df = clean(raw_df)
    df = time_order(df)
    train, val, test = chronological_split(df)
    return prepare_features(train, val, test)
