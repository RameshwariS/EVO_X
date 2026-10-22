"""
Basic smoke tests for the Phase 1 pipeline (Synopsis §7.5 lists PyTest for
"unit/integration testing for drift detectors, controller logic, and safety
gate" — these early tests cover the pre-drift static-baseline components
that later phases build on).

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.synthetic_data import generate_synthetic_cicids
from src.preprocessing import clean, time_order, chronological_split, prepare_features
from src.baseline_ids import train_random_forest
from src.config import LABEL_COLUMN, TIMESTAMP_COLUMN


@pytest.fixture(scope="module")
def raw_df():
    return generate_synthetic_cicids(n_rows=4000, seed=123)


def test_synthetic_data_has_expected_schema(raw_df):
    assert LABEL_COLUMN in raw_df.columns
    assert TIMESTAMP_COLUMN in raw_df.columns
    assert raw_df[LABEL_COLUMN].nunique() > 1


def test_clean_removes_inf_values(raw_df):
    cleaned = clean(raw_df)
    numeric = cleaned.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    assert not numeric.isna().any().any()


def test_time_order_is_monotonic(raw_df):
    cleaned = clean(raw_df)
    ordered = time_order(cleaned)
    ts = pd.to_datetime(ordered[TIMESTAMP_COLUMN])
    assert (ts.diff().dropna() >= pd.Timedelta(0)).all()


def test_chronological_split_is_not_random(raw_df):
    """The defining property: train/val/test partitions must be
    time-contiguous, i.e. every train timestamp <= every val timestamp
    <= every test timestamp (Synopsis §7.6)."""
    cleaned = time_order(clean(raw_df))
    train, val, test = chronological_split(cleaned)
    assert train[TIMESTAMP_COLUMN].max() <= val[TIMESTAMP_COLUMN].min()
    assert val[TIMESTAMP_COLUMN].max() <= test[TIMESTAMP_COLUMN].min()


def test_chronological_split_covers_all_rows(raw_df):
    cleaned = time_order(clean(raw_df))
    train, val, test = chronological_split(cleaned)
    assert len(train) + len(val) + len(test) == len(cleaned)


def test_prepare_features_no_leakage_in_label_space(raw_df):
    """All classes present anywhere in the data must be encodable (closed
    world assumption for the Phase 1 static baseline, see preprocessing.py
    docstring)."""
    cleaned = time_order(clean(raw_df))
    train, val, test = chronological_split(cleaned)
    prepared = prepare_features(train, val, test)
    assert prepared.X_train.shape[0] == len(train)
    assert prepared.X_test.shape[1] == len(prepared.feature_names)
    # no NaNs should survive into the scaled feature matrices
    assert not np.isnan(prepared.X_train).any()
    assert not np.isnan(prepared.X_test).any()


def test_random_forest_trains_and_predicts(raw_df):
    cleaned = time_order(clean(raw_df))
    train, val, test = chronological_split(cleaned)
    prepared = prepare_features(train, val, test)
    model = train_random_forest(prepared.X_train, prepared.y_train)
    preds = model.predict(prepared.X_test)
    assert len(preds) == len(prepared.y_test)
    # A trained baseline should clearly beat random guessing on this
    # separable synthetic data.
    accuracy = (preds == prepared.y_test).mean()
    assert accuracy > 0.5
