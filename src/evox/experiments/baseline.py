"""
EVO-X — Phase 1 Entry Point
=============================
Runs: Data Acquisition -> Preprocessing -> Static Baseline Training
      (Random Forest + XGBoost) -> Evaluation -> Comparison

Corresponds to Synopsis §7.9 Phase 1 ("baseline static IDS on CICIDS2017")
and Synopsis §8.3 Experiment 1 ("Static baseline").

Usage:
    python -m evox baseline                  # uses real CSVs in data/raw/ if present,
                                     # else auto-generates synthetic data
    python -m evox baseline --synthetic      # force synthetic data
    python -m evox baseline --rows 100000    # synthetic dataset size (if used)
"""

from __future__ import annotations

import argparse
import logging

from evox.config import ensure_output_dirs
from evox.models.baseline import train_random_forest, train_xgboost, save_model
from evox.data.loading import load_raw_data
from evox.evaluation.baseline import evaluate_model, compare_models
from evox.data.preprocessing import run_preprocessing_pipeline
from evox.data.synthetic import generate_synthetic_cicids

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="EVO-X Phase 1: static baseline IDS")
    parser.add_argument("--synthetic", action="store_true", help="Force use of synthetic data")
    parser.add_argument("--rows", type=int, default=50_000, help="Rows for synthetic data generation")
    args = parser.parse_args(argv)
    ensure_output_dirs()

    logger.info("=" * 70)
    logger.info("EVO-X Phase 1 — Static Baseline IDS Pipeline")
    logger.info("=" * 70)

    # --- 1. Data Acquisition -------------------------------------------
    if args.synthetic:
        raw_df = generate_synthetic_cicids(n_rows=args.rows)
    else:
        raw_df = load_raw_data()

    # --- 2. Preprocessing (clean -> time-order -> chronological split ---
    #        -> encode -> scale) ----------------------------------------
    prepared = run_preprocessing_pipeline(raw_df)

    # --- 3. Train static baselines --------------------------------------
    rf_model = train_random_forest(prepared.X_train, prepared.y_train)
    save_model(rf_model, "random_forest_baseline_v1")

    # --- 4. Evaluate on chronologically-held-out test set ----------------
    rf_results = evaluate_model(rf_model, prepared.X_test, prepared.y_test, prepared.label_encoder, "random_forest_baseline")
    all_results = [rf_results]

    try:
        xgb_model = train_xgboost(prepared.X_train, prepared.y_train)
        save_model(xgb_model, "xgboost_baseline_v1")
        xgb_results = evaluate_model(xgb_model, prepared.X_test, prepared.y_test, prepared.label_encoder, "xgboost_baseline")
        all_results.append(xgb_results)
    except ImportError as e:
        logger.warning("Skipping XGBoost baseline: %s", e)

    # --- 5. Compare -------------------------------------------------------
    comparison = compare_models(all_results)
    logger.info("\nModel comparison (overall metrics):\n%s", comparison.to_string())

    logger.info("=" * 70)
    logger.info("Phase 1 complete. See results/metrics/ and results/figures/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
