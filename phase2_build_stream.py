"""
EVO-X -- Phase 2 entry point: dataset preprocessing + time-ordered stream
construction (Synopsis §7.9 Phase 2, §7.6).

Usage:
    python phase2_build_stream.py                 # real CSVs in data/raw/ if present, else synthetic week
    python phase2_build_stream.py --synthetic     # force synthetic (pipeline check only)
    python phase2_build_stream.py --reference-fraction 0.5 --batch-size 2000
Outputs:
    data/processed/cicids2017_stream/{stream.npz, stream_meta.json}
    results/metrics/phase2_{class_timeline.csv, batch_composition.csv, stream_summary.json}
    results/figures/phase2_stream_overview.png
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import RAW_DATA_DIR, REFERENCE_FRACTION, REFERENCE_VAL_FRACTION, STREAM_BATCH_SIZE, STREAM_DIR
from src.data_loader import _find_real_csvs, load_raw_data
from src.stream_builder import build_stream
from src.stream_report import validate_stream, write_reports
from src.synthetic_data import generate_synthetic_week

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("phase2")


def main() -> int:
    ap = argparse.ArgumentParser(description="EVO-X Phase 2: build the time-ordered stream")
    ap.add_argument("--synthetic", action="store_true", help="force synthetic week (NOT real results)")
    ap.add_argument("--rows", type=int, default=60_000, help="synthetic rows")
    ap.add_argument("--reference-fraction", type=float, default=REFERENCE_FRACTION)
    ap.add_argument("--reference-val-fraction", type=float, default=REFERENCE_VAL_FRACTION)
    ap.add_argument("--batch-size", type=int, default=STREAM_BATCH_SIZE)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--no-pm-fix", action="store_true", help="disable the 12-hour timestamp fix")
    ap.add_argument("--out-dir", default=str(STREAM_DIR))
    a = ap.parse_args()

    real = [] if a.synthetic else _find_real_csvs(RAW_DATA_DIR)
    if real:
        log.info("Using %d real CSV(s) from %s", len(real), RAW_DATA_DIR)
        raw = load_raw_data(add_provenance=True)
        name = "CICIDS2017"
    else:
        log.warning("SYNTHETIC data in use -- validates the pipeline only; do NOT report these numbers.")
        raw = generate_synthetic_week(a.rows)
        name = "SYNTHETIC-CICIDS2017-week"

    stream = build_stream(raw, a.reference_fraction, a.reference_val_fraction, a.batch_size,
                          deduplicate=not a.no_dedup, fix_pm=not a.no_pm_fix, dataset_name=name)
    stream.save(a.out_dir)
    paths = write_reports(stream)

    checks = validate_stream(stream)
    for k, (ok, msg) in checks.items():
        log.info("  [%s] %s %s", "PASS" if ok else "FAIL", k, msg)
    log.info("Novel-class share of live stream: %.1f%%", 100 * stream.novel_mask().mean())
    log.info("Reports: %s", {k: str(v) for k, v in paths.items()})
    return 0 if all(ok for ok, _ in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
