"""
EVO-X Phase 1 — Data Acquisition (Synopsis §7.3, Module 1)
=============================================================
Loads CICIDS2017 flow data. If one or more real CICIDS2017 CSVs are present
in data/raw/ (download separately from
https://www.unb.ca/cic/datasets/ids-2017.html and place the CSVs there),
they are concatenated and used. Otherwise, a synthetic stand-in dataset is
generated automatically (see src/synthetic_data.py) so the pipeline remains
runnable for development/testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import re

import pandas as pd

from src.config import RAW_DATA_DIR, LABEL_COLUMN, TIMESTAMP_COLUMN
from src.synthetic_data import generate_synthetic_cicids

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}


def _chronological_file_key(path: Path):
    """Sort key putting CICIDS2017 day-files in capture order.

    A plain alphabetical sort (what Phase 1 used) yields Friday, Monday,
    Thursday, Tuesday, Wednesday -- i.e. NOT chronological, which silently
    corrupts any time-ordered split when the CSVs carry no Timestamp column
    (as in the 'MachineLearningCVE' release).  Order: weekday, then
    Morning < Afternoon, then name.
    """
    name = path.name.lower()
    day = next((rank for d, rank in _WEEKDAYS.items() if name.startswith(d)), 99)
    half = 1 if "afternoon" in name else 0
    # Friday afternoon: the PortScan capture (~13:55) precedes DDoS (~15:56),
    # but alphabetically "DDos" < "PortScan".
    late = 1 if "ddos" in name else 0
    return (day, half, late, name)


def _find_real_csvs(raw_dir: Path) -> List[Path]:
    files = [p for p in raw_dir.glob("*.csv") if not p.name.startswith("synthetic_")]
    return sorted(files, key=_chronological_file_key)


def load_raw_data(
    raw_dir: Path = RAW_DATA_DIR,
    force_synthetic: bool = False,
    add_provenance: bool = False,
) -> pd.DataFrame:
    """Load and concatenate raw CICIDS2017 CSV(s).

    Returns a single DataFrame with (at minimum) a Label column and, where
    available, a Timestamp column. Column names are stripped of leading /
    trailing whitespace, which is a well-known quirk of the public
    CICIDS2017 CSV release (columns are shipped with leading spaces, e.g.
    " Label", " Destination Port").

    If ``add_provenance`` is True (used by Phase 2), two helper columns are
    added: ``_day_rank`` (index of the day-file in capture order) and
    ``_row_in_file`` (row position inside its file).  They let the stream
    builder recover time order when the CSVs have no Timestamp column.
    They are underscore-prefixed and are never used as model features.
    """
    real_csvs = [] if force_synthetic else _find_real_csvs(raw_dir)

    if real_csvs:
        logger.info("Found %d real CICIDS2017 CSV(s) in %s — loading real data.", len(real_csvs), raw_dir)
        frames = []
        for path in real_csvs:
            logger.info("  reading %s", path.name)
            df = pd.read_csv(path, low_memory=False, encoding="latin1")
            df.columns = [c.strip() for c in df.columns]
            if add_provenance:
                df["_day_rank"] = len(frames)
                df["_row_in_file"] = range(len(df))
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)
    else:
        logger.warning(
            "No real CICIDS2017 CSVs found in %s. Generating a synthetic "
            "stand-in dataset instead (see src/synthetic_data.py). Place "
            "real CICIDS2017 CSVs in that folder to use actual data.",
            raw_dir,
        )
        data = generate_synthetic_cicids()

    if LABEL_COLUMN not in data.columns:
        raise ValueError(
            f"Loaded data has no '{LABEL_COLUMN}' column after stripping "
            f"whitespace. Found columns: {list(data.columns)[:15]}..."
        )

    logger.info("Loaded data shape: %s", data.shape)
    logger.info("Class distribution:\n%s", data[LABEL_COLUMN].value_counts())
    return data


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(df.shape)
