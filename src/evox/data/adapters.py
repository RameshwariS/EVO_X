"""
EVO-X Phase 2 (Part B) -- adapters for UNSW-NB15 and CSE-CIC-IDS2018
=====================================================================
Needed for RQ7 (cross-dataset generalisation, Synopsis §4.4 / §7.6).
Each adapter returns a DataFrame in the *canonical* schema expected by
``stream_builder.build_stream``: numeric feature columns, a ``Label`` column
(multi-class, BENIGN for normal traffic) and, where available, ``Timestamp``.

Both adapters are written from the public dataset documentation and have
only been tested on small synthetic frames with the same column layout --
verify against the real files (see the checklist in PHASE2_PARTB.md).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from evox.config import BENIGN_LABEL, LABEL_COLUMN, TIMESTAMP_COLUMN

logger = logging.getLogger(__name__)

# Column order of the four raw UNSW-NB15 CSVs (UNSW-NB15_1..4.csv have NO header row).
UNSW_RAW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
]
_UNSW_CATEGORICAL = ["proto", "state", "service"]
_UNSW_ID = ["srcip", "sport", "dstip", "dsport", "Ltime"]
_UNSW_CAT_FIX = {"backdoors": "Backdoor"}


def normalize_unsw_attack_cat(v) -> str:
    """Empty -> BENIGN; strip whitespace ('Fuzzers ' in the raw data); 'Backdoors' -> 'Backdoor'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return BENIGN_LABEL
    v = str(v).strip()
    if v == "" or v.lower() in {"normal", "nan"}:
        return BENIGN_LABEL
    return _UNSW_CAT_FIX.get(v.lower(), v)


def adapt_unsw_nb15_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise a UNSW-NB15 frame that already has the raw column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    multiclass = df["attack_cat"].map(normalize_unsw_attack_cat)
    # Stime = start time, epoch seconds -> Timestamp (the ordering key)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(pd.to_numeric(df["Stime"], errors="coerce"), unit="s")
    drop = [c for c in _UNSW_ID + _UNSW_CATEGORICAL + ["attack_cat", "Label", "Stime"] if c in df.columns]
    # NOTE: proto/state/service are categorical and are dropped in this v1
    # adapter (one-hot encoding with top-k levels fitted on the reference
    # window only is a possible extension; it must not use the stream).
    logger.info("UNSW-NB15: dropping id/categorical columns %s", drop)
    df = df.drop(columns=drop)
    df[LABEL_COLUMN] = multiclass  # NB: replaces the raw 0/1 'Label' column dropped above
    for c in df.columns:
        if c not in (LABEL_COLUMN, TIMESTAMP_COLUMN):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_unsw_nb15(raw_dir: Path) -> pd.DataFrame:
    """Load UNSW-NB15_1.csv .. _4.csv (headerless) from ``raw_dir``."""
    files = sorted(Path(raw_dir).glob("UNSW-NB15_[1-4].csv"))
    if not files:
        raise FileNotFoundError(f"No UNSW-NB15_[1-4].csv in {raw_dir}")
    frames = [pd.read_csv(f, header=None, names=UNSW_RAW_COLUMNS, low_memory=False, encoding="latin1")
              for f in files]
    return adapt_unsw_nb15_frame(pd.concat(frames, ignore_index=True))


def adapt_cse2018_frame(df: pd.DataFrame, subsample_every: int = 1) -> pd.DataFrame:
    """Canonicalise a CSE-CIC-IDS2018 frame (one day-file or several concatenated)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # known quirk: header row repeated inside some files
    df = df[df[LABEL_COLUMN].astype(str).str.strip() != LABEL_COLUMN]
    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], dayfirst=True,
                                              errors="coerce", format="mixed")
    if subsample_every > 1:  # systematic every-k-th row keeps time order intact
        df = df.iloc[::subsample_every]
    for c in df.columns:
        if c not in (LABEL_COLUMN, TIMESTAMP_COLUMN, "Flow ID", "Src IP", "Dst IP"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def load_cse_cic_ids2018(raw_dir: Path, subsample_every: int = 1) -> pd.DataFrame:
    """Load all CSVs in ``raw_dir``.  The full dataset is ~16M rows; use
    ``subsample_every`` (e.g. 10) unless you have >32 GB RAM."""
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {raw_dir}")
    frames = []
    for f in files:
        logger.info("reading %s", f.name)
        frames.append(adapt_cse2018_frame(pd.read_csv(f, low_memory=False, encoding="latin1"), subsample_every))
    return pd.concat(frames, ignore_index=True)
