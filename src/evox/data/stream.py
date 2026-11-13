"""
EVO-X Phase 2 -- Dataset preprocessing and time-ordered stream construction
============================================================================
Synopsis §7.9 Phase 2 ("Dataset preprocessing and time-ordered stream
construction") and §7.6 ("flows are re-ordered by timestamp ... avoiding a
random split relabelled as continual learning").

Output of this module is an :class:`EvoStream`:

    |<------ reference window ------>|<---------- live stream ---------->|
    |  ref_train (fit)  | ref_val    |  batch 0 | batch 1 | ... | batch N |

* The reference window is what the first ("current") IDS is trained on
  (Phase 3).  Its last part is held out as validation (threshold tuning,
  cheap proxy evaluation in the Evolution Controller, §7.4).
* The live stream is replayed batch-by-batch (Phase 4: Kafka; Phase 5:
  synthetic drift is injected on top of it).

Leakage rules enforced here (and checked in stream_report.validate_stream):
  1. Order is chronological; nothing is shuffled.
  2. Imputation medians and scaler statistics are fit on ref_train ONLY.
  3. Class ids are fixed up front (closed id-space) but the set of classes
     actually seen in the reference window is recorded, so a class that first
     appears in the stream is identifiable as *novel* (Phase 5/8).

Real-data quirks handled (all documented in the CICIDS2017 literature):
  * leading spaces in column names; duplicated 'Fwd Header Length' column
  * 'Infinity'/NaN in Flow Bytes/s and Flow Packets/s
  * 'Web Attack \x96 Brute Force' label encoding artefact
  * Timestamp is d/m/Y with 12-hour clock and NO AM/PM marker (13:00 is
    written "1:00"), so a naive sort puts afternoon flows before morning ones
  * the MachineLearningCVE release has no Timestamp at all -> fall back to
    capture-day file order (see data_loader._chronological_file_key)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from evox.config import (
    BENIGN_LABEL,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    REFERENCE_FRACTION,
    REFERENCE_VAL_FRACTION,
    STREAM_BATCH_SIZE,
    TIMESTAMP_COLUMN,
)

logger = logging.getLogger(__name__)

NAT_INT = np.iinfo(np.int64).min  # how NaT is stored in the .npz


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class StreamBatch:
    index: int
    X: np.ndarray
    y: np.ndarray
    row_start: int          # offset inside the live stream
    row_end: int
    t_start: Optional[pd.Timestamp] = None
    t_end: Optional[pd.Timestamp] = None

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class EvoStream:
    X_ref_train: np.ndarray
    y_ref_train: np.ndarray
    X_ref_val: np.ndarray
    y_ref_val: np.ndarray
    X_stream: np.ndarray
    y_stream: np.ndarray
    feature_names: List[str]
    classes: List[str]                       # id -> label; classes[0] == BENIGN
    t_ref: Optional[np.ndarray] = None       # datetime64[ns], ref_train+ref_val
    t_stream: Optional[np.ndarray] = None    # datetime64[ns]
    scaler_mean: Optional[np.ndarray] = None
    scaler_scale: Optional[np.ndarray] = None
    impute_median: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def n_features(self) -> int:
        return self.X_ref_train.shape[1]

    @property
    def reference_class_ids(self) -> List[int]:
        ids = np.union1d(np.unique(self.y_ref_train), np.unique(self.y_ref_val))
        return [int(i) for i in ids]

    @property
    def X_ref(self) -> np.ndarray:
        return np.vstack([self.X_ref_train, self.X_ref_val])

    @property
    def y_ref(self) -> np.ndarray:
        return np.concatenate([self.y_ref_train, self.y_ref_val])

    def novel_mask(self) -> np.ndarray:
        """True for stream flows whose class never occurs in the reference."""
        return ~np.isin(self.y_stream, self.reference_class_ids)

    def iter_batches(self, batch_size: Optional[int] = None) -> Iterator[StreamBatch]:
        """Yield the live stream in time order, ``batch_size`` flows at a time."""
        bs = batch_size or int(self.meta.get("batch_size", STREAM_BATCH_SIZE))
        n = len(self.y_stream)
        for i, start in enumerate(range(0, n, bs)):
            end = min(start + bs, n)
            t0 = t1 = None
            if self.t_stream is not None and len(self.t_stream) == n:
                t0, t1 = pd.Timestamp(self.t_stream[start]), pd.Timestamp(self.t_stream[end - 1])
            yield StreamBatch(i, self.X_stream[start:end], self.y_stream[start:end], start, end, t0, t1)

    def n_batches(self, batch_size: Optional[int] = None) -> int:
        bs = batch_size or int(self.meta.get("batch_size", STREAM_BATCH_SIZE))
        return int(np.ceil(len(self.y_stream) / bs))

    # -- persistence (npz + json: no pickle, so it is safe to share) --------
    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _t(a):
            return np.array([], dtype=np.int64) if a is None else a.astype("datetime64[ns]").astype(np.int64)

        np.savez_compressed(
            out_dir / "stream.npz",
            X_ref_train=self.X_ref_train, y_ref_train=self.y_ref_train,
            X_ref_val=self.X_ref_val, y_ref_val=self.y_ref_val,
            X_stream=self.X_stream, y_stream=self.y_stream,
            t_ref=_t(self.t_ref), t_stream=_t(self.t_stream),
            scaler_mean=self.scaler_mean, scaler_scale=self.scaler_scale,
            impute_median=self.impute_median,
        )
        meta = dict(self.meta)
        meta["feature_names"] = self.feature_names
        meta["classes"] = self.classes
        (out_dir / "stream_meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        logger.info("Saved stream to %s", out_dir)
        return out_dir

    @classmethod
    def load(cls, out_dir: Path) -> "EvoStream":
        out_dir = Path(out_dir)
        z = np.load(out_dir / "stream.npz")
        meta = json.loads((out_dir / "stream_meta.json").read_text())

        def _t(a):
            return None if len(a) == 0 else a.astype("datetime64[ns]")

        return cls(
            X_ref_train=z["X_ref_train"], y_ref_train=z["y_ref_train"],
            X_ref_val=z["X_ref_val"], y_ref_val=z["y_ref_val"],
            X_stream=z["X_stream"], y_stream=z["y_stream"],
            feature_names=meta.pop("feature_names"), classes=meta.pop("classes"),
            t_ref=_t(z["t_ref"]), t_stream=_t(z["t_stream"]),
            scaler_mean=z["scaler_mean"], scaler_scale=z["scaler_scale"],
            impute_median=z["impute_median"], meta=meta,
        )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# Step 1 -- cleaning
# ---------------------------------------------------------------------------
_ODD_DASH = re.compile(r"\s*[\x80-\x9f\u2013\u2014\ufffd]\s*")


def normalize_label(label: str) -> str:
    """'Web Attack \\x96 Brute Force' -> 'Web Attack - Brute Force'.
    Leaves genuine hyphens ('FTP-Patator') untouched."""
    label = _ODD_DASH.sub(" - ", str(label)).strip()
    return BENIGN_LABEL if label.lower() == "benign" else label  # CSE-CIC-IDS2018 uses 'Benign'


def _feature_columns(df: pd.DataFrame) -> List[str]:
    skip = set(NON_FEATURE_COLUMNS) | {LABEL_COLUMN, TIMESTAMP_COLUMN}
    return [
        c for c in df.columns
        if c not in skip and not c.startswith("_") and pd.api.types.is_numeric_dtype(df[c])
    ]


def clean_for_stream(df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
    """Leakage-free cleaning.  Unlike Phase 1's ``clean`` this does NOT impute
    with a whole-dataset median (that would use future data); imputation is
    deferred to ``build_stream`` where it is fit on the reference window."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # CICIDS2017 (TrafficLabelling) ships 'Fwd Header Length' twice; pandas
    # renames the second to 'Fwd Header Length.1'.
    dup = [c for c in df.columns if c.endswith(".1") and c[:-2] in df.columns]
    if dup:
        logger.info("Dropping duplicated columns: %s", dup)
        df = df.drop(columns=dup)

    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(normalize_label)

    # Numeric columns that were read as text (e.g. because of 'Infinity').
    skip = set(NON_FEATURE_COLUMNS) | {LABEL_COLUMN, TIMESTAMP_COLUMN}
    for c in df.columns:
        if c in skip or c.startswith("_") or pd.api.types.is_numeric_dtype(df[c]):
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().mean() >= 0.9:
            df[c] = conv

    feats = _feature_columns(df)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)

    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=feats + [LABEL_COLUMN], keep="first")
        logger.info("Dedup: %d -> %d rows (%d feature+label duplicates removed)", before, len(df), before - len(df))

    df[feats] = df[feats].astype(np.float32)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 -- timestamps and ordering
# ---------------------------------------------------------------------------
def parse_cicids_timestamps(series: pd.Series, fix_pm: bool = True) -> pd.Series:
    """Parse CICIDS2017 timestamps (d/m/Y H:M, 12-hour clock without AM/PM).

    The capture runs ~08:00-17:00 but the CSV writes 13:00 as "1:00".  If the
    parsed hours all fall in 0-12 and some are < 8, those are shifted +12h.
    Already-datetime input (e.g. ISO strings from the synthetic generator,
    whose hours go past 12) is left alone.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    ts = pd.to_datetime(series, format="%d/%m/%Y %H:%M", errors="coerce")
    bad = ts.isna() & series.notna()
    if bad.any():
        fallback = pd.to_datetime(series, dayfirst=True, errors="coerce", format="mixed")
        ts = ts.where(~bad, fallback)

    if fix_pm and ts.notna().any():
        hours = ts.dt.hour
        if hours.max() <= 12 and (hours < 8).any():
            n_fix = int((hours < 8).sum())
            logger.warning("Timestamp 12-hour quirk detected: shifting %d flows with hour<8 by +12h.", n_fix)
            ts = ts.where(~(hours < 8), ts + pd.Timedelta(hours=12))
    return ts


def establish_order(df: pd.DataFrame, fix_pm: bool = True) -> Tuple[pd.DataFrame, str]:
    """Return (time-ordered df, ordering_source).

    ordering_source is one of:
      'timestamp'  -- sorted by parsed Timestamp (stable, so same-minute flows
                      keep their file order)
      'file_order' -- no usable Timestamp; ordered by capture-day file, then
                      row position inside the file
      'row_order'  -- nothing to go on; input order assumed chronological
    """
    df = df.copy()
    if TIMESTAMP_COLUMN in df.columns:
        ts = parse_cicids_timestamps(df[TIMESTAMP_COLUMN], fix_pm=fix_pm)
        if ts.notna().mean() >= 0.99:
            df[TIMESTAMP_COLUMN] = ts
            n_bad = int(ts.isna().sum())
            if n_bad:
                logger.warning("Dropping %d rows with unparseable timestamps.", n_bad)
                df = df[ts.notna()]
            df = df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
            return df, "timestamp"
        logger.warning("Timestamp column present but <99%% parseable -- ignoring it.")
        df = df.drop(columns=[TIMESTAMP_COLUMN])

    if {"_day_rank", "_row_in_file"} <= set(df.columns):
        df = df.sort_values(["_day_rank", "_row_in_file"], kind="mergesort").reset_index(drop=True)
        logger.warning(
            "No usable Timestamp: ordered by capture-day file order. Prefer the CICIDS2017 "
            "'TrafficLabelling' (GeneratedLabelledFlows) CSVs, which include Timestamp."
        )
        return df, "file_order"

    logger.warning("No Timestamp and no file provenance: assuming input row order is chronological.")
    return df.reset_index(drop=True), "row_order"


# ---------------------------------------------------------------------------
# Step 3 -- split + scale + encode  ->  EvoStream
# ---------------------------------------------------------------------------
def build_stream(
    raw_df: pd.DataFrame,
    reference_fraction: float = REFERENCE_FRACTION,
    reference_val_fraction: float = REFERENCE_VAL_FRACTION,
    batch_size: int = STREAM_BATCH_SIZE,
    deduplicate: bool = True,
    fix_pm: bool = True,
    dataset_name: str = "CICIDS2017",
) -> EvoStream:
    if not 0.05 <= reference_fraction <= 0.9:
        raise ValueError("reference_fraction should be in [0.05, 0.9]")

    df, ordering_source = prepare_ordered(raw_df, deduplicate=deduplicate, fix_pm=fix_pm)
    return build_stream_from_ordered(
        df, ordering_source, reference_fraction, reference_val_fraction,
        batch_size, deduplicate, dataset_name)


def prepare_ordered(raw_df: pd.DataFrame, deduplicate: bool = True, fix_pm: bool = True):
    """clean + order.  Expensive on real data, so exposed separately so that
    several reference fractions can be compared without redoing it."""
    df = clean_for_stream(raw_df, deduplicate=deduplicate)
    return establish_order(df, fix_pm=fix_pm)


def build_stream_from_ordered(
    df: pd.DataFrame,
    ordering_source: str,
    reference_fraction: float = REFERENCE_FRACTION,
    reference_val_fraction: float = REFERENCE_VAL_FRACTION,
    batch_size: int = STREAM_BATCH_SIZE,
    deduplicate: bool = True,
    dataset_name: str = "CICIDS2017",
) -> EvoStream:
    n = len(df)

    n_ref = int(n * reference_fraction)
    n_ref_train = int(n_ref * (1 - reference_val_fraction))
    ref_train, ref_val, live = df.iloc[:n_ref_train], df.iloc[n_ref_train:n_ref], df.iloc[n_ref:]
    if len(ref_train) == 0 or len(live) == 0:
        raise ValueError("Dataset too small for the requested split.")
    logger.info("Stream split -> ref_train=%d ref_val=%d live_stream=%d (ordering: %s)",
                len(ref_train), len(ref_val), len(live), ordering_source)

    # features: drop only columns that are constant over the WHOLE dataset
    # (carry no information anywhere); a column constant in the reference but
    # not later is a genuine drift signal and must be kept.
    feats = _feature_columns(df)
    const = [c for c in feats if df[c].nunique(dropna=True) <= 1]
    feats = [c for c in feats if c not in const]
    if const:
        logger.info("Dropping %d globally-constant features: %s", len(const), const)

    # fit preprocessing statistics on ref_train ONLY
    train_vals = ref_train[feats].to_numpy(dtype=np.float64)
    median = np.nanmedian(train_vals, axis=0)
    median = np.where(np.isnan(median), 0.0, median)  # all-NaN column in ref
    filled = np.where(np.isnan(train_vals), median, train_vals)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)  # same convention as sklearn

    def transform(part: pd.DataFrame) -> np.ndarray:
        v = part[feats].to_numpy(dtype=np.float64)
        v = np.where(np.isnan(v), median, v)
        return ((v - mean) / scale).astype(np.float32)

    # closed id-space, BENIGN = 0
    labels = sorted(set(df[LABEL_COLUMN]) - {BENIGN_LABEL})
    classes = ([BENIGN_LABEL] if BENIGN_LABEL in set(df[LABEL_COLUMN]) else []) + labels
    cid = {c: i for i, c in enumerate(classes)}

    def enc(part: pd.DataFrame) -> np.ndarray:
        return part[LABEL_COLUMN].map(cid).to_numpy(dtype=np.int16)

    def times(part: pd.DataFrame):
        if ordering_source != "timestamp":
            return None
        return part[TIMESTAMP_COLUMN].to_numpy(dtype="datetime64[ns]")

    t_ref = None
    if ordering_source == "timestamp":
        t_ref = np.concatenate([times(ref_train), times(ref_val)])

    stream = EvoStream(
        X_ref_train=transform(ref_train), y_ref_train=enc(ref_train),
        X_ref_val=transform(ref_val), y_ref_val=enc(ref_val),
        X_stream=transform(live), y_stream=enc(live),
        feature_names=feats, classes=classes,
        t_ref=t_ref, t_stream=times(live),
        scaler_mean=mean, scaler_scale=scale, impute_median=median,
        meta=dict(
            dataset=dataset_name,
            ordering_source=ordering_source,
            reference_fraction=reference_fraction,
            reference_val_fraction=reference_val_fraction,
            batch_size=batch_size,
            n_total=n, n_ref_train=len(ref_train), n_ref_val=len(ref_val), n_stream=len(live),
            dropped_constant_features=const,
            deduplicated=deduplicate,
        ),
    )
    return stream
