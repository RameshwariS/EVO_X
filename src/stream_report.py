"""
EVO-X Phase 2 -- stream validation and diagnostics
===================================================
Two jobs:

1. ``validate_stream`` -- integrity checks that must pass before any later
   phase trusts the stream (ordering, no leakage, finite values).
2. Diagnostics (class timeline, per-batch composition, descriptive shift
   curve, overview figure) -- these tell you what the *natural* drift in the
   dataset looks like, and are the evidence for the reference-window design
   decision.  The shift curve is purely descriptive; real drift detection
   (ADWIN, Page-Hinkley, PSI, MMD) is Phase 6.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR
from src.stream_builder import EvoStream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------
def validate_stream(s: EvoStream) -> Dict[str, Tuple[bool, str]]:
    checks: Dict[str, Tuple[bool, str]] = {}

    def add(name: str, ok: bool, msg: str = ""):
        checks[name] = (bool(ok), msg)

    n_feat = s.n_features
    add("shapes_consistent",
        s.X_ref_val.shape[1] == n_feat == s.X_stream.shape[1] == len(s.feature_names)
        and len(s.X_ref_train) == len(s.y_ref_train)
        and len(s.X_ref_val) == len(s.y_ref_val)
        and len(s.X_stream) == len(s.y_stream),
        f"{n_feat} features")

    finite = all(np.isfinite(a).all() for a in (s.X_ref_train, s.X_ref_val, s.X_stream))
    add("all_values_finite", finite, "no NaN/inf in any split")

    ys = np.concatenate([s.y_ref_train, s.y_ref_val, s.y_stream])
    add("labels_in_range", ys.min() >= 0 and ys.max() < len(s.classes), f"{len(s.classes)} classes")
    add("benign_is_class_0", s.classes[0] == "BENIGN" if "BENIGN" in s.classes else True)

    if s.t_stream is not None and s.t_ref is not None:
        t_all = np.concatenate([s.t_ref, s.t_stream])
        mono = bool((np.diff(t_all.astype("datetime64[ns]").astype(np.int64)) >= 0).all())
        add("time_monotonic", mono, "reference+stream timestamps non-decreasing")
        add("reference_precedes_stream", s.t_ref.max() <= s.t_stream.min(),
            f"ref ends {pd.Timestamp(s.t_ref.max())}, stream starts {pd.Timestamp(s.t_stream.min())}")
    else:
        add("time_monotonic", True,
            f"skipped: ordering_source={s.meta.get('ordering_source')} (no timestamps)")

    # leakage: ref_train must be (approximately) standardised, because the
    # scaler was fit on it and nothing else.
    sd = s.X_ref_train.std(axis=0)
    mu = s.X_ref_train.mean(axis=0)
    nz = sd > 1e-6
    add("scaler_fit_on_ref_train_only",
        bool(np.allclose(mu[nz], 0, atol=1e-2) and np.allclose(sd[nz], 1, atol=1e-2)),
        "ref_train mean~0, std~1 on non-constant features")
    return checks


def assert_valid(s: EvoStream) -> None:
    failed = {k: v[1] for k, v in validate_stream(s).items() if not v[0]}
    if failed:
        raise AssertionError(f"Stream validation failed: {failed}")


# ---------------------------------------------------------------------------
# 2. Diagnostics
# ---------------------------------------------------------------------------
def batch_composition(s: EvoStream, batch_size: int | None = None) -> pd.DataFrame:
    """Rows = replay batch index, columns = class name, values = flow counts."""
    bs = batch_size or int(s.meta["batch_size"])
    idx = np.arange(len(s.y_stream)) // bs
    comp = pd.crosstab(idx, s.y_stream).reindex(columns=range(len(s.classes)), fill_value=0)
    comp.columns = s.classes
    comp.index.name = "batch"
    return comp


def class_timeline(s: EvoStream, batch_size: int | None = None) -> pd.DataFrame:
    """Per-class: how many flows in reference vs stream, and when the class
    first shows up in the stream.  Classes with 0 reference flows are the
    naturally-occurring *novel-class* events."""
    bs = batch_size or int(s.meta["batch_size"])
    comp = batch_composition(s, bs)
    ref_counts = np.bincount(s.y_ref, minlength=len(s.classes))
    rows = []
    for i, name in enumerate(s.classes):
        col = comp[name]
        seen = col[col > 0]
        first = int(seen.index[0]) if len(seen) else None
        n_ref, n_stream = int(ref_counts[i]), int(col.sum())
        status = ("absent" if n_ref + n_stream == 0 else
                  "known" if n_ref > 0 else "NOVEL (first seen in stream)")
        first_t = None
        if first is not None and s.t_stream is not None:
            first_t = str(pd.Timestamp(s.t_stream[first * bs]))
        rows.append(dict(class_id=i, label=name, ref_flows=n_ref, stream_flows=n_stream,
                         first_stream_batch=first, first_stream_time=first_t, status=status))
    return pd.DataFrame(rows)


def shift_curve(s: EvoStream, batch_size: int | None = None) -> np.ndarray:
    """Mean |z| of the per-feature batch mean (z-scored with reference
    statistics).  ~0 means 'looks like the reference'.  DESCRIPTIVE only."""
    bs = batch_size or int(s.meta["batch_size"])
    return np.array([np.abs(b.X.mean(axis=0)).mean() for b in s.iter_batches(bs)])


def plot_overview(s: EvoStream, out_path: Path, batch_size: int | None = None) -> Path:
    comp = batch_composition(s, batch_size)
    frac = comp.div(comp.sum(axis=1), axis=0).fillna(0)
    curve = shift_curve(s, batch_size)
    ref_level = float(np.abs(s.X_ref_val.mean(axis=0)).mean())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    cmap = plt.get_cmap("tab20")
    ax1.stackplot(frac.index, frac.T.values, labels=frac.columns,
                  colors=[cmap(i % 20) for i in range(frac.shape[1])])
    ax1.set_ylabel("class share per batch")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"EVO-X Phase 2: live-stream class composition ({s.meta.get('dataset')}, "
                  f"ordering={s.meta.get('ordering_source')})")
    ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)

    ax2.plot(curve, lw=1)
    ax2.axhline(ref_level, color="grey", ls="--", lw=1, label="reference-validation level")
    ax2.set_ylabel("mean |z| of batch mean")
    ax2.set_xlabel(f"replay batch (size {batch_size or s.meta['batch_size']})")
    ax2.set_title("Descriptive feature-shift vs reference (not a drift detector)", fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def write_reports(s: EvoStream, tag: str = "phase2") -> Dict[str, Path]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    tl = class_timeline(s)
    paths["class_timeline"] = METRICS_DIR / f"{tag}_class_timeline.csv"
    tl.to_csv(paths["class_timeline"], index=False)

    comp = batch_composition(s)
    paths["batch_composition"] = METRICS_DIR / f"{tag}_batch_composition.csv"
    comp.to_csv(paths["batch_composition"])

    checks = validate_stream(s)
    summary = dict(
        meta=s.meta,
        n_features=s.n_features,
        classes=s.classes,
        reference_class_ids=s.reference_class_ids,
        novel_class_stream_flows=int(s.novel_mask().sum()),
        novel_class_stream_share=float(s.novel_mask().mean()),
        n_batches=s.n_batches(),
        validation={k: {"passed": v[0], "detail": v[1]} for k, v in checks.items()},
    )
    paths["summary"] = METRICS_DIR / f"{tag}_stream_summary.json"
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str))

    paths["figure"] = plot_overview(s, FIGURES_DIR / f"{tag}_stream_overview.png")
    return paths
