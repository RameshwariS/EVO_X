"""
EVO-X -- Phase 3 / Experiment 1: static IDS under a time-ordered stream
(Synopsis §7.9 Phase 3, §8.3 Experiment 1, §8.6 seeds).

Train RF and XGBoost ONCE on ref_train, never adapt, replay the live stream
batch by batch, and record how performance decays.

    python phase3_static_ids.py                      # stream from data/processed/, else synthetic week
    python phase3_static_ids.py --synthetic          # force synthetic (pipeline check only)
    python phase3_static_ids.py --seeds 5 --window-batches 10 --max-train-per-class 100000

Outputs (results/):
    metrics/phase3_batch_log.csv         per (model, seed, batch): confusion counts, confidence, novel stats
    metrics/phase3_window_metrics.csv    per (model, seed, window): binary / known-class / novel metrics
    metrics/phase3_summary.csv           mean +/- std over seeds (the Experiment 1 table)
    metrics/phase3_flow_log_<model>_seed<k>.npz   y_pred + confidence per stream flow (input to Phase 6)
    figures/phase3_degradation.png
"""
from __future__ import annotations

import argparse
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, N_SEEDS, RANDOM_SEED, STREAM_DIR
from src.ids_model import IDSModel
from src.metrics_stream import binary_counts, binary_metrics, known_class_macro_f1, novel_detection
from src.stream_builder import EvoStream, build_stream
from src.stream_report import assert_valid, batch_composition
from src.synthetic_data import generate_synthetic_week

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("phase3")


def cap_per_class(X, y, cap, seed):
    """Optional per-class cap (keeps rare classes whole) to bound training time on the real dataset."""
    if not cap:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        keep.append(idx if len(idx) <= cap else np.sort(rng.choice(idx, cap, replace=False)))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def run_one(kind: str, seed: int, s: EvoStream, batch_size: int, window_batches: int, max_train_per_class,
            train_on_full_reference: bool = False):
    # Protocol A (default): train on ref_train, keep ref_val held out as a pre-deployment check.
    # Protocol B: train on the WHOLE reference (train+val) -> ref_val scores are then in-sample and not reported.
    X0, y0 = (s.X_ref, s.y_ref) if train_on_full_reference else (s.X_ref_train, s.y_ref_train)
    Xtr, ytr = cap_per_class(X0, y0, max_train_per_class, seed)
    model = IDSModel(kind, seed=seed).fit(Xtr, ytr)
    known = model.known_

    # replay
    y_pred = np.empty(len(s.y_stream), dtype=np.int16)
    conf = np.empty(len(s.y_stream), dtype=np.float32)
    batch_rows, t_inf = [], 0.0
    for b in s.iter_batches(batch_size):
        t0 = time.perf_counter()
        p, c = model.predict_with_confidence(b.X)
        t_inf += time.perf_counter() - t0
        y_pred[b.row_start:b.row_end], conf[b.row_start:b.row_end] = p, c
        nd = novel_detection(b.y, p, known)
        batch_rows.append(dict(model=kind, seed=seed, batch=b.index, n=len(b),
                               **binary_counts(b.y, p), mean_confidence=float(c.mean()),
                               error_rate=float((p != b.y).mean()), **{k: nd[k] for k in ("n_novel", "n_novel_flagged")}))
    latency_ms_per_1k = 1000 * t_inf / (len(s.y_stream) / 1000)

    # windowed metrics (non-overlapping windows of `window_batches` batches)
    win_rows, w = [], window_batches * batch_size
    for i, start in enumerate(range(0, len(s.y_stream), w)):
        sl = slice(start, min(start + w, len(s.y_stream)))
        yt, yp = s.y_stream[sl], y_pred[sl]
        bm, nd = binary_metrics(yt, yp), novel_detection(yt, yp, known)
        win_rows.append(dict(model=kind, seed=seed, window=i, batch_start=start // batch_size,
                             batch_end=(sl.stop - 1) // batch_size, n=len(yt), **{f"bin_{k}": v for k, v in bm.items()},
                             known_macro_f1=known_class_macro_f1(yt, yp, known), **nd,
                             mean_confidence=float(conf[sl].mean())))

    # reference-validation (pre-deployment) reference point
    pv = model.predict(s.X_ref_val)
    ref_bin, ref_known = binary_metrics(s.y_ref_val, pv), known_class_macro_f1(s.y_ref_val, pv, known)
    # whole-stream numbers + known-only view (separates novelty from covariate shift)
    st_bin = binary_metrics(s.y_stream, y_pred)
    km = np.isin(s.y_stream, known)
    st_bin_known = binary_metrics(s.y_stream[km], y_pred[km]) if km.any() else {"f1": float("nan"), "n_attack": 0}
    # an F1 over a slice with no attacks is undefined, not 0 -> report NaN plus the support
    if ref_bin["n_attack"] == 0 or train_on_full_reference:
        ref_bin["f1"] = float("nan")
        if train_on_full_reference:  # in-sample -> not a held-out measurement
            ref_known, ref_bin["fpr"] = float("nan"), float("nan")
    if st_bin_known["n_attack"] == 0:
        st_bin_known["f1"] = float("nan")
    summary = dict(model=kind, seed=seed, train_protocol="full_reference" if train_on_full_reference else "ref_train_only", n_train=model.n_train_, n_known_classes=len(known),
                   fit_seconds=model.fit_seconds_, latency_ms_per_1k_flows=latency_ms_per_1k,
                   refval_bin_f1=ref_bin["f1"], refval_n_attack=ref_bin["n_attack"], stream_known_only_n_attack=st_bin_known["n_attack"], refval_bin_fpr=ref_bin["fpr"], refval_known_macro_f1=ref_known,
                   stream_bin_f1=st_bin["f1"], stream_bin_recall=st_bin["recall"], stream_bin_fpr=st_bin["fpr"],
                   stream_known_only_bin_f1=st_bin_known["f1"],
                   stream_known_macro_f1=known_class_macro_f1(s.y_stream, y_pred, known),
                   stream_novel_flag_rate=novel_detection(s.y_stream, y_pred, known)["novel_flag_rate"],
                   stream_novel_share=float((~km).mean()))
    np.savez_compressed(METRICS_DIR / f"phase3_flow_log_{kind}_seed{seed}.npz", y_pred=y_pred, confidence=conf)
    return summary, pd.DataFrame(batch_rows), pd.DataFrame(win_rows)


def plot_degradation(s: EvoStream, win: pd.DataFrame, batch_size: int, window_batches: int, path):
    comp = batch_composition(s, batch_size)
    frac = comp.div(comp.sum(axis=1), axis=0).fillna(0)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
    for kind, g in win.groupby("model"):
        agg = g.groupby("window").agg(x=("batch_start", "first"), xe=("batch_end", "last"),
                                      f1=("bin_f1", "mean"), f1s=("bin_f1", "std"),
                                      rec=("bin_recall", "mean"), fpr=("bin_fpr", "mean"), conf=("mean_confidence", "mean"))
        x = (agg.x + agg.xe) / 2
        axes[0].plot(x, agg.rec, marker="o", ms=3, label=f"{kind} recall")
        axes[0].plot(x, agg.f1, ls="--", label=f"{kind} F1")
        axes[0].fill_between(x, agg.f1 - agg.f1s.fillna(0), agg.f1 + agg.f1s.fillna(0), alpha=.15)
        axes[1].plot(x, agg.conf, label=kind)
    axes[0].set_ylabel("binary attack detection"); axes[0].set_ylim(-0.02, 1.02); axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("EVO-X Experiment 1: static IDS on the time-ordered stream (mean over seeds)")
    axes[1].set_ylabel("mean confidence"); axes[1].legend(fontsize=7)
    cmap = plt.get_cmap("tab20")
    axes[2].stackplot(frac.index, frac.T.values, labels=frac.columns, colors=[cmap(i % 20) for i in range(frac.shape[1])])
    axes[2].set_ylabel("class share"); axes[2].set_xlabel(f"replay batch (size {batch_size}; windows of {window_batches} batches)")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="EVO-X Phase 3: static IDS baseline")
    ap.add_argument("--synthetic", action="store_true", help="build and use a synthetic stream (NOT real results)")
    ap.add_argument("--stream-dir", default=str(STREAM_DIR))
    ap.add_argument("--models", nargs="+", default=["random_forest", "xgboost"])
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--batch-size", type=int, default=None, help="default: the stream's own batch size")
    ap.add_argument("--window-batches", type=int, default=10)
    ap.add_argument("--max-train-per-class", type=int, default=None)
    ap.add_argument("--train-on-full-reference", action="store_true",
                    help="train on ref_train+ref_val (ref_val scores become in-sample and are not reported)")
    ap.add_argument("--dry-run", action="store_true", help="print dataset/training sizes and exit")
    a = ap.parse_args()

    if a.synthetic or not (STREAM_DIR / "stream.npz").exists():
        log.warning("SYNTHETIC stream in use -- validates the pipeline only; do NOT report these numbers.")
        s = build_stream(generate_synthetic_week(40_000), reference_fraction=0.4, batch_size=500)
        tag = "SYNTHETIC"
    else:
        s, tag = EvoStream.load(a.stream_dir), "REAL"
    assert_valid(s)
    bs = a.batch_size or int(s.meta["batch_size"])
    log.info("[%s] ref_train=%d ref_val=%d stream=%d  known classes in ref=%d/%d",
             tag, len(s.y_ref_train), len(s.y_ref_val), len(s.y_stream), len(s.reference_class_ids), len(s.classes))

    if a.dry_run:
        X0, y0 = (s.X_ref, s.y_ref) if a.train_on_full_reference else (s.X_ref_train, s.y_ref_train)
        _, yc = cap_per_class(X0, y0, a.max_train_per_class, RANDOM_SEED)
        counts = pd.Series(yc).map(dict(enumerate(s.classes))).value_counts()
        print(f"[{tag}] training rows after cap: {len(yc)}  (features: {s.n_features}, stream flows: {len(s.y_stream)})")
        print(counts.to_string())
        print(f"runs planned: {len(a.models)} models x {a.seeds} seeds = {len(a.models) * a.seeds}")
        return 0

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    sums, batches, wins = [], [], []
    for kind in a.models:
        for k in range(a.seeds):
            seed = RANDOM_SEED + k
            log.info("%s seed=%d ...", kind, seed)
            su, ba, wi = run_one(kind, seed, s, bs, a.window_batches, a.max_train_per_class, a.train_on_full_reference)
            su["data_tag"] = tag
            sums.append(su); batches.append(ba); wins.append(wi)
    S, B, W = pd.DataFrame(sums), pd.concat(batches), pd.concat(wins)
    B.to_csv(METRICS_DIR / "phase3_batch_log.csv", index=False)
    W.to_csv(METRICS_DIR / "phase3_window_metrics.csv", index=False)

    num = S.drop(columns=["seed"]).groupby("model")
    table = num.mean(numeric_only=True).add_suffix("_mean").join(num.std(numeric_only=True).add_suffix("_std"))
    table = table[sorted(table.columns, key=lambda c: (c.rsplit("_", 1)[0], c))]
    table.T.to_csv(METRICS_DIR / "phase3_summary.csv")
    S.to_csv(METRICS_DIR / "phase3_summary_per_seed.csv", index=False)
    plot_degradation(s, W, bs, a.window_batches, FIGURES_DIR / "phase3_degradation.png")

    show = ["refval_n_attack", "refval_bin_f1", "refval_bin_fpr", "stream_known_only_n_attack", "stream_bin_f1", "stream_bin_recall", "stream_bin_fpr",
            "stream_known_only_bin_f1", "stream_novel_flag_rate", "stream_novel_share", "fit_seconds", "latency_ms_per_1k_flows"]
    print(f"\n[{tag}] Experiment 1 summary (mean over {a.seeds} seeds)")
    print(S.groupby("model")[show].mean().T.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
