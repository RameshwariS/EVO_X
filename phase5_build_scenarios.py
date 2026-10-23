"""
EVO-X Phase 5 -- build the synthetic drift scenario library (Synopsis §7.9 Phase 5, Module 2)
============================================================================================
    python phase5_build_scenarios.py --synthetic            # pipeline check only (numbers are NOT results)
    python phase5_build_scenarios.py                        # real stream from data/processed/cicids2017_stream
    python phase5_build_scenarios.py --save-streams         # also write each scenario's arrays (large!)

Outputs
    data/processed/scenarios/<name>/drift_truth.json         ground truth for every scenario (reproducible from seed)
    results/metrics/phase5_library.csv                       one row per scenario
    results/metrics/phase5_magnitude_calibration.csv         static-model degradation per kind x magnitude
    results/figures/phase5_scenarios.png                     class composition + shift curve for the 6 types
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, SCENARIO_DIR, STREAM_DIR, TEST_SEEDS, VALIDATION_SEEDS
from src.drift_eval import static_run, train_static
from src.drift_scenarios import (KINDS, MAGNITUDES, LabSource, ScenarioSpec, batch_class_composition,
                                 generate_scenario, scenario_library)
from src.metrics_stream import binary_metrics
from src.stream_builder import EvoStream, build_stream
from src.synthetic_data import generate_synthetic_week

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("phase5")


def load_source(synthetic: bool, stream_dir):
    if synthetic or not (STREAM_DIR / "stream.npz").exists():
        log.warning("SYNTHETIC source -- validates the pipeline only; do NOT report these numbers.")
        s = build_stream(generate_synthetic_week(60_000, seed=2), reference_fraction=0.5, batch_size=500)
        return LabSource.from_evostream(s, seed=0), "SYNTHETIC"
    return LabSource.from_evostream(EvoStream.load(stream_dir), seed=0), "REAL"


def calibrate(src: LabSource, tag: str, seeds=VALIDATION_SEEDS) -> pd.DataFrame:
    """Plan risk: 'generator makes drift that is trivially detectable or undetectable'.  Measure how much each
    kind x magnitude actually degrades the STATIC model (validation seeds), so the levels can be judged/re-chosen."""
    rows = []
    for k in KINDS:
        for m in MAGNITUDES:
            for sd in seeds:
                sc = generate_scenario(src, ScenarioSpec(k, m, sd, "validation"))
                run = static_run(sc, train_static(sc))
                p = np.array(sc.truth["per_batch"]["p_b"]); c = sc.truth["change_batch"]
                bs = sc.spec.batch_size
                pre = slice(0, c * bs)
                post_b = np.flatnonzero(p >= 0.99)
                post = np.concatenate([np.arange(b * bs, (b + 1) * bs) for b in post_b])
                y, yp = sc.stream.y_stream, run.y_pred
                bp, bq = binary_metrics(y[pre], yp[pre]), binary_metrics(y[post], yp[post])
                rows.append(dict(kind=k, magnitude=m, seed=sd, data_tag=tag, err_pre=float((y[pre] != yp[pre]).mean()),
                                 err_post=float((y[post] != yp[post]).mean()), f1_pre=bp["f1"], f1_post=bq["f1"],
                                 fpr_pre=bp["fpr"], fpr_post=bq["fpr"],
                                 mix_tv=sc.truth["magnitude_detail"]["mix_tv_distance"],
                                 offset_sigma=sc.truth["magnitude_detail"]["offset_sigma"]))
    df = pd.DataFrame(rows)
    out = df.groupby(["kind", "magnitude"], sort=False).mean(numeric_only=True).drop(columns="seed")
    out["err_increase"] = out.err_post - out.err_pre
    out["f1_drop"] = out.f1_pre - out.f1_post
    return out.reset_index()


def figure(src: LabSource, path, tag: str):
    fig, axes = plt.subplots(2, 6, figsize=(24, 6.5), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    cmap = plt.get_cmap("tab20")
    for j, k in enumerate(KINDS):
        sc = generate_scenario(src, ScenarioSpec(k, "high", VALIDATION_SEEDS[0], "validation"))
        comp = batch_class_composition(sc).astype(float)
        frac = comp / comp.sum(axis=1, keepdims=True)
        present = [i for i in range(frac.shape[1]) if frac[:, i].any()]
        axes[0, j].stackplot(range(len(frac)), frac[:, present].T, labels=[sc.stream.classes[i] for i in present],
                             colors=[cmap(i % 20) for i in present])
        axes[0, j].set_title(k + " (high)"); axes[0, j].set_ylim(0, 1)
        if j == 0: axes[0, j].set_ylabel("class share per batch")
        axes[0, j].legend(fontsize=5, loc="lower left", ncol=2)
        p = np.array(sc.truth["per_batch"]["p_b"]); md = sc.truth["magnitude_detail"]
        shift = p * (md["mix_tv_distance"] + md["offset_sigma"] / 3.0)       # descriptive shift curve
        axes[1, j].plot(shift, color="k"); axes[1, j].set_xlabel("replay batch")
        for e in sc.truth["events"]:
            axes[1, j].axvspan(e["start_batch"], max(e["end_batch"], e["start_batch"] + .5), color="r" if e["scored"] else "gray", alpha=.3)
        if j == 0: axes[1, j].set_ylabel("shift (mix TV + sigma/3)")
    fig.suptitle(f"EVO-X synthetic drift scenarios (high magnitude)   [{tag}{' -- pipeline check, not a result' if tag == 'SYNTHETIC' else ''}]\n"
                 "red band = scored change event, grey = return-to-A (unscored)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--stream-dir", default=str(STREAM_DIR))
    ap.add_argument("--save-streams", action="store_true", help="write each scenario's arrays (~15 MB each)")
    ap.add_argument("--controls", type=int, default=2, help="drift-free control streams per family")
    a = ap.parse_args()
    src, tag = load_source(a.synthetic, a.stream_dir)
    log.info("[%s] source: %d train flows, classes %s; dropped (too few flows): %s", tag, len(src.y_train),
             [src.classes[c] for c in src.class_ids], src.dropped_classes)
    lib = scenario_library(VALIDATION_SEEDS, TEST_SEEDS, controls_per_family=a.controls)
    rows = []
    for spec in lib:
        sc = generate_scenario(src, spec)
        sc.save(SCENARIO_DIR / spec.name, with_stream=a.save_streams)
        t = sc.truth
        rows.append(dict(name=spec.name, kind=spec.kind, magnitude=spec.magnitude, seed=spec.seed, family=spec.family,
                         n_batches=spec.n_batches, n_events=sum(e["scored"] for e in t["events"]),
                         change_batch=t["change_batch"], mix_tv=t["magnitude_detail"]["mix_tv_distance"],
                         offset_sigma=t["magnitude_detail"]["offset_sigma"], novel=",".join(t["novel_classes"]), data_tag=tag))
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_DIR / "phase5_library.csv", index=False)
    log.info("wrote %d scenario truth files to %s", len(rows), SCENARIO_DIR)
    cal = calibrate(src, tag)
    cal.to_csv(METRICS_DIR / "phase5_magnitude_calibration.csv", index=False)
    print(cal[["kind", "magnitude", "mix_tv", "offset_sigma", "err_pre", "err_post", "err_increase", "f1_pre", "f1_post", "f1_drop"]].round(3).to_string(index=False))
    figure(src, FIGURES_DIR / "phase5_scenarios.png", tag)
    log.info("figure -> %s", FIGURES_DIR / "phase5_scenarios.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
