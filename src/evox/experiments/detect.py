"""
EVO-X Phase 6 -- detector comparison + drift-analyzer evaluation (Synopsis §7.9 Phase 6, Objectives 2 & 3)
=========================================================================================================
    python -m evox detect --synthetic        # pipeline check (numbers are NOT results)
    python -m evox detect                    # real stream

Protocol (written down BEFORE any test scenario is touched):
  1. Build validation scenarios (VALIDATION_SEEDS) and test scenarios (TEST_SEEDS), disjoint pools.
  2. Tune each detector's single operating parameter on VALIDATION ONLY with the pre-registered criterion
        J = detection_rate - 0.02*false_alarms_per_1000_batches - 0.002*mean_delay      (drift_eval.criterion)
  3. Tune fusion weights (grid {0,.25,.5,1}^4, normalised) and theta on VALIDATION ONLY.
  4. FREEZE -> results/metrics/phase6_tuned_config.json   (written before step 5)
  5. Evaluate on TEST with the frozen config; analyzer on every event the fused detector detects.
Label policy: error-based detectors see labels delayed by LABEL_DELAY_BATCHES; label-free ones are unaffected.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from evox.experiments.scenarios import load_source
from evox.config import ensure_output_dirs
from evox.config import FIGURES_DIR, LABEL_DELAY_BATCHES, METRICS_DIR, STREAM_DIR, TEST_SEEDS, VALIDATION_SEEDS
from evox.drift.analyzer import DriftAnalyzer
from evox.evaluation.drift import (ALL_NAMES, MEMBER_NAMES, MemberConfig, aggregate, compute_signals, criterion, fused_alarms,
                            run_member, run_members, score_alarms, static_run, train_static)
from evox.drift.scenarios import KINDS, MAGNITUDES, generate_scenario, scenario_library

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("phase6")
TOL = 10

# one tunable parameter set per detector (grids are part of the pre-registration)
GRIDS = {
    "adwin_err": [dict(adwin_delta=d) for d in (0.0005, 0.002, 0.01, 0.05)],
    "ph_err": [dict(ph_delta=dl, ph_threshold=th) for dl in (0.002, 0.005) for th in (0.05, 0.1, 0.25, 0.5)],
    "psi": [dict(psi_threshold=t) for t in (0.05, 0.1, 0.25, 0.5)],
    "mmd": [dict(mmd_mult=m) for m in (1.0, 1.5, 3.0, 6.0)],
    "adwin_conf": [dict(adwin_conf_delta=d) for d in (0.0005, 0.002, 0.01)],
}
W_GRID = (0.0, 0.25, 0.5, 1.0)
THETAS = (0.25, 0.4, 0.5, 0.6, 0.75, 0.9)


def build_all(src, specs, tag):
    out = []
    for i, sp in enumerate(specs):
        sc = generate_scenario(src, sp)
        run = static_run(sc, train_static(sc))
        out.append((sc, compute_signals(sc, run)))
        if (i + 1) % 25 == 0:
            log.info("[%s] %d/%d scenarios prepared", tag, i + 1, len(specs))
    return out


def evaluate(prepared, name, cfg, k=LABEL_DELAY_BATCHES, members_cache=None):
    res = []
    for j, (sc, sig) in enumerate(prepared):
        o = run_member(name, sig, cfg, k) if members_cache is None else members_cache[j][name]
        res.append((sc.truth, score_alarms(o.alarms, sc.truth, TOL)))
    return res


def tune_members(val):
    best_cfg, rows = {}, []
    for name, grid in GRIDS.items():
        best = None
        for g in grid:
            cfg = MemberConfig(**g)
            agg = aggregate(evaluate(val, name, cfg))
            J = criterion(agg)
            rows.append(dict(detector=name, **g, J=J, **{k: agg[k] for k in ("detection_rate", "mean_delay", "fa_per_1000", "n_missed")}))
            if best is None or (J, -agg["fa_per_1000"]) > (best[0], -best[1]["fa_per_1000"]):
                best = (J, agg, g)
        best_cfg[name] = best[2]
        log.info("tuned %-10s -> %s  (J=%.3f det=%.2f delay=%.1f FA/1000=%.1f)", name, best[2], best[0],
                 best[1]["detection_rate"], best[1]["mean_delay"], best[1]["fa_per_1000"])
    return best_cfg, pd.DataFrame(rows)


def merged_cfg(best_cfg) -> MemberConfig:
    d = {}
    for g in best_cfg.values():
        d.update(g)
    return MemberConfig(**d)


def tune_fusion(val, cfg):
    cache = [run_members(sig, cfg, ALL_NAMES) for _, sig in val]
    ws, seen = [], set()
    for w in itertools.product(W_GRID, repeat=len(MEMBER_NAMES)):
        if sum(w) == 0:
            continue
        key = tuple(np.round(np.array(w) / sum(w), 6))
        if key not in seen:
            seen.add(key); ws.append(dict(zip(MEMBER_NAMES, w)))
    rows, best = [], None
    for w in ws:
        for th in THETAS:
            res = [(sc.truth, score_alarms(fused_alarms(m, w, th).alarms, sc.truth, TOL)) for (sc, _), m in zip(val, cache)]
            agg = aggregate(res); J = criterion(agg)
            rows.append(dict(**{f"w_{k}": v / sum(w.values()) for k, v in w.items()}, theta=th, J=J,
                             **{k: agg[k] for k in ("detection_rate", "mean_delay", "fa_per_1000", "n_missed")}))
            if best is None or (J, -agg["fa_per_1000"]) > (best[0], -best[1]["fa_per_1000"]):
                best = (J, agg, w, th)
    log.info("fusion tuned -> weights %s theta %.2f (J=%.3f det=%.2f delay=%.1f FA/1000=%.1f)", best[2], best[3], best[0],
             best[1]["detection_rate"], best[1]["mean_delay"], best[1]["fa_per_1000"])
    return best[2], best[3], pd.DataFrame(rows)


def per_group(res_list, specs_truth):
    rows = []
    for key in ("kind", "magnitude"):
        for g in sorted({t[key] for t in specs_truth}, key=lambda x: (KINDS + ("stationary",)).index(x) if key == "kind" else MAGNITUDES.index(x)):
            sub = [(t, r) for t, r in res_list if t[key] == g]
            if sub:
                rows.append(dict(group=key, value=g, **aggregate(sub)))
    return rows


def wilcoxon_p(a, b):
    d = np.array(a) - np.array(b)
    if np.allclose(d, 0):
        return float("nan")
    try:
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def analyzer_eval(test, cfg, weights, theta, k):
    rows = []
    type_truth = lambda kind, ev: ("temporary" if kind == "temporary" else "gradual" if kind in ("gradual", "incremental")
                                   else "recurring" if ev.get("recurrence") else "abrupt")
    for sc, sig in test:
        if not sc.truth["events"]:
            continue
        mem = run_members(sig, cfg, ALL_NAMES, k)
        f = fused_alarms(mem, weights, theta)
        res = score_alarms(f.alarms, sc.truth, TOL)
        model = train_static(sc)
        an = DriftAnalyzer(sig.ref_psi_detector, cfg.psi_threshold, cfg.mmd_mult * sig.mmd_null_q99,
                           ref_val_conf=model.predict_with_confidence(sc.stream.X_ref_val)[1])
        nb, bs = sc.spec.n_batches, sc.spec.batch_size
        Xb = sc.stream.X_stream.reshape(nb, bs, -1)
        for e in sorted([e for e in res["events"] if e["detected"]], key=lambda e: e["alarm_batch"]):
            ev = next(x for x in sc.truth["events"] if x["id"] == e["event_id"])
            s = an.characterize(e["alarm_batch"], min(e["alarm_batch"] + 8, nb - 1), sig.psi_stat, sig.mmd_stat, Xb, sig.run.conf_batches)
            rows.append(dict(scenario=sc.truth["name"], kind=sc.truth["kind"], magnitude=sc.truth["magnitude"], seed=sc.truth["seed"],
                             event_id=e["event_id"], true_start=ev["start_batch"], alarm_batch=e["alarm_batch"],
                             onset_estimate=s.onset_estimate, onset_error=s.onset_estimate - ev["start_batch"],
                             magnitude_sig=s.magnitude, persistence=s.persistence, ended=s.ended, novelty=s.novelty,
                             low_conf_share=s.low_conf_share, recurrence=s.recurrence, type_guess=s.type_guess,
                             type_truth=type_truth(sc.truth["kind"], ev), is_second_onset=bool(ev.get("recurrence"))))
    return pd.DataFrame(rows)


def figure_comparison(tab, path, tag):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    names = list(tab.detector)
    for a, (col, ttl) in zip(ax, (("detection_rate", "detection rate (higher better)"), ("mean_delay", "mean delay, batches (lower better)"),
                                  ("fa_per_1000", "false alarms / 1000 batches (lower better)"))):
        a.bar(names, tab[col], color=["#888"] * (len(names) - 1) + ["#c33"]); a.set_title(ttl, fontsize=9)
        a.tick_params(axis="x", rotation=40, labelsize=8)
    fig.suptitle(f"Detector comparison on TEST scenarios [{tag}{' -- pipeline check, not a result' if tag == 'SYNTHETIC' else ''}]", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def figure_timeline(item, cfg, weights, theta, k, path, tag):
    sc, sig = item
    mem = run_members(sig, cfg, ALL_NAMES, k); f = fused_alarms(mem, weights, theta)
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(sig.run.batch_err, label="static-model error rate"); ax[0].set_ylabel("error")
    ax[1].plot(sig.psi_stat / cfg.psi_threshold, label="PSI / thr"); ax[1].plot(np.maximum(sig.mmd_stat, 0) / (cfg.mmd_mult * sig.mmd_null_q99), label="MMD / thr")
    ax[1].axhline(1, color="k", lw=.5); ax[1].set_yscale("symlog"); ax[1].legend(fontsize=7); ax[1].set_ylabel("score")
    for nm in ALL_NAMES:
        ax[2].plot(mem[nm].score, label=nm, lw=1)
    ax[2].legend(fontsize=6, ncol=5); ax[2].set_ylabel("member score")
    rows = list(ALL_NAMES) + ["fused"]
    for i, nm in enumerate(rows):
        al = f.alarms if nm == "fused" else mem[nm].alarms
        ax[3].scatter(al, [i] * len(al), marker="|", s=200)
    ax[3].set_yticks(range(len(rows))); ax[3].set_yticklabels(rows, fontsize=7); ax[3].set_xlabel("replay batch")
    for a in ax:
        for e in sc.truth["events"]:
            a.axvspan(e["start_batch"], max(e["end_batch"], e["start_batch"] + .5), color="r" if e["scored"] else "gray", alpha=.25)
    fig.suptitle(f"{sc.truth['name']}  [{tag}] -- red = scored truth event, grey = unscored revert", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--stream-dir", default=str(STREAM_DIR))
    ap.add_argument("--controls", type=int, default=2)
    a = ap.parse_args(argv)
    ensure_output_dirs()
    t0 = time.time()
    src, tag = load_source(a.synthetic, a.stream_dir)
    lib = scenario_library(VALIDATION_SEEDS, TEST_SEEDS, controls_per_family=a.controls)
    val_specs, test_specs = [s for s in lib if s.family == "validation"], [s for s in lib if s.family == "test"]
    log.info("[%s] %d validation / %d test scenarios (label delay k=%d batches, tolerance %d)", tag, len(val_specs), len(test_specs),
             LABEL_DELAY_BATCHES, TOL)
    val = build_all(src, val_specs, "val")

    # ---- 2/3: tune on VALIDATION only ------------------------------------------------------------------------
    best_cfg, sens = tune_members(val)
    cfg = merged_cfg(best_cfg)
    w, theta, fsens = tune_fusion(val, cfg)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    sens.to_csv(METRICS_DIR / "phase6_sensitivity_members.csv", index=False)
    fsens.to_csv(METRICS_DIR / "phase6_sensitivity_fusion.csv", index=False)
    # ---- 4: FREEZE before the test scenarios are generated -----------------------------------------------------
    frozen = dict(data_tag=tag, label_delay_batches=LABEL_DELAY_BATCHES, tolerance=TOL, members=best_cfg,
                  fusion=dict(weights=w, theta=theta), criterion="J = det_rate - 0.02*FA/1000 - 0.002*mean_delay",
                  validation_seeds=list(VALIDATION_SEEDS), test_seeds=list(TEST_SEEDS), written_before_test_scenarios_built=True)
    (METRICS_DIR / "phase6_tuned_config.json").write_text(json.dumps(frozen, indent=1))
    log.info("frozen config written; building TEST scenarios now")

    # ---- 5: TEST ------------------------------------------------------------------------------------------------
    test = build_all(src, test_specs, "test")
    cache = [run_members(sig, cfg, ALL_NAMES) for _, sig in test]
    systems = {nm: [(sc.truth, score_alarms(c[nm].alarms, sc.truth, TOL)) for (sc, _), c in zip(test, cache)] for nm in ALL_NAMES}
    systems["fused"] = [(sc.truth, score_alarms(fused_alarms(c, w, theta).alarms, sc.truth, TOL)) for (sc, _), c in zip(test, cache)]
    tab = pd.DataFrame([dict(detector=nm, needs_labels=nm in ("adwin_err", "ph_err", "fused"), **aggregate(r)) for nm, r in systems.items()])
    tab.insert(1, "data_tag", tag)
    tab.to_csv(METRICS_DIR / "phase6_detector_comparison.csv", index=False)
    print("\n== TEST: detector comparison (tuned on validation) =="); print(tab.round(3).to_string(index=False))

    per = []
    for nm, r in systems.items():
        for row in per_group(r, [t for t, _ in r]):
            per.append(dict(detector=nm, **row))
    pd.DataFrame(per).to_csv(METRICS_DIR / "phase6_detector_by_scenario.csv", index=False)
    pk = pd.DataFrame(per); pk = pk[pk.group == "kind"].pivot(index="detector", columns="value", values="detection_rate")
    print("\n== TEST: detection rate by drift kind =="); print(pk.round(2).to_string())
    pm = pd.DataFrame(per); pm = pm[pm.group == "kind"].pivot(index="detector", columns="value", values="mean_delay")
    print("\n== TEST: mean delay (batches) by drift kind =="); print(pm.round(1).to_string())

    # mean +/- std over seeds (the plan: >=5 seeds; here 6)
    sd_rows = []
    for nm, r in systems.items():
        for seed in TEST_SEEDS:
            sub = [(t, x) for t, x in r if t["seed"] == seed and t["kind"] != "stationary"]
            g = aggregate(sub); sd_rows.append(dict(detector=nm, seed=seed, detection_rate=g["detection_rate"], mean_delay=g["mean_delay"], fa_per_1000=g["fa_per_1000"]))
    sd = pd.DataFrame(sd_rows).groupby("detector").agg(["mean", "std"])
    sd.columns = ["_".join(c) for c in sd.columns]; sd.drop(columns=[c for c in sd.columns if c.startswith("seed")], errors="ignore").to_csv(METRICS_DIR / "phase6_detector_seed_stats.csv")

    # paired Wilcoxon: fused vs each single detector over matched test scenarios (per-scenario detected fraction / false alarms)
    sel = [i for i, (t, _) in enumerate(systems["fused"]) if t["kind"] != "stationary"]
    wrows = []
    for nm in ALL_NAMES:
        for metric, f in (("detected_frac", lambda r: r["n_detected"] / max(r["n_events"], 1)), ("false_alarms", lambda r: r["n_false"])):
            av = [f(systems["fused"][i][1]) for i in sel]; bv = [f(systems[nm][i][1]) for i in sel]
            wrows.append(dict(fused_vs=nm, metric=metric, fused_mean=np.mean(av), other_mean=np.mean(bv), wilcoxon_p=wilcoxon_p(av, bv), n_pairs=len(sel)))
    pd.DataFrame(wrows).to_csv(METRICS_DIR / "phase6_fusion_vs_single_wilcoxon.csv", index=False)

    # label-delay sensitivity (error detectors only; NOT re-tuned)
    lrows = []
    for kk in (0, 2, 5):
        for nm in ("adwin_err", "ph_err"):
            g = aggregate(evaluate(test, nm, cfg, k=kk)); lrows.append(dict(label_delay=kk, detector=nm, **{x: g[x] for x in ("detection_rate", "mean_delay", "fa_per_1000")}))
    ld = pd.DataFrame(lrows); ld.to_csv(METRICS_DIR / "phase6_label_delay_sensitivity.csv", index=False)
    print("\n== label-delay sensitivity (not re-tuned) =="); print(ld.round(2).to_string(index=False))

    # ---- analyzer on every detected event ---------------------------------------------------------------------
    sig_df = analyzer_eval(test, cfg, w, theta, LABEL_DELAY_BATCHES)
    sig_df.insert(0, "data_tag", tag)
    sig_df.to_csv(METRICS_DIR / "phase6_signatures.csv", index=False)
    if len(sig_df):
        acc = (sig_df.type_guess == sig_df.type_truth).mean()
        cm = pd.crosstab(sig_df.type_truth, sig_df.type_guess)
        print(f"\n== analyzer: type_guess accuracy {acc:.2f} over {len(sig_df)} detected events (look-ahead 8 batches) =="); print(cm.to_string())
        cm.to_csv(METRICS_DIR / "phase6_type_confusion.csv")
        rho = {}
        for k_ in KINDS:
            s_ = sig_df[sig_df.kind == k_]
            if len(s_) > 3:
                r_, _ = spearmanr(s_.magnitude.map(MAGNITUDES.index), s_.magnitude_sig); rho[k_] = round(float(r_), 2)
        print("Spearman(magnitude level, signature magnitude) per kind:", rho)
        print("onset_estimate error (batches): median %.1f, 90th pct |err| %.1f" % (sig_df.onset_error.median(), sig_df.onset_error.abs().quantile(.9)))
        rec2 = sig_df[sig_df.is_second_onset]; others = sig_df[(sig_df.kind != "recurring") | (~sig_df.is_second_onset)]
        print("recurrence: second onset of recurring mean %.2f (n=%d) vs all other events mean %.2f" % (rec2.recurrence.mean() if len(rec2) else float('nan'), len(rec2), others.recurrence.mean()))
        print("novelty by kind:", sig_df.groupby("kind").novelty.mean().round(2).to_dict())
        json.dump(dict(type_accuracy=float(acc), spearman=rho, onset_error_median=float(sig_df.onset_error.median()),
                       recurrence_second_onset=float(rec2.recurrence.mean()) if len(rec2) else None,
                       recurrence_others=float(others.recurrence.mean()), n_events=int(len(sig_df))),
                  open(METRICS_DIR / "phase6_analyzer_summary.json", "w"), indent=1)

    figure_comparison(tab, FIGURES_DIR / "phase6_detector_comparison.png", tag)
    for nm in ("abrupt_med_tes2001", "gradual_med_tes2001", "novel_class_med_tes2001", "recurring_med_tes2001"):
        it = next(((sc, sg) for sc, sg in test if sc.truth["name"] == nm), None)
        if it:
            figure_timeline(it, cfg, w, theta, LABEL_DELAY_BATCHES, FIGURES_DIR / f"phase6_timeline_{nm}.png", tag)
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
