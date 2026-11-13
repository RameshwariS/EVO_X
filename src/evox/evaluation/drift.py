"""
EVO-X Phase 6 -- detector evaluation harness (Synopsis §8.4; plan A3 "Evaluation")
==================================================================================
Pipeline per scenario:
    static model (trained once on the scenario's reference)  ->  per-flow predictions / confidence
    -> threshold-free *signals* (error flags, confidence, PSI(t), MMD^2(t))      [compute_signals]
    -> alarms for ANY detector hyper-parameters, cheaply                         [run_members]
    -> matched against ground truth                                              [score_alarms]
Signals are computed once so thresholds can be swept/tuned on VALIDATION scenarios and then frozen for TEST.

Scoring rules (explicit tolerance window ``tol`` batches):
  * Every truth event with ``scored=True`` owns the window [start_batch, end_batch + tol].
    The FIRST alarm in that window is the detection; delay = alarm_batch - start_batch.
    A scored event with no alarm in its window is a MISS.
  * Further alarms inside an already-matched window are 'repeat' alarms: not false, reported separately.
  * Alarms inside the window of an unscored event (the return-to-A of recurring/temporary drift) are 'neutral':
    reference-anchored detectors legitimately fire when data moves back, so they are neither rewarded nor punished.
  * Every other alarm is a FALSE ALARM.  Rate = false alarms per 1000 batches.
  * Error-based detectors learn about a change only when labels arrive, so their delay includes LABEL_DELAY_BATCHES.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evox.config import LABEL_DELAY_BATCHES
from evox.drift.detectors import (ADWINDetector, BatchObs, EdgeTrigger, MMDDetector, PageHinkleyDetector,
                                 PSIDetector, _Held, fuse, normalise_weights)
from evox.drift.scenarios import Scenario
from evox.models.ids import IDSModel

MEMBER_NAMES = ("adwin_err", "ph_err", "psi", "mmd")           # the four synopsis detectors (fusion members)
ALL_NAMES = MEMBER_NAMES + ("adwin_conf",)                       # + label-free confidence ADWIN (extra)


# ---------------------------------------------------------------------------
# static model + replay
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[tuple, IDSModel] = {}


def train_static(sc: Scenario, kind: str = "random_forest", seed: int = 0, cap_per_class: int = 5000,
                 n_estimators: int = 60) -> IDSModel:
    """One static model per distinct reference (cached): the deployed IDS that never adapts."""
    key = (kind, seed, tuple(sc.truth["known_class_ids"]), id(sc.stream.X_ref_train) if False else sc.stream.X_ref_train.shape,
           float(sc.stream.X_ref_train[:50].sum()))
    if key not in _MODEL_CACHE:
        X, y = sc.stream.X_ref_train, sc.stream.y_ref_train
        rng = np.random.default_rng(seed)
        keep = np.concatenate([np.flatnonzero(y == c) if (y == c).sum() <= cap_per_class else
                               rng.choice(np.flatnonzero(y == c), cap_per_class, replace=False) for c in np.unique(y)])
        _MODEL_CACHE[key] = IDSModel(kind, seed=seed, params=dict(n_estimators=n_estimators)).fit(X[keep], y[keep])
    return _MODEL_CACHE[key]


@dataclass
class StaticRun:
    y_pred: np.ndarray
    conf: np.ndarray
    err_flags: np.ndarray          # (n_batches, batch_size) 0/1
    conf_batches: np.ndarray       # (n_batches, batch_size)

    @property
    def batch_err(self) -> np.ndarray:
        return self.err_flags.mean(axis=1)


def static_run(sc: Scenario, model: IDSModel) -> StaticRun:
    s, bs, nb = sc.stream, sc.spec.batch_size, sc.spec.n_batches
    # one call over the whole stream (same per-row result as batch-wise scoring -- asserted in tests -- but ~10x faster)
    yp, cf = model.predict_with_confidence(s.X_stream)
    yp, cf = yp.astype(np.int16), cf.astype(np.float32)
    err = (yp != s.y_stream).astype(np.float32)
    return StaticRun(yp, cf, err.reshape(nb, bs), cf.reshape(nb, bs))


# ---------------------------------------------------------------------------
# threshold-free signals
# ---------------------------------------------------------------------------
@dataclass
class Signals:
    n_batches: int
    batch_size: int
    run: StaticRun
    psi_stat: np.ndarray           # (n,) aggregate PSI of the sliding window
    psi_vec: np.ndarray            # (n, d) per-feature PSI of the sliding window
    mmd_stat: np.ndarray           # (n,) MMD^2 of the sliding window
    mmd_null_q99: float
    window_batches: int
    truth: Dict
    ref_psi_detector: Optional[PSIDetector] = None   # kept so the analyzer can reuse the reference bins


def compute_signals(sc: Scenario, run: StaticRun, window_batches: int = 4, seed: int = 0,
                    mmd_n_sub: int = 400, mmd_null: int = 60) -> Signals:
    ref = sc.stream.X_ref_train
    psi = PSIDetector(ref, window_batches=window_batches, seed=seed)
    mmd = MMDDetector(ref, window_batches=window_batches, n_sub=mmd_n_sub, n_ref=mmd_n_sub, n_null=mmd_null, seed=seed)
    nb, bs = sc.spec.n_batches, sc.spec.batch_size
    ps, pv, ms = np.empty(nb), np.empty((nb, ref.shape[1])), np.empty(nb)
    for b in sc.stream.iter_batches(bs):
        ps[b.index] = psi.observe(b.X); pv[b.index] = psi.vector; ms[b.index] = mmd.observe(b.X)
    return Signals(nb, bs, run, ps, pv, ms, mmd.null_q99, window_batches, sc.truth, psi)


@dataclass
class MemberConfig:
    adwin_delta: float = 0.002
    ph_delta: float = 0.005
    ph_threshold: float = 0.5
    psi_threshold: float = 0.25
    mmd_mult: float = 1.5
    adwin_conf_delta: float = 0.002
    hold: int = 5


@dataclass
class MemberOut:
    alarms: List[int]
    score: np.ndarray              # normalised per-batch score (1.0 == own alarm threshold)


def _edge_alarms(raw: np.ndarray, denom: float) -> MemberOut:
    tr, al, sc = EdgeTrigger(), [], np.empty(len(raw))
    for t, v in enumerate(raw):
        sc[t] = max(v, 0.0) / denom
        if tr.update(sc[t]):
            al.append(t)
    return MemberOut(al, sc)


def run_member(name: str, sig: Signals, cfg: MemberConfig, label_delay: int = LABEL_DELAY_BATCHES) -> MemberOut:
    n = sig.n_batches
    if name == "psi":
        return _edge_alarms(sig.psi_stat, cfg.psi_threshold)
    if name == "mmd":
        return _edge_alarms(sig.mmd_stat, cfg.mmd_mult * sig.mmd_null_q99)
    if name in ("adwin_err", "ph_err", "adwin_conf"):
        det = {"adwin_err": lambda: ADWINDetector("error", cfg.adwin_delta, cfg.hold),
               "ph_err": lambda: PageHinkleyDetector(cfg.ph_delta, cfg.ph_threshold),
               "adwin_conf": lambda: ADWINDetector("conf", cfg.adwin_conf_delta, cfg.hold)}[name]()
        held, al, sc = _Held(cfg.hold), [], np.zeros(n)
        for t in range(n):
            k = t - label_delay
            obs = BatchObs(t, np.empty((0, 0), np.float32), sig.run.conf_batches[t],
                           sig.run.err_flags[k] if k >= 0 else None, k if k >= 0 else None)
            a = det.update(obs)
            sc[t] = det.score if name != "ph_err" else min(det.score, 10.0)
            if a:
                al.append(t)
                if name == "ph_err":
                    sc[t] = 1.0                       # PH resets its statistic on alarm; keep the alarm visible
        if name == "ph_err":                           # same hold-and-decay as ADWIN so the fusion sees alarms
            h = _Held(cfg.hold); sc = np.array([max(sc[t], h.step(t in al)) for t in range(n)])
        return MemberOut(al, sc)
    raise KeyError(name)


def run_members(sig: Signals, cfg: MemberConfig, names: Sequence[str] = ALL_NAMES,
                label_delay: int = LABEL_DELAY_BATCHES) -> Dict[str, MemberOut]:
    return {nm: run_member(nm, sig, cfg, label_delay) for nm in names}


def fused_alarms(members: Dict[str, MemberOut], weights: Dict[str, float], theta: float) -> MemberOut:
    w = normalise_weights({k: v for k, v in weights.items() if k in members and v > 0})
    n = len(next(iter(members.values())).score)
    D = np.zeros(n)
    for k, wk in w.items():
        D += wk * np.minimum(np.maximum(members[k].score, 0.0), 1.0)
    return _edge_alarms(D, theta)


# ---------------------------------------------------------------------------
# scoring against truth
# ---------------------------------------------------------------------------
def score_alarms(alarms: Sequence[int], truth: Dict, tol: int = 10) -> Dict:
    ev = truth["events"]
    scored = [e for e in ev if e["scored"]]
    neutral = [e for e in ev if not e["scored"]]
    win = lambda e: (e["start_batch"], e["end_batch"] + tol)
    matched: Dict[int, int] = {}
    n_false = n_repeat = n_neutral = 0
    for a in sorted(alarms):
        hit = [e for e in scored if win(e)[0] <= a <= win(e)[1]]
        if hit:
            e = hit[0]
            if e["id"] in matched: n_repeat += 1
            else: matched[e["id"]] = a
        elif any(win(e)[0] <= a <= win(e)[1] for e in neutral):
            n_neutral += 1
        else:
            n_false += 1
    rows = [dict(event_id=e["id"], start_batch=e["start_batch"], detected=e["id"] in matched,
                 alarm_batch=matched.get(e["id"]), delay=(matched[e["id"]] - e["start_batch"]) if e["id"] in matched else None,
                 recurrence=bool(e.get("recurrence", False))) for e in scored]
    return dict(events=rows, n_events=len(scored), n_detected=len(matched), n_missed=len(scored) - len(matched),
                n_false=n_false, n_repeat=n_repeat, n_neutral=n_neutral, n_batches=truth["n_batches"],
                n_alarms=len(alarms))


def aggregate(results: Sequence[Tuple[Dict, Dict]]) -> Dict:
    """results = [(truth, score_alarms output)] -> detection rate, mean/median delay, misses, false alarms per 1000 batches."""
    ne = sum(r["n_events"] for _, r in results); nd = sum(r["n_detected"] for _, r in results)
    delays = [e["delay"] for _, r in results for e in r["events"] if e["delay"] is not None]
    nf = sum(r["n_false"] for _, r in results); nb = sum(r["n_batches"] for _, r in results)
    return dict(n_events=ne, n_detected=nd, n_missed=ne - nd, detection_rate=(nd / ne) if ne else float("nan"),
                mean_delay=float(np.mean(delays)) if delays else float("nan"),
                median_delay=float(np.median(delays)) if delays else float("nan"),
                n_false=nf, n_batches=nb, fa_per_1000=1000.0 * nf / nb if nb else float("nan"),
                n_repeat=sum(r["n_repeat"] for _, r in results))


def criterion(agg: Dict, lam_fa: float = 0.02, lam_delay: float = 0.002) -> float:
    """PRE-REGISTERED selection criterion (validation scenarios only):
         J = detection_rate - 0.02 * false_alarms_per_1000_batches - 0.002 * mean_delay_batches
    (a miss costs 1 event; an extra false alarm per 1000 batches costs 0.02; a batch of delay 0.002).
    Undefined delay (nothing detected) counts as 0 -- the missed events already cost the full detection_rate."""
    d = agg["mean_delay"]
    return agg["detection_rate"] - lam_fa * agg["fa_per_1000"] - lam_delay * (0.0 if np.isnan(d) else d)
