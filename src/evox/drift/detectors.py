"""
EVO-X Phase 6 -- drift detectors (Synopsis §7.3 Module 5, §7.4 fusion)
=====================================================================
All detectors consume a :class:`BatchObs` once per replay batch and expose

    update(obs) -> bool     alarm raised at THIS step (time of alarm = time the detector learned of the change)
    score       float       normalised drift score, 1.0 == the detector's own alarm threshold
    needs_labels bool

Label policy.  Error-based detectors (ADWIN-err, Page-Hinkley) can only see labels that have arrived:
the runner fills ``obs.err`` with the 0/1 error flags of batch ``t - LABEL_DELAY_BATCHES``.  Their alarms
therefore include the label delay.  PSI, MMD and confidence-ADWIN never touch labels.

Detector input granularity (found empirically, see tests):
  * ADWIN is fed one value PER FLOW (the Bernoulli error stream it was designed for).  Fed batch-mean error
    rates it did not fire even 200 batches after a 0.02 -> 0.15 step, because its Hoeffding-type bound is
    too conservative for ~100 points.
  * Page-Hinkley is fed the per-batch error rate (it is a lightweight sequential test).
  * PSI / MMD compare a sliding window of the last W batches against the reference window.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class BatchObs:
    index: int
    X: np.ndarray                         # features of this batch (always available)
    conf: np.ndarray                      # model confidence per flow (always available)
    err: Optional[np.ndarray] = None      # 0/1 error flags of a batch whose labels have JUST arrived
    err_batch: Optional[int] = None       # which batch those labels belong to


class EdgeTrigger:
    """Alarm on the rising edge of ``score >= 1``; re-arm once the score falls below ``rearm``."""

    def __init__(self, rearm: float = 0.5):
        self.rearm, self.active = rearm, False

    def update(self, score: float) -> bool:
        if not self.active and score >= 1.0:
            self.active = True
            return True
        if self.active and score < self.rearm:
            self.active = False
        return False

    def reset(self):
        self.active = False


class _Held:
    """Turn an instantaneous alarm into a score that decays linearly over ``hold`` batches (for fusion)."""

    def __init__(self, hold: int = 5):
        self.hold, self.age = hold, 10 ** 9

    def step(self, alarm: bool) -> float:
        self.age = 0 if alarm else self.age + 1
        return max(0.0, 1.0 - self.age / self.hold)


# ---------------------------------------------------------------------------
# ADWIN (River) -- error stream (needs labels) or confidence stream (label-free)
# ---------------------------------------------------------------------------
class ADWINDetector:
    def __init__(self, signal: str = "error", delta: float = 0.002, hold: int = 5):
        from river.drift import ADWIN
        if signal not in ("error", "conf"):
            raise ValueError("signal must be 'error' or 'conf'")
        self.signal, self.delta, self._cls = signal, delta, ADWIN
        self.name = "adwin_err" if signal == "error" else "adwin_conf"
        self.needs_labels = signal == "error"
        self._held = _Held(hold)
        self.reset()

    def reset(self):
        self.adwin = self._cls(delta=self.delta, clock=1, min_window_length=5, grace_period=10)
        self.score = 0.0

    def update(self, obs: BatchObs) -> bool:
        if self.signal == "error":
            vals = obs.err
        else:
            vals = 1.0 - obs.conf                   # low confidence ~ unfamiliar traffic; needs no labels
        alarm = False
        if vals is not None:
            for v in np.asarray(vals, dtype=float):
                self.adwin.update(float(v))
                if self.adwin.drift_detected:
                    alarm = True
        self.score = self._held.step(alarm)
        return alarm


# ---------------------------------------------------------------------------
# Page-Hinkley on the per-batch error rate (increase), implemented here so the statistic is observable
# ---------------------------------------------------------------------------
class PageHinkleyDetector:
    name, needs_labels = "ph_err", True

    def __init__(self, delta: float = 0.005, threshold: float = 0.5, min_instances: int = 10):
        self.delta, self.threshold, self.min_instances = delta, threshold, min_instances
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.M, self.stat, self.score = 0, 0.0, 0.0, 0.0, 0.0, 0.0

    def update_value(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m += x - self.mean - self.delta
        self.M = min(self.M, self.m)
        self.stat = self.m - self.M
        self.score = self.stat / self.threshold
        if self.n >= self.min_instances and self.stat > self.threshold:
            self.reset()                          # restart after an alarm
            return True
        return False

    def update(self, obs: BatchObs) -> bool:
        if obs.err is None or len(obs.err) == 0:
            return False
        return self.update_value(float(np.mean(obs.err)))


# ---------------------------------------------------------------------------
# PSI (label-free): per-feature population stability index vs the reference window
# ---------------------------------------------------------------------------
class PSIDetector:
    """Bins come from REFERENCE quantiles (n_bins), outermost bins open-ended.
    Conventional cut-offs: 0.1 moderate, 0.25 major (``threshold`` default).
    Aggregate over features = mean of the ``topk`` largest per-feature PSIs: a plain mean over ~70 features
    dilutes a shift that touches a handful of them; max alone is noisy."""
    name, needs_labels = "psi", False

    def __init__(self, reference: np.ndarray, window_batches: int = 4, n_bins: int = 10, threshold: float = 0.25,
                 topk: int = 5, max_ref: int = 50_000, seed: int = 0):
        rng = np.random.default_rng(seed)
        ref = reference if len(reference) <= max_ref else reference[rng.choice(len(reference), max_ref, replace=False)]
        self.d, self.threshold, self.topk, self.n_bins = ref.shape[1], threshold, min(topk, ref.shape[1]), n_bins
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        self.edges: List[np.ndarray] = []
        self.p_ref: List[np.ndarray] = []
        for j in range(self.d):
            e = np.unique(np.quantile(ref[:, j], qs))
            self.edges.append(e)
            self.p_ref.append(self._props(ref[:, j], e))
        self.window: Deque[np.ndarray] = deque(maxlen=window_batches)
        self.trigger = EdgeTrigger()
        self.score, self.stat, self.vector = 0.0, 0.0, np.zeros(self.d)

    @staticmethod
    def _props(col: np.ndarray, edges: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(edges, col, side="right")
        c = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        return np.clip(c / max(c.sum(), 1.0), 1e-4, None)

    def psi_vector(self, X: np.ndarray) -> np.ndarray:
        v = np.empty(self.d)
        for j in range(self.d):
            pa = self._props(X[:, j], self.edges[j]); pe = self.p_ref[j]
            pa, pe = pa / pa.sum(), pe / pe.sum()
            v[j] = float(np.sum((pa - pe) * np.log(pa / pe)))
        return v

    def aggregate(self, vec: np.ndarray) -> float:
        return float(np.sort(vec)[-self.topk:].mean())

    def observe(self, X: np.ndarray) -> float:
        """Push a batch, return the (threshold-free) aggregate PSI of the current window."""
        self.window.append(X)
        self.vector = self.psi_vector(np.vstack(self.window))
        self.stat = self.aggregate(self.vector)
        return self.stat

    def update(self, obs: BatchObs) -> bool:
        self.observe(obs.X)
        self.score = self.stat / self.threshold
        return self.trigger.update(self.score)

    def reset(self):
        self.window.clear(); self.trigger.reset(); self.score = self.stat = 0.0


# ---------------------------------------------------------------------------
# MMD (label-free): RBF kernel, median-heuristic bandwidth, subsampled, null-calibrated threshold
# ---------------------------------------------------------------------------
class MMDDetector:
    """Unbiased MMD^2 between a subsample of the sliding window and a FIXED reference subsample.
    The alarm threshold is calibrated from a null distribution: pseudo-windows drawn from a *disjoint* half of
    the reference are compared to the fixed reference subsample; score = MMD^2 / (``mult`` * null 99th percentile)."""
    name, needs_labels = "mmd", False

    def __init__(self, reference: np.ndarray, window_batches: int = 4, n_sub: int = 400, n_ref: int = 400,
                 mult: float = 1.5, n_null: int = 60, seed: int = 0):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(reference))
        half = len(perm) // 2
        self.ref = reference[perm[:half][:n_ref]].astype(np.float64)
        pool_b = reference[perm[half:]].astype(np.float64)
        self.n_sub, self.mult, self.rng = n_sub, mult, np.random.default_rng([seed, 99])
        sub = self.ref[: min(len(self.ref), 500)]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[np.triu_indices(len(sub), 1)])
        self.gamma = 1.0 / (2.0 * max(med, 1e-12))                 # median heuristic: bandwidth^2 = median dist^2
        self.k_rr = self._k(self.ref, self.ref)
        null = [self._mmd2(pool_b[self.rng.choice(len(pool_b), min(n_sub, len(pool_b)), replace=False)])
                for _ in range(n_null)]
        self.null = np.array(null)
        self.null_q99 = float(max(np.quantile(null, 0.99), 1e-6))
        self.window: Deque[np.ndarray] = deque(maxlen=window_batches)
        self.trigger = EdgeTrigger()
        self.score, self.stat = 0.0, 0.0

    def _k(self, A, B):
        a2, b2 = (A ** 2).sum(1)[:, None], (B ** 2).sum(1)[None, :]
        return np.exp(-self.gamma * np.maximum(a2 + b2 - 2 * A @ B.T, 0.0))

    def _mmd2(self, W: np.ndarray) -> float:
        W = W.astype(np.float64)
        n, m = len(W), len(self.ref)
        kxx, kxy = self._k(W, W), self._k(W, self.ref)
        return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (self.k_rr.sum() - np.trace(self.k_rr)) / (m * (m - 1))
                     - 2.0 * kxy.mean())

    def observe(self, X: np.ndarray) -> float:
        self.window.append(X)
        W = np.vstack(self.window)
        if len(W) > self.n_sub:
            W = W[self.rng.choice(len(W), self.n_sub, replace=False)]
        self.stat = self._mmd2(W)
        return self.stat

    def update(self, obs: BatchObs) -> bool:
        self.observe(obs.X)
        self.score = max(self.stat, 0.0) / (self.mult * self.null_q99)
        return self.trigger.update(self.score)

    def reset(self):
        self.window.clear(); self.trigger.reset(); self.score = self.stat = 0.0


# ---------------------------------------------------------------------------
# Fusion  D_fused = sum_i w_i * min(score_i, 1),  sum w_i = 1   (§7.4)
# ---------------------------------------------------------------------------
def normalise_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(w.values()))
    if s <= 0:
        raise ValueError("weights must have positive sum")
    return {k: v / s for k, v in w.items()}


def fuse(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Each member score is clipped at 1 (= at its own threshold) so no single detector can saturate the fusion."""
    return float(sum(w * min(max(scores.get(k, 0.0), 0.0), 1.0) for k, w in weights.items()))


class FusedDetector:
    """Declares drift on the rising edge of ``D_fused >= theta``.  Whether this beats the best single detector is
    MEASURED (phase6_detect_eval.py), not assumed."""
    name, needs_labels = "fused", True

    def __init__(self, members: Sequence, weights: Dict[str, float], theta: float = 0.5):
        self.members = {m.name: m for m in members}
        self.weights = normalise_weights({k: v for k, v in weights.items() if k in self.members})
        self.theta = theta
        self.trigger = EdgeTrigger()
        self.score = self.D = 0.0
        self.member_alarms: Dict[str, bool] = {}

    def update(self, obs: BatchObs) -> bool:
        self.member_alarms = {k: m.update(obs) for k, m in self.members.items()}
        self.D = fuse({k: m.score for k, m in self.members.items()}, self.weights)
        self.score = self.D / self.theta
        return self.trigger.update(self.score)

    def reset(self):
        for m in self.members.values():
            m.reset()
        self.trigger.reset(); self.score = self.D = 0.0
