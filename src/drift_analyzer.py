"""
EVO-X Phase 6 -- Drift Analyzer (Synopsis §7.3 Module 6)
========================================================
Turns "an alarm fired" into a :class:`DriftSignature` the Evolution Controller (Phase B) can reason about:

    magnitude        mean over the event window of 0.5*(PSI score + MMD score), each in units of its own alarm
                     threshold and capped at 10   (1.0 = "just at threshold")
    psi_topk, mmd_ratio   the two ingredients, for transparency
    persistence      longest run of consecutive batches with a label-free score >= 1 inside the window
    duration         batches from the estimated onset to the observation point
    ended            drift already gone at the observation point (both scores < 0.5)
    type_guess       'abrupt' | 'gradual' | 'temporary' | 'recurring'  (incremental is reported as 'gradual')
    novelty          share of low-confidence flows in the window, in excess of the reference baseline
    per_feature_psi  PSI vector of the window vs the reference (feature-level shift)
    recurrence       cosine similarity of that vector to the best-matching STORED past signature

NO LOOK-AHEAD: ``characterize(..., now_batch)`` only reads batches <= now_batch.  The harness uses
now = alarm + ``lookahead`` batches, i.e. the controller would wait that long before deciding; that
waiting time is a real cost and is reported.

Everything here is label-free (confidence, features); labels are not an input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.drift_detectors import PSIDetector


@dataclass
class DriftSignature:
    signature_id: int
    alarm_batch: int
    now_batch: int
    onset_estimate: int
    magnitude: float
    psi_topk: float
    mmd_ratio: float
    persistence: int
    duration: int
    ended: bool
    type_guess: str
    rise_slope: float
    novelty: float
    low_conf_share: float
    recurrence: float
    recurrence_match: Optional[int]
    per_feature_psi: List[float] = field(default_factory=list)

    def to_dict(self, with_vector: bool = True) -> Dict:
        d = asdict(self)
        if not with_vector:
            d.pop("per_feature_psi")
        return d


class SignatureDB:
    """Stored past signatures (their per-feature PSI vectors) for recurrence matching."""

    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.ids: List[int] = []

    def best_match(self, vec: np.ndarray) -> Tuple[float, Optional[int]]:
        if not self.vectors:
            return 0.0, None
        n = np.linalg.norm(vec)
        if n == 0:
            return 0.0, None
        sims = [float(vec @ v / (n * np.linalg.norm(v))) if np.linalg.norm(v) > 0 else 0.0 for v in self.vectors]
        i = int(np.argmax(sims))
        return sims[i], self.ids[i]

    def add(self, sig_id: int, vec: np.ndarray):
        self.vectors.append(np.asarray(vec, float)); self.ids.append(sig_id)

    def __len__(self):
        return len(self.ids)


class DriftAnalyzer:
    def __init__(self, psi: PSIDetector, psi_threshold: float, mmd_threshold: float,
                 ref_val_conf: Optional[np.ndarray] = None, baseline_low_share: float = 0.05,
                 lookback: int = 15, short_persistence: int = 15, rise_slope_thr: float = 0.02,
                 recurrence_thr: float = 0.9, cap: float = 10.0):
        """psi              a PSIDetector built on the reference (its bins and ``psi_vector`` are reused)
        psi/mmd_threshold  the OPERATING thresholds of those detectors (so scores are comparable to the alarms)
        ref_val_conf       confidences of the deployed model on held-out reference data; its
                           ``baseline_low_share`` quantile defines 'low confidence' (default 5%)."""
        self.psi, self.psi_thr, self.mmd_thr = psi, psi_threshold, mmd_threshold
        self.lookback, self.short, self.slope_thr, self.rec_thr, self.cap = lookback, short_persistence, rise_slope_thr, recurrence_thr, cap
        self.baseline = baseline_low_share
        self.conf_cut = float(np.quantile(ref_val_conf, baseline_low_share)) if ref_val_conf is not None else 0.5
        self.db = SignatureDB()
        self._next_id = 0

    # -- helpers -------------------------------------------------------------
    def _scores(self, psi_stat: np.ndarray, mmd_stat: np.ndarray):
        return (np.minimum(psi_stat / self.psi_thr, self.cap), np.minimum(np.maximum(mmd_stat, 0) / self.mmd_thr, self.cap))

    def estimate_onset(self, ps: np.ndarray, ms: np.ndarray, alarm: int) -> int:
        t = alarm
        while t - 1 >= max(0, alarm - self.lookback) and (ps[t - 1] >= 0.5 or ms[t - 1] >= 0.5):
            t -= 1
        return t

    @staticmethod
    def _longest_run(mask: np.ndarray) -> int:
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    # -- main --------------------------------------------------------------------
    def characterize(self, alarm_batch: int, now_batch: int, psi_stat: np.ndarray, mmd_stat: np.ndarray,
                     X_batches: np.ndarray, conf_batches: np.ndarray, register: bool = True) -> DriftSignature:
        """psi_stat / mmd_stat : per-batch raw statistics (only indices <= now_batch are read)
        X_batches            : (n_batches, bs, d) or list of per-batch arrays;  conf_batches: (n_batches, bs)"""
        now = now_batch
        ps, ms = self._scores(np.asarray(psi_stat)[: now + 1], np.asarray(mmd_stat)[: now + 1])
        onset = self.estimate_onset(ps, ms, alarm_batch)
        win = slice(onset, now + 1)
        comb = 0.5 * (ps + ms)
        magnitude = float(comb[win].mean())
        above = (ps[win] >= 1.0) | (ms[win] >= 1.0)
        persistence = self._longest_run(above)
        ended = bool((ps[now] < 0.5) and (ms[now] < 0.5) and above.any())

        # feature-level shift: batches in the alarm neighbourhood (drift certainly in effect there)
        w = self.psi.window.maxlen or 4
        lo, hi = max(onset, alarm_batch - 1), min(now, alarm_batch + w)
        Xw = np.vstack([X_batches[i] for i in range(lo, hi + 1)])
        vec = self.psi.psi_vector(Xw)

        # novelty from confidence (label-free)
        cw = np.concatenate([np.asarray(conf_batches[i]).ravel() for i in range(onset, now + 1)])
        low = float((cw < self.conf_cut).mean())
        novelty = max(0.0, low - self.baseline)

        rec, match = self.db.best_match(vec)

        # type guess: relative growth of the PSI score after the sliding window has filled
        st = onset + w
        seg = ps[st: now + 1]
        slope = 0.0
        if len(seg) >= 3 and seg.mean() > 0:
            slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0] / seg.mean())
        if ended and persistence <= self.short:
            tg = "temporary"
        elif rec >= self.rec_thr:
            tg = "recurring"
        elif slope > self.slope_thr:
            tg = "gradual"
        else:
            tg = "abrupt"

        sig = DriftSignature(self._next_id, alarm_batch, now, onset, magnitude, float(np.sort(psi_stat[:now + 1][win])[-1]) if len(ps[win]) else 0.0,
                             float(np.max(np.maximum(mmd_stat[:now + 1][win], 0) / max(self.mmd_thr, 1e-12))),
                             persistence, now - onset + 1, ended, tg, slope, novelty, low, rec, match, vec.tolist())
        if register:
            self.db.add(sig.signature_id, vec)
        self._next_id += 1
        return sig
