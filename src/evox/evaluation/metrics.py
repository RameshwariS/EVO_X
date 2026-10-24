"""
EVO-X Phase 3 -- streaming evaluation metrics (Synopsis §8.4)
=============================================================
Three views, because a static model cannot name classes it never saw:

 1. binary      attack-vs-benign:   precision / recall / F1 / FPR / FNR
 2. known-class multi-class macro-F1 on flows whose true class is in the
                reference window (isolates covariate/concept shift from novelty)
 3. novel       share of novel-class flows that are at least flagged as attack

Accuracy is deliberately not computed (§8.4).  All ratios use zero_division=0
semantics: an undefined ratio is reported as 0.0 AND its support is reported,
so empty batches are never mistaken for perfect ones.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import f1_score

BENIGN_ID = 0


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    t, p = y_true != BENIGN_ID, y_pred != BENIGN_ID
    return dict(tp=int((t & p).sum()), fp=int((~t & p).sum()),
                fn=int((t & ~p).sum()), tn=int((~t & ~p).sum()))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    c = binary_counts(y_true, y_pred)
    prec, rec = _div(c["tp"], c["tp"] + c["fp"]), _div(c["tp"], c["tp"] + c["fn"])
    return dict(precision=prec, recall=rec, f1=_div(2 * prec * rec, prec + rec),
                fpr=_div(c["fp"], c["fp"] + c["tn"]), fnr=_div(c["fn"], c["fn"] + c["tp"]),
                n_attack=c["tp"] + c["fn"], n_benign=c["fp"] + c["tn"])


def known_class_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, known: Iterable[int]) -> float:
    """Macro-F1 over flows whose TRUE class is known, averaged over the known
    classes that actually occur in that slice (absent classes are not scored)."""
    known = np.asarray(list(known))
    m = np.isin(y_true, known)
    if not m.any():
        return float("nan")
    labels = np.unique(y_true[m])
    return float(f1_score(y_true[m], y_pred[m], labels=labels, average="macro", zero_division=0))


def novel_detection(y_true: np.ndarray, y_pred: np.ndarray, known: Iterable[int]) -> Dict[str, float]:
    nov = ~np.isin(y_true, np.asarray(list(known)))
    n = int(nov.sum())
    flagged = int((y_pred[nov] != BENIGN_ID).sum())
    return dict(n_novel=n, n_novel_flagged=flagged, novel_flag_rate=_div(flagged, n) if n else float("nan"))
