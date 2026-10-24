"""Full-candidate evaluation on released, independent gate data (Phase C)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from evox.adaptation.data import LabelPolicy, Samples
from evox.adaptation.strategies import CandidateModel
from evox.evaluation.metrics import binary_metrics
from evox.models.ids import IDSModel


def classification_metrics(y, pred, labels=None):
    y, pred = np.asarray(y), np.asarray(pred)
    if not len(y):
        raise ValueError("Evaluation needs nonempty labelled data")
    labels = np.unique(y) if labels is None else np.asarray(labels)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = {}
    for i, c in enumerate(labels):
        truth, positive = y == c, pred == c
        fp, negatives = int((~truth & positive).sum()), int((~truth).sum())
        fn, positives = int((truth & ~positive).sum()), int(truth.sum())
        per_class[str(int(c))] = dict(precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]),
                                     support=int(support[i]), fpr=fp / negatives if negatives else None,
                                     fnr=fn / positives if positives else None)
    return dict(macro_f1=float(f1.mean()), binary=binary_metrics(y, pred), per_class=per_class, n=len(y))


def model_metrics(model, samples, labels=None):
    return classification_metrics(samples.y, model.predict(samples.X), labels)


@dataclass(frozen=True)
class EvaluationSets:
    recent: Samples
    historical: Samples

    def validate(self, candidate: CandidateModel, policy: LabelPolicy, now_batch: int, forbidden_ids=()):
        if not len(self.recent) or not len(self.historical):
            raise ValueError("Gate evaluation needs nonempty recent and historical data")
        recent, historical = set(self.recent.ids), set(self.historical.ids)
        if recent & historical:
            raise ValueError("Recent and historical holdouts overlap")
        if (recent | historical) & (set(candidate.training_ids) | set(forbidden_ids)):
            raise ValueError("Evaluation overlaps candidate/ancestor/proxy training or selection data")
        for s in (self.recent, self.historical):
            if len(policy.available(s, now_batch)) != len(s):
                raise ValueError("Evaluation attempted to use unreleased/unbudgeted labels")
        if not np.any(np.concatenate((self.recent.y, self.historical.y)) == 0):
            raise ValueError("Gate cannot assess false positives without benign support")


class CandidateEvaluator:
    def evaluate(self, candidate: CandidateModel, parent: IDSModel, sets: EvaluationSets,
                 policy: LabelPolicy, now_batch: int, best_historical=None, initial_historical=None,
                 forbidden_ids=()):
        sets.validate(candidate, policy, now_batch, forbidden_ids)
        labels = np.unique(sets.historical.y)
        recent_labels = np.unique(sets.recent.y)
        parent_new = model_metrics(parent, sets.recent, recent_labels)
        parent_old = model_metrics(parent, sets.historical, labels)
        new = model_metrics(candidate.model, sets.recent, recent_labels)
        old = model_metrics(candidate.model, sets.historical, labels)
        y = np.concatenate((sets.recent.y, sets.historical.y))
        X = np.concatenate((sets.recent.X, sets.historical.X))
        parent_fpr = binary_metrics(y, parent.predict(X))["fpr"]
        fpr = binary_metrics(y, candidate.model.predict(X))["fpr"]
        best = best_historical or {c: m["f1"] for c, m in parent_old["per_class"].items()}
        initial = initial_historical or {c: m["f1"] for c, m in parent_old["per_class"].items()}
        forgetting = [max(0.0, best.get(c, m["f1"]) - m["f1"]) for c, m in old["per_class"].items()]
        backward = [m["f1"] - initial.get(c, m["f1"]) for c, m in old["per_class"].items()]
        probe = sets.recent.X[:256]
        candidate.model.predict(probe)
        timings = []
        for _ in range(3):
            start = time.perf_counter(); candidate.model.predict(probe)
            timings.append((time.perf_counter() - start) * 1e6 / len(probe))
        result = dict(candidate_id=candidate.candidate_id, parent_version=candidate.parent_version,
                      evaluated_at_batch=now_batch, recent=new, historical=old,
                      parent_recent=parent_new, parent_historical=parent_old,
                      delta_f1_new=new["macro_f1"] - parent_new["macro_f1"],
                      delta_f1_historical=old["macro_f1"] - parent_old["macro_f1"],
                      forgetting=float(np.mean(forgetting)), backward_transfer=float(np.mean(backward)),
                      delta_fpr=fpr - parent_fpr, training_seconds=candidate.profile["training_seconds"],
                      latency_ms_per_1k=float(np.median(timings)),
                      recent_snapshot=sets.recent.snapshot(), historical_snapshot=sets.historical.snapshot(),
                      evaluation_ids=sets.recent.ids.tolist() + sets.historical.ids.tolist())
        candidate.metrics = result
        return result


def counterfactual_replay(candidate, parent, future: Samples, policy: LabelPolicy, now_batch: int,
                          start_batch: int, end_batch: int, historical_delta: float,
                          harmful_tolerance: float = 0.03):
    """Frozen candidate vs frozen parent on future reporting rows, after labels arrive.

    This is a finite-horizon shadow comparison, not a re-simulation of later
    adaptation decisions. An empty horizon is reported as censored, never safe.
    """
    eligible = policy.available(future, now_batch)
    eligible = eligible.take((eligible.batches >= start_batch) & (eligible.batches <= end_batch))
    if set(eligible.ids) & set(candidate.training_ids):
        raise ValueError("Counterfactual evaluation overlaps training data")
    if not len(eligible):
        return dict(candidate_id=candidate.candidate_id, n=0, censored=True, harmful=None, successful=None)
    new, old = model_metrics(candidate.model, eligible), model_metrics(parent, eligible)
    gain = new["macro_f1"] - old["macro_f1"]
    fpr = new["binary"]["fpr"] - old["binary"]["fpr"]
    return dict(candidate_id=candidate.candidate_id, n=len(eligible), censored=False,
                start_batch=start_batch, end_batch=end_batch, delta_f1_new=gain,
                delta_f1_historical=historical_delta, delta_fpr=fpr,
                harmful=bool(gain < -harmful_tolerance or historical_delta < -harmful_tolerance),
                successful=bool(gain >= 0.01 and historical_delta >= -harmful_tolerance),
                unnecessary=bool(gain < 0.01))
