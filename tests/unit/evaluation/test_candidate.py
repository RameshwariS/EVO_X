from dataclasses import replace

import numpy as np
import pytest

from evox.adaptation.data import LabelPolicy, Samples
from evox.adaptation.strategies import CandidateModel
from evox.evaluation.candidate import CandidateEvaluator, EvaluationSets, classification_metrics, counterfactual_replay


class RuleModel:
    def __init__(self, correct=True):
        self.correct = correct

    def predict(self, X):
        values = X[:, 0].astype(int)
        return values if self.correct else 1 - values


def samples(prefix, batch=-1):
    y = np.array([0, 0, 1, 1])
    return Samples(y[:, None], y, np.array([f"{prefix}:{i}" for i in range(4)]), np.full(4, batch))


def candidate():
    return CandidateModel("c1", "v1", "replay", RuleModel(), "snapshot", ["training"],
                          dict(training_seconds=.1))


def test_metrics_hand_computed_per_class_and_binary():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["binary"]["fpr"] == .5
    assert m["per_class"]["1"]["f1"] == pytest.approx(.8)
    assert m["macro_f1"] == pytest.approx((2/3 + .8) / 2)


def test_candidate_gain_and_historical_best_forgetting():
    c = candidate()
    result = CandidateEvaluator().evaluate(c, RuleModel(False), EvaluationSets(samples("new", 3), samples("hist")),
                                           LabelPolicy(2), 5, {"0": 1., "1": 1.})
    assert result["delta_f1_new"] == 1 and result["delta_f1_historical"] == 1
    assert result["forgetting"] == 0 and result["delta_fpr"] == -1
    assert result["latency_ms_per_1k"] > 0 and c.metrics == result


@pytest.mark.parametrize("case", ["candidate", "ancestor", "overlap", "future", "empty"])
def test_evaluator_enforces_data_isolation_and_label_release(case):
    c, recent, old, forbidden = candidate(), samples("new", 3), samples("old"), ()
    if case == "candidate": c.training_ids = ["new:0"]
    if case == "ancestor": forbidden = ("new:0",)
    if case == "overlap": old = recent
    if case == "future": recent = samples("new", 6)
    if case == "empty": recent = Samples.empty(1)
    with pytest.raises(ValueError):
        CandidateEvaluator().evaluate(c, RuleModel(), EvaluationSets(recent, old), LabelPolicy(2), 5,
                                       forbidden_ids=forbidden)


def test_shadow_counterfactual_uses_only_released_future_rows_and_censors_empty():
    c = candidate()
    c.model = RuleModel(False)
    s = samples("future", 7)
    empty = counterfactual_replay(c, RuleModel(), s, LabelPolicy(2), 8, 7, 10, 0.)
    assert empty["censored"] and empty["harmful"] is None
    result = counterfactual_replay(c, RuleModel(), s, LabelPolicy(2), 9, 7, 10, 0.)
    assert result["harmful"] and result["unnecessary"] and not result["successful"]
