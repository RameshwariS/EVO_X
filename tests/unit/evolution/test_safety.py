import pytest

from evox.evolution.safety import GateThresholds, SafetyGate, tune_gate


def passing():
    return dict(delta_f1_new=.01, delta_f1_historical=-.03, forgetting=.05,
                delta_fpr=.01, training_seconds=60., latency_ms_per_1k=100.)


def test_every_gate_boundary_is_inclusive():
    decision = SafetyGate().decide(passing())
    assert decision.promote and all(c["passed"] for c in decision.checks.values())


@pytest.mark.parametrize("key,value,reason", [
    ("delta_f1_new", .009, "new_gain"), ("delta_f1_historical", -.031, "historical_retention"),
    ("forgetting", .051, "forgetting"), ("delta_fpr", .011, "false_positives"),
    ("training_seconds", 60.1, "training_budget"), ("latency_ms_per_1k", 100.1, "latency_budget"),
])
def test_one_failed_threshold_rejects_the_candidate(key, value, reason):
    evaluation = passing(); evaluation[key] = value
    result = SafetyGate().decide(evaluation)
    assert not result.promote and result.reason == reason


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.1"])
def test_missing_or_nonfinite_metrics_fail_closed(bad):
    evaluation = passing(); evaluation["delta_f1_new"] = bad
    assert not SafetyGate().decide(evaluation).promote


def test_gate_selection_never_accepts_test_seeds_or_test_family():
    record = dict(family="test", seed=2001, evaluation=passing(), audit={})
    for changes in ({}, {"family": "validation"}, {"family": "test", "seed": 1001}):
        with pytest.raises(ValueError, match="validation"):
            tune_gate([{**record, **changes}])


def test_gate_grid_and_conservative_selection_on_validation():
    e = {**passing(), "delta_f1_new": .1, "delta_f1_historical": 0., "forgetting": 0., "delta_fpr": 0.,
         "training_seconds": .1}
    record = dict(family="validation", seed=1001, evaluation=e,
                  audit=dict(delta_f1_new=.2, delta_f1_historical=0.))
    chosen, sensitivity = tune_gate([record])
    assert len(sensitivity) == 48 and chosen.min_new_gain == .02
    assert SafetyGate(chosen).decide(e).promote


@pytest.mark.parametrize("key,value",[("training_seconds",-1),("latency_ms_per_1k",-1),("forgetting",-.1),
                                      ("delta_f1_new",1.1),("delta_fpr",-1.1)])
def test_physically_invalid_metrics_fail_closed(key,value):
    e=passing(); e[key]=value
    assert not SafetyGate().decide(e).promote
