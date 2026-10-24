"""Phase B rule routing, hand-computed utilities, validation isolation and events."""
import json
from dataclasses import replace

import numpy as np
import pytest

from evox.adaptation.strategies import Action
from evox.config import TOPIC_EVOLUTION
from evox.drift.analyzer import DriftSignature
from evox.evolution.controller import (ActionEstimate, ControllerThresholds, EvolutionController,
                                      UtilityWeights, publish_decision, tune_weights, weight_grid)
from evox.streaming.transport import InMemoryTransport


def signature(**kw):
    s = DriftSignature(1, 40, 48, 40, 2.0, 1.0, 1.0, 5, 9, False, "abrupt",
                       0.0, 0.0, 0.05, 0.0, None, [1.0, 0.0])
    return replace(s, **kw)


@pytest.mark.parametrize("changes,rule,actions", [
    ({"magnitude": 0.99}, "insignificant_short_or_ended", {Action.NO_OP}),
    ({"persistence": 1}, "insignificant_short_or_ended", {Action.NO_OP}),
    ({"ended": True}, "insignificant_short_or_ended", {Action.NO_OP}),
    ({"recurrence": 0.9}, "recurring", {Action.REPLAY, Action.INCREMENTAL, Action.NO_OP}),
    ({"magnitude": 3.0, "novelty": 0.2}, "high_unfamiliarity_and_magnitude",
     {Action.RETRAIN, Action.STRATEGIC, Action.NO_OP}),
    ({"magnitude": 3.0, "persistence": 10}, "major_persistent", {Action.RETRAIN, Action.REPLAY, Action.NO_OP}),
    ({}, "moderate", {Action.INCREMENTAL, Action.REPLAY, Action.STRATEGIC, Action.NO_OP}),
])
def test_every_rule_branch_and_inclusive_boundaries(changes, rule, actions):
    choices, reason = EvolutionController().rule_actions(signature(**changes))
    assert reason == rule and set(choices) == actions


def test_hand_computed_utility_and_budget_filter():
    c = EvolutionController(UtilityWeights(1, 1, 1, 1, 1), cost_scale=10, latency_scale=5)
    e = ActionEstimate(gain=.5, forgetting=.1, cost_seconds=2, delta_fpr=.05, latency_ms_per_1k=1)
    assert c.utility(e)["total"] == pytest.approx(.2 * (.5 - .1 - .2 - .05 - .2))
    r = c.decide(signature(), {Action.INCREMENTAL: e}, "v1", "event", budget_seconds=1)
    assert r.chosen_action == "no_op" and r.excluded["incremental"] == "adaptation_budget"


def test_no_op_wins_nonpositive_or_tied_utility_and_missing_candidates():
    c = EvolutionController()
    for estimates in ({}, {Action.INCREMENTAL: ActionEstimate()},
                      {Action.REPLAY: ActionEstimate(gain=-.1)}):
        assert c.decide(signature(), estimates, "v1", "event").chosen_action == "no_op"


def test_ablation_modes_are_distinct_and_deterministic():
    est = {Action.INCREMENTAL: ActionEstimate(gain=-.1), Action.RETRAIN: ActionEstimate(gain=.9)}
    def choose(mode):
        return EvolutionController(mode=mode).decide(signature(), est, "v1", "event").chosen_action
    assert choose("rules_only") == "incremental"
    assert choose("utility_only") == "full_retrain"
    assert choose("rules_utility") == "no_op"


@pytest.mark.parametrize("build", [lambda: UtilityWeights(-1, 0, 0, 0, 0),
                                      lambda: UtilityWeights(0, 0, 0, 0, 0),
                                      lambda: ActionEstimate(gain=float("nan")),
                                      lambda: ActionEstimate(cost_seconds=-1),
                                      lambda: ControllerThresholds(high_magnitude=.5),
                                      lambda: EvolutionController(cost_scale=0)])
def test_invalid_configuration_is_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_grid_is_normalized_and_validation_seeds_are_enforced():
    grid = list(weight_grid())
    keys = [tuple(w.normalized().values()) for w in grid]
    assert len(set(keys)) == len(keys) and all(sum(k) == pytest.approx(1) for k in keys)
    episode = dict(signature=signature(), family="test", seed=2001, estimates={}, outcomes={}, budget_seconds=10)
    with pytest.raises(ValueError, match="validation"):
        tune_weights([episode], 1, 1)
    episode.update(family="validation")  # renaming the family cannot launder a test seed
    with pytest.raises(ValueError, match="validation"):
        tune_weights([episode], 1, 1)


def test_weight_selection_prefers_validation_outcomes_and_logs_full_grid():
    episode = dict(signature=signature(), family="validation", seed=1001,
                   estimates={Action.NO_OP: ActionEstimate(), Action.INCREMENTAL: ActionEstimate(gain=.5, cost_seconds=.1)},
                   outcomes={"no_op": dict(recent_f1=.2, historical_f1=.8),
                             "incremental": dict(recent_f1=.9, historical_f1=.8)}, budget_seconds=10)
    weights, rows = tune_weights([episode], 1, 1)
    assert len(rows) == len(list(weight_grid()))
    r = EvolutionController(weights).decide(signature(), episode["estimates"], "v1", "event")
    assert r.chosen_action == "incremental"


def test_event_log_and_transport_preserve_full_reasoning(tmp_path):
    c = EvolutionController()
    r = c.decide(signature(), {Action.INCREMENTAL: ActionEstimate(gain=.5)}, "v3", "event-17")
    transport, p = InMemoryTransport(), tmp_path / "events.jsonl"
    publish_decision(r, path=p, transport=transport)
    message = transport.poll(TOPIC_EVOLUTION)
    wire = json.loads(message.value)
    assert wire == json.loads(p.read_text())
    assert wire["chosen_action"] == "incremental" and wire["deployment_status"] == "SHADOW_ONLY"
    assert {"gain", "forgetting", "cost", "fpr", "latency"} == set(wire["utilities"]["incremental"]["terms"])
    publish_decision(r, path=p)
    assert len(p.read_text().splitlines()) == 2


def test_repeated_source_flows_never_cross_holdout_roles():
    from evox.experiments.evolve import split_roles
    sources = np.array(["validation:3:17", "validation:0:21", "validation:3:17"])
    roles = split_roles(sources)
    assert roles[0] == roles[2]


def test_full_training_costs_use_validation_profiles_only():
    from evox.experiments.evolve import with_frozen_costs
    from evox.adaptation.strategies import StrategyConfig
    from evox.adaptation.data import AdaptationData, LabelPolicy, Samples
    from evox.models.ids import IDSModel
    s = Samples(np.zeros((8, 2)), np.resize([0, 3], 8), np.array([str(i) for i in range(8)]), np.full(8, -1))
    base = IDSModel("random_forest", params=dict(n_estimators=2))
    ep = dict(estimates={Action.INCREMENTAL: ActionEstimate(gain=.1, cost_seconds=999, latency_ms_per_1k=999)},
              data=AdaptationData(s, Samples.empty(2), 0, "v1"), policy=LabelPolicy(), base=base)
    out = with_frozen_costs(ep, {"incremental": dict(training_seconds_per_row=2, added_latency_ms_per_1k=3)}, StrategyConfig())
    assert out[Action.INCREMENTAL].cost_seconds == 16 and out[Action.INCREMENTAL].latency_ms_per_1k == 3


def test_fractional_label_budget_also_applies_to_error_detectors():
    from evox.evaluation.drift import Signals, StaticRun
    from evox.adaptation.data import LabelPolicy
    from evox.experiments.evolve import budget_detector_errors, sample_set
    n, bs = 4, 25
    samples = sample_set(np.zeros((n * bs, 2)), np.zeros(n * bs, int), "s", np.repeat(np.arange(n), bs))
    run = StaticRun(np.zeros(n * bs, int), np.ones(n * bs), np.ones((n, bs)), np.ones((n, bs)))
    sig = Signals(n, bs, run, np.zeros(n), np.zeros((n, 2)), np.zeros(n), 1., 4, {})
    policy = LabelPolicy(2, .25)
    masked = budget_detector_errors(sig, samples, policy)
    assert sum(map(len, masked.run.err_flags)) == len(policy.available(samples, n + 2))
    assert sum(map(len, masked.run.err_flags)) < n * bs
    assert sig.run.err_flags.shape == (n, bs)  # parent signals are untouched


def test_real_source_never_silently_falls_back_to_synthetic(tmp_path):
    from types import SimpleNamespace
    from evox.experiments.evolve import load_source
    with pytest.raises(ValueError, match="No processed stream"):
        load_source(SimpleNamespace(synthetic=False, stream_dir=tmp_path))
