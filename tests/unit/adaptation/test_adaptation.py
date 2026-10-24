"""Phase B invariants: candidate isolation, global labels, data access and memory."""
import numpy as np
import pytest

from evox.adaptation.strategies import Action, AdaptationEngine, StrategyConfig
from evox.adaptation.data import AdaptationData, LabelPolicy, ReplayBuffer, Samples, combine
from evox.evolution.proxy import ProxyEvaluator, ProxyHoldouts
from evox.models.ids import IDSModel, IncrementalForest, UnsupportedUpdate


def samples(n=120, prefix="r", batch=0, classes=(0, 3), seed=0):
    rng = np.random.default_rng(seed)
    y = np.resize(np.array(classes), n)
    X = rng.normal(0, 0.2, (n, 4)).astype(np.float32) + y[:, None]
    return Samples(X, y, np.array([f"{prefix}:{i}" for i in range(n)]), np.full(n, batch))


@pytest.fixture
def base():
    s = samples(batch=-1)
    return IDSModel("random_forest", seed=1, params=dict(n_estimators=8, n_jobs=1)).fit(s.X, s.y)


def data(recent=None, memory=None, now=4, excluded=()):
    return AdaptationData(recent if recent is not None else samples(),
                          memory if memory is not None else samples(prefix="m", batch=-1, seed=2), now, "v7", excluded)


def test_delayed_labels_and_fraction_are_id_stable_and_reference_is_available():
    a = samples(batch=5)
    policy = LabelPolicy(2, 0.5, 3)
    assert len(policy.available(a, 6)) == 0
    ready = policy.available(a, 7)
    assert 0 < len(ready) < len(a)
    reverse = policy.available(a.take(np.arange(len(a))[::-1]), 7)
    assert set(ready.ids) == set(reverse.ids)
    assert len(policy.available(samples(batch=-1), 0)) == 120


def test_buffer_is_bounded_balanced_and_does_not_accept_future_labels():
    buffer = ReplayBuffer(30, 4)
    policy = LabelPolicy(2)
    buffer.add(samples(classes=(0, 3, 7), batch=2), policy, 3)
    assert len(buffer.samples) == 0
    buffer.add(samples(classes=(0, 3, 7), batch=2), policy, 4)
    assert len(buffer.samples) == 30
    assert np.bincount(np.searchsorted([0, 3, 7], buffer.samples.y)).tolist() == [10, 10, 10]
    buffer.add(samples(prefix="next", classes=(0, 3, 7), batch=3), policy, 5)
    assert len(buffer.samples) == 30 and len(set(buffer.samples.ids)) == 30
    buffer.expire(3)
    assert (buffer.samples.batches >= 3).all()


def test_union_deduplicates_and_rejects_conflicting_provenance():
    a = samples()
    assert len(combine(a, a)) == len(a)
    bad = Samples(a.X + 1, a.y, a.ids, a.batches)
    with pytest.raises(ValueError, match="Conflicting"):
        combine(a, bad)


@pytest.mark.parametrize("action", list(Action))
def test_every_strategy_produces_lineage_without_changing_parent(base, action):
    d = data()
    before = base.predict(d.recent.X).copy()
    c = AdaptationEngine(StrategyConfig(recent_capacity=80, memory_capacity=40, selection_capacity=50)).apply(
        action, base, d, LabelPolicy())
    assert c.parent_version == "v7" and c.action == action.value
    np.testing.assert_array_equal(base.predict(d.recent.X), before)
    assert c.model is base if action == Action.NO_OP else c.model is not base
    assert c.status == ("NO_OP" if action == Action.NO_OP else "CANDIDATE")
    assert set(c.training_ids) <= set(d.recent.ids) | set(d.memory.ids)
    if action != Action.NO_OP:
        assert c.profile["training_seconds"] > 0 and c.profile["model_bytes"] > 0
        assert c.profile["latency_ms_per_1k"] > 0 and len(c.training_snapshot_id) == 64
        assert c.profile["sampled_peak_rss_bytes"] >= c.profile["rss_before_bytes"] > 0
        assert c.profile["cpu_seconds"] >= 0


def test_incremental_preserves_trees_full_retrain_does_not_and_new_classes_align(base):
    d = data(recent=samples(classes=(0, 7), seed=3))
    engine = AdaptationEngine()
    inc = engine.apply(Action.INCREMENTAL, base, d, LabelPolicy()).model
    full = engine.apply(Action.RETRAIN, base, d, LabelPolicy()).model
    assert isinstance(inc.model_, IncrementalForest)
    assert len(inc.model_.estimators_) == len(base.model_.estimators_) == 8
    assert inc.known_.tolist() == [0, 3, 7] and full.known_.tolist() == [0, 7]
    assert inc.model_.members[0][1].tolist() == [0, 3]
    np.testing.assert_allclose(inc.predict_proba(d.recent.X).sum(axis=1), 1)
    assert set(np.unique(inc.predict(d.recent.X))) <= {0, 3, 7}
    assert (inc.predict(d.recent.X[d.recent.y == 7]) == 7).mean() > .9
    # A further update remains bounded and keeps the correct class space.
    inc.update(d.recent.X, d.recent.y)
    assert len(inc.model_.estimators_) == 8 and inc.known_.tolist() == [0, 3, 7]


def test_incremental_serialization_and_single_class_batch(base, tmp_path):
    recent = samples(classes=(7,))
    candidate = AdaptationEngine().apply(Action.INCREMENTAL, base, data(recent), LabelPolicy())
    restored = IDSModel.load(candidate.model.save(tmp_path / "incremental.joblib"))
    np.testing.assert_array_equal(restored.predict(recent.X), candidate.model.predict(recent.X))
    assert restored.known_.tolist() == [0, 3, 7]


def test_replay_uses_memory_retrain_uses_only_recent_and_strategic_is_informative(base):
    d = data(recent=samples(classes=(0, 7)), memory=samples(prefix="m", classes=(0, 3), batch=-1))
    engine = AdaptationEngine(StrategyConfig(selection_capacity=15))
    replay = engine.apply(Action.REPLAY, base, d, LabelPolicy())
    full = engine.apply(Action.RETRAIN, base, d, LabelPolicy())
    strategic = engine.apply(Action.STRATEGIC, base, d, LabelPolicy())
    assert replay.model.known_.tolist() == [0, 3, 7]
    assert set(full.training_ids) <= set(d.recent.ids)
    assert len(strategic.training_ids) == 15 and strategic.model.known_.tolist() == [0, 3, 7]
    selected_novel = sum(x in set(d.recent.ids[d.recent.y == 7]) for x in strategic.training_ids)
    assert selected_novel >= 10  # erroneous/unfamiliar novel flows dominate after class anchors


def test_no_strategy_sees_future_labels_or_evaluation_ids(base):
    engine = AdaptationEngine()
    d = data(recent=samples(batch=10), memory=Samples.empty(4), now=10)
    for action in (Action.RETRAIN, Action.REPLAY, Action.INCREMENTAL, Action.STRATEGIC):
        with pytest.raises(ValueError, match="No released"):
            engine.apply(action, base, d, LabelPolicy(2))
    with pytest.raises(ValueError, match="held-out"):
        engine.apply(Action.RETRAIN, base, data(excluded=("r:0",)), LabelPolicy())


def test_strategic_drops_stale_stream_memory_but_keeps_reference(base):
    recent = samples(classes=(0, 7), batch=100)
    stale = samples(prefix="stale", batch=0)
    ref = samples(prefix="historical", batch=-1)
    d = data(recent, combine(stale, ref), now=102)
    c = AdaptationEngine().apply(Action.STRATEGIC, base, d, LabelPolicy())
    assert not set(c.training_ids) & set(stale.ids)
    assert set(c.training_ids) & set(ref.ids)


def test_xgboost_continues_booster_and_rejects_class_expansion():
    s = samples(classes=(0, 3))
    m = IDSModel("xgboost", params=dict(n_estimators=4, n_jobs=1)).fit(s.X, s.y)
    rounds = m.model_.get_booster().num_boosted_rounds()
    m.update(s.X, s.y, boosting_rounds=3)
    assert m.model_.get_booster().num_boosted_rounds() == rounds + 3
    np.testing.assert_allclose(m.predict_proba(s.X).sum(axis=1), 1)
    novel = samples(classes=(0, 7))
    with pytest.raises(UnsupportedUpdate):
        m.update(novel.X, novel.y)


def test_proxy_holdouts_are_disjoint_and_obey_label_delay(base):
    d = data()
    good = ProxyHoldouts(samples(prefix="eval"), samples(prefix="hist", batch=-1))
    estimates, outcomes, profiles, unavailable, baseline = ProxyEvaluator().evaluate(base, d, LabelPolicy(), good)
    assert set(estimates) == set(Action) and not unavailable and len(profiles) == 4
    assert outcomes["no_op"]["recent_f1"] == baseline["recent_f1"]
    bad = ProxyHoldouts(d.recent, good.historical)
    with pytest.raises(ValueError, match="overlaps"):
        ProxyEvaluator().evaluate(base, d, LabelPolicy(), bad)
    future = ProxyHoldouts(samples(prefix="future", batch=4), good.historical)
    with pytest.raises(ValueError, match="unreleased"):
        ProxyEvaluator().evaluate(base, d, LabelPolicy(2), future)


def test_snapshot_tracks_values_and_arrays_are_readonly():
    s = samples()
    assert s.snapshot() == samples().snapshot()
    changed = Samples(s.X + 1, s.y, s.ids, s.batches)
    assert changed.snapshot() != s.snapshot()
    with pytest.raises(ValueError):
        s.X[0, 0] = 99
