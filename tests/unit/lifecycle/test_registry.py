import json
import sys

import numpy as np
import pytest

from evox.adaptation.data import AdaptationData, LabelPolicy, Samples
from evox.adaptation.strategies import Action, AdaptationEngine
from evox.evaluation.candidate import EvaluationSets
from evox.evolution.safety import GateThresholds, SafetyGate
from evox.lifecycle.manager import EvolutionLifecycle, MonitorConfig, RollbackMonitor
from evox.lifecycle.registry import FileModelRegistry, RegistryError
from evox.models.ids import IDSModel


def sample(prefix, n=80, batch=-1, flipped=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    if flipped: y = 1 - y
    return Samples(X, y, np.array([f"{prefix}:{i}" for i in range(n)]), np.full(n, batch))


def setup(root):
    train = sample("ref")
    m = IDSModel("random_forest", params=dict(n_estimators=8, n_jobs=1)).fit(train.X, train.y)
    registry = FileModelRegistry(root)
    registry.initialize(m, train.ids)
    recent = sample("train", batch=0)
    c = AdaptationEngine().apply(Action.RETRAIN, m, AdaptationData(recent, train, 2, "v1"), LabelPolicy(2))
    return registry, m, c


def accept():
    return dict(promote=True, reason="test_accept")


def test_registry_persists_rejection_promotion_rollback_and_exact_predictions(tmp_path):
    registry, parent, c = setup(tmp_path)
    try:
        v2 = registry.register(c, {"signature": {"magnitude": 2}})
        assert registry.register(c) == v2
        registry.promote(v2, {"delta": .2}, accept())
        assert registry.active_version == v2 and registry.versions["v1"]["status"] == "SUPERSEDED"
        assert registry.rollback("degradation") == "v1"
        assert registry.versions[v2]["status"] == "ROLLED_BACK"
        expected = parent.predict(sample("test").X)
        np.testing.assert_array_equal(registry.load().predict(sample("test").X), expected)
        kinds = [e["kind"] for e in registry.events]
    finally:
        registry.close()
    with FileModelRegistry(tmp_path) as restored:
        assert restored.active_version == "v1"
        assert [e["kind"] for e in restored.events] == kinds
        np.testing.assert_array_equal(restored.load().predict(sample("test").X), expected)


def test_rejection_keeps_pointer_and_journal_is_append_only(tmp_path):
    r, _, c = setup(tmp_path)
    try:
        before = r.journal.read_bytes()
        v = r.register(c)
        r.reject(v, {"gain": -.1}, {"promote": False})
        assert r.active_version == "v1" and r.versions[v]["status"] == "REJECTED"
        assert r.journal.read_bytes().startswith(before)
        with pytest.raises(RegistryError): r.promote(v, {}, accept())
        with pytest.raises(RegistryError): FileModelRegistry(tmp_path)
    finally: r.close()


def test_stale_parent_and_artifact_tampering_are_rejected(tmp_path):
    r, _, c = setup(tmp_path)
    try:
        c.parent_version = "v99"
        with pytest.raises(RegistryError, match="Stale"): r.register(c)
        artifact = r.root / r.versions["v1"]["artifact"]
        with artifact.open("ab") as f: f.write(b"tampered")
        with pytest.raises(RegistryError, match="checksum"): r.load()
    finally: r.close()


def test_history_checksum_and_partial_tail_fail_closed(tmp_path):
    r, _, _ = setup(tmp_path)
    path = r.journal
    r.close()
    original = path.read_text()
    path.write_text(original.replace('INITIALIZED', 'CORRUPTION'))
    with pytest.raises(RegistryError, match="checksum"): FileModelRegistry(tmp_path)
    path.write_text(original + '{"partial":')
    with pytest.raises(RegistryError, match="Incomplete"): FileModelRegistry(tmp_path)


def test_caller_mutation_cannot_rewrite_committed_history(tmp_path):
    r, _, _ = setup(tmp_path)
    try:
        context = dict(reason="original")
        r.record_event("NOTE", context)
        context["reason"] = "mutated"
        assert r.events[-1]["payload"]["reason"] == "original"
    finally: r.close()
    with FileModelRegistry(tmp_path) as restored:
        assert restored.events[-1]["payload"]["reason"] == "original"


def test_lifecycle_evaluates_before_promotion_and_rejects_leakage_even_when_ungated(tmp_path):
    r, _, c = setup(tmp_path)
    try:
        lifecycle = EvolutionLifecycle(r, SafetyGate(GateThresholds(min_new_gain=0, max_historical_drop=1,
                                                                   max_forgetting=1, max_fpr_increase=1)))
        old = sample("old", seed=2)
        sets = EvaluationSets(sample("new", batch=0, seed=1), old)
        c.training_ids += [sets.recent.ids[0]]
        result = lifecycle.process(c, sets, LabelPolicy(2), 2, bypass_gate=True)
        assert result.status == "REJECTED" and lifecycle.holder.get()[1] == "v1"
        assert "invalid_evaluation" in result.decision["reason"]
        duplicate = lifecycle.process(c, sets, LabelPolicy(2), 2)
        assert duplicate.duplicate and len(r.versions) == 2
    finally: r.close()


def test_monitor_respects_delay_then_restores_parent_bit_identically(tmp_path):
    r, parent, c = setup(tmp_path)
    try:
        # Build a deliberately harmful candidate via an unconditional experimental policy.
        flipped = sample("flipped", batch=0, flipped=True)
        c = AdaptationEngine().apply(Action.RETRAIN, parent,
                AdaptationData(flipped, Samples.empty(3), 2, "v1"), LabelPolicy(2))
        life = EvolutionLifecycle(r, SafetyGate())
        sets = EvaluationSets(sample("gate", batch=0), sample("historical"))
        result = life.process(c, sets, LabelPolicy(2), 2, bypass_gate=True)
        assert result.status == "DEPLOYED"
        monitor = RollbackMonitor(life, parent, 2, sample("monitor-hist"), MonitorConfig(min_batches=2, min_rows=10))
        with pytest.raises(ValueError, match="unreleased"):
            monitor.observe(sample("m3", batch=3), LabelPolicy(2), 4, result.version)
        assert monitor.observe(sample("m3", batch=3), LabelPolicy(2), 5, result.version) is None
        evidence = monitor.observe(sample("m4", batch=4), LabelPolicy(2), 6, result.version)
        assert evidence["rolled_back"] and life.holder.get()[1] == "v1"
        test = sample("test").X
        np.testing.assert_array_equal(life.holder.get()[0].predict(test), parent.predict(test))
    finally: r.close()
