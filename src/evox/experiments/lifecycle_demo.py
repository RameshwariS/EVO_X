"""Small deterministic Phase C demonstration of promote, reject and rollback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evox.adaptation.data import AdaptationData, LabelPolicy, Samples
from evox.adaptation.strategies import Action, AdaptationEngine, StrategyConfig
from evox.config import RESULTS_DIR
from evox.evaluation.candidate import EvaluationSets
from evox.evolution.safety import GateThresholds, SafetyGate
from evox.experiments.evolve import write_json
from evox.lifecycle.manager import EvolutionLifecycle, MonitorConfig, RollbackMonitor
from evox.lifecycle.registry import FileModelRegistry
from evox.models.ids import IDSModel


def regime_samples(prefix, regime, flipped=False, batch=-1, seed=0, n=400):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    X = np.column_stack([x, np.full(n, regime), rng.normal(0, .01, n)]).astype(np.float32)
    y = (x > 0).astype(int)
    if flipped:
        y = 1 - y
    return Samples(X, y, np.array([f"{prefix}:{i}" for i in range(n)]), np.full(n, batch))


def run_demo(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty demo output directory")
    policy = LabelPolicy(2)
    historical_train = regime_samples("historical-train", 0, seed=1)
    new_train = regime_samples("new-train", 3, True, 0, seed=2)
    historical_gate = regime_samples("historical-gate", 0, seed=3)
    new_gate = regime_samples("new-gate", 3, True, 0, seed=4)
    base = IDSModel("random_forest", seed=5, params=dict(n_estimators=64, max_depth=10, n_jobs=1)).fit(
        historical_train.X, historical_train.y)
    gate = SafetyGate(GateThresholds(min_new_gain=.05, max_historical_drop=.05, max_forgetting=.05,
                                     max_fpr_increase=.02, max_training_seconds=60))
    engine = AdaptationEngine(StrategyConfig(recent_capacity=400, memory_capacity=400))
    output.mkdir(parents=True, exist_ok=True)
    with FileModelRegistry(output / "registry") as registry:
        registry.initialize(base, historical_train.ids, dict(data_tag="SYNTHETIC", purpose="lifecycle demonstration"))
        life = EvolutionLifecycle(registry, gate)
        data = AdaptationData(new_train, historical_train, 2, "v1",
                              tuple(new_gate.ids) + tuple(historical_gate.ids))
        replay = engine.apply(Action.REPLAY, base, data, policy)
        promoted = life.process(replay, EvaluationSets(new_gate, historical_gate), policy, 2)
        if promoted.status != "DEPLOYED":
            raise RuntimeError("Demo replay did not meet the declared promotion criteria")
        deployed, version = life.holder.get()
        fresh = engine.apply(Action.RETRAIN, deployed,
                AdaptationData(new_train, historical_train, 2, version, data.excluded_ids), policy)
        rejected = life.process(fresh, EvaluationSets(new_gate, historical_gate), policy, 2)
        if rejected.status != "REJECTED" or life.holder.get()[1] != version:
            raise RuntimeError("Unsafe recent-only candidate was not rejected")
        # Traffic keeps its new feature regime but returns to the original label
        # relationship. Parent predictions become preferable on fresh monitor data.
        monitor = RollbackMonitor(life, base, 2, regime_samples("monitor-historical", 0, seed=7),
                                  MonitorConfig(min_batches=2, min_rows=32))
        first = monitor.observe(regime_samples("monitor-3", 3, False, 3, seed=8), policy, 5, version)
        second = monitor.observe(regime_samples("monitor-4", 3, False, 4, seed=9), policy, 6, version)
        if not second or not second["rolled_back"]:
            raise RuntimeError("Post-promotion monitor did not trigger the expected rollback")
        probe = regime_samples("verification", 0, seed=10)
        identical = np.array_equal(life.holder.get()[0].predict(probe.X), base.predict(probe.X))
        summary = dict(data_tag="SYNTHETIC", promoted_version=promoted.version, rejected_version=rejected.version,
                       restored_version=life.holder.get()[1], parent_predictions_identical=bool(identical),
                       promotion=promoted.evaluation, rejection=rejected.decision, rollback=second)
        write_json(output / "summary.json", summary)
    # Confirm durable replay restores the active pointer, independently of memory.
    with FileModelRegistry(output / "registry") as reopened:
        if reopened.active_version != "v1":
            raise RuntimeError("Registry replay did not restore the rollback state")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "runs" / "phase_c_lifecycle_demo")
    args = parser.parse_args(argv)
    summary = run_demo(args.output_dir)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("promotion","rejection","rollback")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
