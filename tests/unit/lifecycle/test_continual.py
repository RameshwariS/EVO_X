from dataclasses import replace

import numpy as np
import pytest

from evox.adaptation.data import LabelPolicy
from evox.adaptation.strategies import StrategyConfig
from evox.data.stream import build_stream
from evox.data.synthetic import generate_synthetic_week
from evox.drift.analyzer import DriftSignature
from evox.drift.scenarios import LabSource, ScenarioSpec
from evox.evolution.safety import SafetyGate
from evox.experiments.continual import (ReplayConfig, grouped_roles, historical_roles, pilot_candidates,
                                       prepare_context, simulate)
from evox.lifecycle.registry import FileModelRegistry


@pytest.fixture(scope="module")
def context():
    source = LabSource.from_evostream(build_stream(generate_synthetic_week(12000, seed=2), reference_fraction=.5))
    detector = dict(members={"psi":dict(psi_threshold=.25)}, fusion=dict(weights={"psi":1.},theta=.5))
    cfg = ReplayConfig(trees=6, reference_cap=250, holdout_cap=80, mmd_samples=20, mmd_null=5)
    ctx = prepare_context(source, ScenarioSpec("abrupt","med",2001,"test",batch_size=100), cfg, detector, LabelPolicy(2))
    ctx.signatures = [DriftSignature(0,40,48,40,5.,5.,5.,9,9,False,"abrupt",0.,.3,.35,0.,None,[1.])]
    return ctx


def test_five_source_roles_and_four_historical_roles_are_disjoint(context):
    roles = grouped_roles(["same","other","same"])
    assert roles[0] == roles[2]
    historical = list(context.historical.values())
    for i,a in enumerate(historical):
        for b in historical[i+1:]: assert not set(a.ids)&set(b.ids)
    data, held = context.training_data(48,"v1",StrategyConfig())
    assert not set(data.recent.ids) & {s for p in held.values() for s in p.ids}
    assert (data.recent.batches <= 46).all()


def test_test_scenarios_cannot_be_used_as_gate_pilots(context):
    with pytest.raises(ValueError,match="validation"):
        pilot_candidates(context,StrategyConfig())


def test_replay_scores_before_update_and_switches_the_active_model(context,tmp_path):
    config=StrategyConfig(recent_capacity=128,memory_capacity=128,selection_capacity=128)
    frozen=dict(weights=dict(gain=1.,forgetting=0.,cost=0.,fpr=0.,latency=0.), thresholds={},
                cost_scale=1.,latency_scale=1.,profiles={})
    # Fixed policies do not consult utility estimates; the controller config
    # still needs a valid threshold structure.
    from evox.evolution.controller import ControllerThresholds
    from dataclasses import asdict
    frozen["thresholds"]=asdict(ControllerThresholds())
    row,events,probes,batches,classes=simulate(context,"always_retrain",tmp_path/"retrain",SafetyGate(),
                                             frozen,config,config,"SYNTHETIC")
    assert row["updates"] == 1 and row["attempts"] == 1
    assert all(b["version"] == "v1" for b in batches[:49])
    assert all(b["version"] == "v2" for b in batches[49:])
    assert events[0]["parent_version"]=="v1" and events[0]["version"]=="v2"
    with FileModelRegistry(tmp_path/"retrain") as registry:
        assert registry.active_version=="v2"
        record=registry.versions["v2"]
        for sample_id in record["training_ids"]:
            if ":flow:" in sample_id:
                index=int(sample_id.rsplit(":",1)[1])
                assert index//100<=46 and context.roles[index]==0
    static,_,_,static_batches,_=simulate(context,"static",tmp_path/"static",SafetyGate(),frozen,config,config,"SYNTHETIC")
    assert static["updates"]==0 and static["n_report"]==row["n_report"]
    assert all(b["version"]=="v1" for b in static_batches)
