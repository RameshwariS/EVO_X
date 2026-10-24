import json
from dataclasses import asdict

import pytest

from evox.evolution.controller import ControllerThresholds, UtilityWeights
from evox.experiments.compare import load_frozen_controller


def config(tmp_path):
    data=dict(data_tag="SYNTHETIC",algorithm="random_forest",validation_seeds=[1001],
              written_before_test_scenarios_built=True,label_policy=dict(delay_batches=2,labelled_fraction=1.,seed=0),
              weights=asdict(UtilityWeights()),thresholds=asdict(ControllerThresholds()),cost_scale=1.,latency_scale=1.,
              profiles={"incremental":dict(training_seconds_per_row=.01,added_latency_ms_per_1k=.1)})
    path=tmp_path/"controller.json"
    path.write_text(json.dumps(data))
    return path,data


@pytest.mark.parametrize("change",[{"data_tag":"REAL"},{"algorithm":"xgboost"},{"validation_seeds":[2001]},
                                     {"written_before_test_scenarios_built":False},{"cost_scale":0}])
def test_mismatched_or_unfrozen_phase_b_configuration_is_rejected(tmp_path,change):
    path,data=config(tmp_path)
    path.write_text(json.dumps({**data,**change}))
    with pytest.raises(ValueError): load_frozen_controller(path,"SYNTHETIC","random_forest")


def test_self_contained_phase_b_config_is_accepted(tmp_path):
    path,data=config(tmp_path)
    frozen,protocol=load_frozen_controller(path,"SYNTHETIC","random_forest")
    assert frozen==data and protocol["label_policy"]["delay_batches"]==2
