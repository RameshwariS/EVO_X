"""Phase D1: portability classification, transfer metrics, adapter checks and report plumbing."""
import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evox.data.adapters import UNSW_RAW_COLUMNS, adapt_unsw_nb15_frame
from evox.evaluation.generalization import (advantage_table, classify_portability, decision_agreement,
                                            portability_table, summarize_transfer, transfer_ratio,
                                            utility_rank_correlation)
from evox.experiments import generalize


def results(f1, seeds=(1, 2), kinds=("abrupt", "recurring"), updates=None):
    """f1: {system: {kind: value}} -> per-scenario table with one scenario per (kind, seed)."""
    rows = []
    for system, by_kind in f1.items():
        for kind in kinds:
            for seed in seeds:
                rows.append(dict(system=system, scenario=f"{kind}_{seed}", seed=seed, kind=kind, macro_f1=by_kind[kind],
                                 updates=(updates or {}).get(system, 1), training_seconds=(updates or {}).get(system, 1) * 2.0))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("reference,observed,expected", [
    (0.05, 0.05, "holds"), (0.05, 0.03, "holds"), (0.05, 0.02, "shrinks"),
    (0.05, 0.0, "absent"), (0.05, -0.005, "absent"), (0.05, -0.02, "reversed"),
    (0.0, 0.03, "emerges"), (0.0, 0.0, "none"), (0.0, -0.05, "none"),
    (float("nan"), 0.05, "unknown"), (0.05, float("nan"), "unknown"), (None, 0.05, "unknown"),
])
def test_portability_is_classified_not_averaged(reference, observed, expected):
    assert classify_portability(reference, observed) == expected


def test_transfer_ratio_hand_computed_and_no_gain_is_undefined():
    assert transfer_ratio(0.70, 0.90, 0.80) == pytest.approx(0.5)
    assert transfer_ratio(0.70, 0.90, 0.90) == pytest.approx(1.0)
    assert transfer_ratio(0.70, 0.90, 0.60) == pytest.approx(-0.5)   # transfer can be worse than static
    assert transfer_ratio(0.70, 0.705, 0.70) is None                 # native gain below margin: nothing to transfer


def test_agreement_and_rank_correlation():
    assert decision_agreement({"a": "replay", "b": "no_op"}, {"a": "replay", "b": "full_retrain", "c": "x"}) == dict(n_decisions=2, agreement=0.5)
    assert decision_agreement({}, {"a": "x"}) == dict(n_decisions=0, agreement=None)
    assert utility_rank_correlation({"a": 1., "b": 2., "c": 3.}, {"a": 10., "b": 20., "c": 30.}) == pytest.approx(1.0)
    assert utility_rank_correlation({"a": 1., "b": 2., "c": 3.}, {"a": 3., "b": 2., "c": 1.}) == pytest.approx(-1.0)
    assert utility_rank_correlation({"a": 1., "b": 1., "c": 1.}, {"a": 1., "b": 2., "c": 3.}) is None   # constant ranking
    assert utility_rank_correlation({"a": 1., "b": 2.}, {"a": 1., "b": 2.}) is None                    # too few actions


def test_advantage_table_hand_computed_with_best_fixed():
    table = results(dict(evox=dict(abrupt=.90, recurring=.80), always_retrain=dict(abrupt=.80, recurring=.80),
                         always_replay=dict(abrupt=.85, recurring=.70), static=dict(abrupt=.5, recurring=.5)))
    adv = advantage_table(table, "DS", "random_forest")
    pick = lambda kind, base: adv[(adv.kind == kind) & (adv.baseline == base)].iloc[0]
    assert pick("abrupt", "always_retrain").advantage == pytest.approx(.10)
    assert pick("recurring", "always_retrain").advantage == pytest.approx(0.0)
    assert pick("abrupt", "best_fixed").advantage == pytest.approx(.05)       # best fixed in hindsight: replay .85
    assert pick("recurring", "best_fixed").advantage == pytest.approx(0.0)    # retrain .80 is the best fixed
    assert pick("ALL", "always_retrain").advantage == pytest.approx(.05)
    assert pick("ALL", "static").advantage == pytest.approx(.35)
    assert (adv.p_wilcoxon.isna()).all()                                      # two seeds: no significance claim
    assert pick("ALL", "always_retrain").n_seeds == 2 and pick("abrupt", "always_retrain").n_scenarios == 2


def test_advantage_table_cost_ratio_and_p_value_with_six_seeds():
    seeds = tuple(range(1, 7))
    table = results(dict(evox=dict(abrupt=.9, recurring=.9), always_retrain=dict(abrupt=.8, recurring=.8)), seeds=seeds,
                    updates=dict(evox=1, always_retrain=4))
    row = advantage_table(table, "DS", "xgboost").query("kind == 'ALL' and baseline == 'always_retrain'").iloc[0]
    assert row.cost_ratio == pytest.approx(0.25) and row.noninferior
    assert row.p_wilcoxon is not None and 0 < row.p_wilcoxon < 0.05          # six identical positive differences


def test_advantage_table_rejects_bad_input():
    with pytest.raises(ValueError):
        advantage_table(pd.DataFrame(dict(system=["evox"])), "DS", "random_forest")
    with pytest.raises(ValueError):
        advantage_table(results(dict(static=dict(abrupt=.5, recurring=.5))), "DS", "random_forest")


def test_portability_table_uses_reference_dataset_and_never_mixes_algorithms():
    strong = results(dict(evox=dict(abrupt=.90, recurring=.90), always_retrain=dict(abrupt=.80, recurring=.80)))
    weak = results(dict(evox=dict(abrupt=.80, recurring=.80), always_retrain=dict(abrupt=.80, recurring=.80)))
    both = pd.concat([advantage_table(strong, "REF", "random_forest"), advantage_table(weak, "OTHER", "random_forest"),
                      advantage_table(weak, "OTHER", "xgboost")], ignore_index=True)
    out = portability_table(both, "REF")
    pick = lambda d, a: out[(out.dataset == d) & (out.algorithm == a) & (out.kind == "ALL") & (out.baseline == "always_retrain")].iloc[0]
    assert pick("REF", "random_forest").portability == "reference"
    assert pick("OTHER", "random_forest").portability == "absent"
    assert math.isnan(pick("OTHER", "xgboost").reference_advantage) and pick("OTHER", "xgboost").portability == "unknown"
    with pytest.raises(ValueError):
        portability_table(both, "MISSING")


def transfer_table(native, transfer, static=0.70, seeds=range(1, 7)):
    rows = []
    for s in seeds:
        for system, value in (("static", static), ("evox_native", native), ("evox_transfer", transfer)):
            rows.append(dict(system=system, scenario=f"x_{s}", seed=s, kind="abrupt", macro_f1=value, updates=1, training_seconds=1.0))
    return pd.DataFrame(rows)


AGREE = pd.DataFrame(dict(agreement=[1.0, .5], n_decisions=[2, 2], rank_correlation=[.9, None]))


def test_h5_cannot_be_supported_by_synthetic_quick_or_few_seeds():
    for tag, quick, seeds in (("SYNTHETIC", False, range(1, 7)), ("REAL", True, range(1, 7)), ("REAL", False, range(1, 4))):
        out = summarize_transfer(transfer_table(.9, .9, seeds=seeds), AGREE, tag, quick)
        assert out["H5"]["status"] == "inconclusive"


def test_h5_supported_not_supported_and_no_gain_to_transfer():
    ok = summarize_transfer(transfer_table(.90, .88), AGREE, "REAL", False)
    assert ok["H5"]["status"] == "supported" and ok["transfer_ratio_pooled"] == pytest.approx(0.9)
    assert ok["action_agreement"] == pytest.approx(.75) and ok["n_decision_points"] == 4
    bad = summarize_transfer(transfer_table(.90, .72), AGREE, "REAL", False)
    assert bad["H5"]["status"] == "not_supported"
    partial = summarize_transfer(transfer_table(.90, .82), AGREE, "REAL", False)
    assert partial["H5"]["status"] == "inconclusive"
    flat = summarize_transfer(transfer_table(.70, .70), AGREE, "REAL", False)
    assert flat["H5"]["status"] == "inconclusive" and flat["transfer_ratio_pooled"] is None
    with pytest.raises(ValueError):
        summarize_transfer(transfer_table(.9, .9).query("system != 'static'"), AGREE, "REAL", False)


# --- adapters ---------------------------------------------------------------
def fake_unsw(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.random((n, len(UNSW_RAW_COLUMNS))), columns=UNSW_RAW_COLUMNS)
    d["srcip"], d["dstip"], d["proto"], d["state"], d["service"] = "10.0.0.1", "10.0.0.2", "tcp", "FIN", "-"
    d["Stime"] = 1421927414 + np.sort(rng.integers(0, 80_000, n))
    cats = np.array(["", "Fuzzers ", "Backdoors", "Exploits", "Generic"])
    d["attack_cat"] = cats[np.minimum(4, (d.Stime - d.Stime.min()) // 16_500)]
    d["Label"] = (d.attack_cat != "").astype(int)
    return d


def test_adapter_check_passes_good_frame_and_flags_each_defect():
    good = adapt_unsw_nb15_frame(fake_unsw())
    result = generalize.check_adapter_frame(good)
    assert result["ok"], [c for c in result["checks"] if not c["ok"]]
    assert result["summary"]["class_counts"]["BENIGN"] > 0
    failed = lambda frame: {c["name"] for c in generalize.check_adapter_frame(frame)["checks"] if not c["ok"]}
    assert "has_label_column" in failed(good.drop(columns=["Label"]))
    assert "timestamps_parse" in failed(good.drop(columns=["Timestamp"]))
    assert "at_least_two_attack_classes" in failed(good.assign(Label=np.where(good.Label == "Fuzzers", "Fuzzers", "BENIGN")))
    assert "has_benign" in failed(good.assign(Label="Fuzzers"))
    nan_heavy = good.copy()
    feats = [c for c in nan_heavy.columns if c not in ("Label", "Timestamp")]
    nan_heavy[feats] = np.nan
    assert "nan_share_reasonable" in failed(nan_heavy)


def test_build_stream_for_another_dataset_never_touches_cicids_reports(tmp_path):
    raw, out, metrics, figures = tmp_path / "raw", tmp_path / "stream", tmp_path / "metrics", tmp_path / "figures"
    raw.mkdir()
    fake_unsw().to_csv(raw / "UNSW-NB15_1.csv", header=False, index=False)
    with patch("evox.evaluation.stream.METRICS_DIR", metrics), patch("evox.evaluation.stream.FIGURES_DIR", figures):
        code = generalize.main(["build-stream", "--dataset", "unsw-nb15", "--raw-dir", str(raw), "--out-dir", str(out),
                                "--batch-size", "200"])
    assert code == 0 and (out / "stream.npz").is_file()
    assert (metrics / "phase_d_unsw_nb15_class_timeline.csv").is_file()
    assert not list(metrics.glob("phase2_*"))
    from evox.data.stream import EvoStream
    stream = EvoStream.load(out)
    assert stream.meta["dataset"] == "UNSW-NB15" and stream.meta["ordering_source"] == "timestamp"


def test_build_stream_refuses_a_failing_adapter_unless_forced(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    bad = fake_unsw()
    bad["attack_cat"] = ""          # everything benign: not an intrusion dataset
    bad.to_csv(raw / "UNSW-NB15_1.csv", header=False, index=False)
    assert generalize.main(["build-stream", "--dataset", "unsw-nb15", "--raw-dir", str(raw), "--out-dir", str(tmp_path / "o")]) == 1
    assert not (tmp_path / "o").exists()
    assert generalize.main(["adapter-check", "--dataset", "unsw-nb15", "--raw-dir", str(raw)]) == 1
    assert generalize.main(["adapter-check", "--dataset", "unsw-nb15", "--raw-dir", str(tmp_path / "missing")]) == 1


# --- report command ---------------------------------------------------------
def fake_phase_c_run(path, algorithm, advantage, data_tag="REAL", quick=False, seeds=range(1, 7)):
    path.mkdir(parents=True)
    results(dict(evox=dict(abrupt=.80 + advantage, recurring=.80 + advantage), always_retrain=dict(abrupt=.80, recurring=.80),
                 always_replay=dict(abrupt=.79, recurring=.79), static=dict(abrupt=.5, recurring=.5)),
            seeds=tuple(seeds)).to_csv(path / "per_scenario.csv", index=False)
    (path / "preregistered_protocol.json").write_text(json.dumps(dict(data_tag=data_tag, quick=quick, algorithm=algorithm)))


def test_parse_run_spec():
    assert generalize.parse_run_spec("UNSW-NB15/xgboost=runs/a b") == ("UNSW-NB15", "xgboost", generalize.Path("runs/a b"))
    for bad in ("nodataset=runs", "DS/svm=runs", "DS/random_forest"):
        with pytest.raises(ValueError):
            generalize.parse_run_spec(bad)


def test_report_classifies_each_dataset_and_flags_pipeline_only(tmp_path):
    fake_phase_c_run(tmp_path / "ref", "random_forest", 0.06)
    fake_phase_c_run(tmp_path / "unsw", "random_forest", 0.02)
    fake_phase_c_run(tmp_path / "syn", "random_forest", 0.06, data_tag="SYNTHETIC")
    out = tmp_path / "out"
    code = generalize.main(["report", "--run", f"CICIDS2017/random_forest={tmp_path / 'ref'}",
                            "--run", f"UNSW-NB15/random_forest={tmp_path / 'unsw'}", "--output-dir", str(out)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["reference_dataset"] == "CICIDS2017" and summary["pipeline_check_only"] is False
    assert summary["verdicts"] == {"CICIDS2017/random_forest": "reference", "UNSW-NB15/random_forest": "shrinks"}
    assert (out / "portability.csv").is_file() and (out / "portability.png").is_file()
    assert "REAL, >=6 seeds" in (out / "report.md").read_text()
    out2 = tmp_path / "out2"
    assert generalize.main(["report", "--run", f"A/random_forest={tmp_path / 'ref'}",
                            "--run", f"B/random_forest={tmp_path / 'syn'}", "--output-dir", str(out2)]) == 0
    assert json.loads((out2 / "summary.json").read_text())["pipeline_check_only"] is True


def test_report_rejects_nonempty_output_wrong_algorithm_and_missing_run(tmp_path):
    fake_phase_c_run(tmp_path / "ref", "random_forest", 0.05)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    run = f"A/random_forest={tmp_path / 'ref'}"
    assert generalize.main(["report", "--run", run, "--output-dir", str(tmp_path / "busy")]) == 2
    assert generalize.main(["report", "--run", f"A/xgboost={tmp_path / 'ref'}", "--output-dir", str(tmp_path / "o1")]) == 2
    assert generalize.main(["report", "--run", f"A/random_forest={tmp_path / 'none'}", "--output-dir", str(tmp_path / "o2")]) == 2
    assert generalize.main(["report", "--run", run, "--reference-dataset", "ZZ", "--output-dir", str(tmp_path / "o3")]) == 2


def test_transfer_fails_closed_on_missing_or_mismatched_inputs(tmp_path):
    common = ["transfer", "--synthetic", "--source-model", "xgboost", "--model", "random_forest",
              "--output-dir", str(tmp_path / "out")]
    assert generalize.main(common + ["--source-controller", str(tmp_path / "nope.json"), "--target-run", str(tmp_path / "t")]) == 2
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    assert generalize.main(["transfer", "--synthetic", "--source-model", "xgboost", "--model", "random_forest",
                            "--source-controller", "a", "--target-run", "b", "--output-dir", str(tmp_path / "busy")]) == 2
