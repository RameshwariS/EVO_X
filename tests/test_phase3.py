"""Phase 3 tests: wrapper, label remapping, metric functions on hand-computed cases."""
import numpy as np
import pytest

from src.ids_model import IDSModel
from src.metrics_stream import binary_metrics, known_class_macro_f1, novel_detection
from src.stream_builder import build_stream
from src.synthetic_data import generate_synthetic_week


@pytest.fixture(scope="module")
def stream():
    return build_stream(generate_synthetic_week(12_000, seed=3), reference_fraction=0.5, batch_size=400)


@pytest.mark.parametrize("kind", ["random_forest", "xgboost"])
def test_wrapper_returns_global_ids_and_handles_gaps(stream, kind):
    # known classes in the reference are NOT contiguous (global ids with gaps) -> XGBoost would crash without remap
    m = IDSModel(kind, seed=0, params=dict(n_estimators=20)).fit(stream.X_ref_train, stream.y_ref_train)
    assert set(np.unique(m.predict(stream.X_ref_val))) <= set(m.known_)
    assert set(m.known_) == set(np.unique(stream.y_ref_train))
    p, c = m.predict_with_confidence(stream.X_stream[:50])
    assert p.shape == c.shape == (50,) and ((c > 0) & (c <= 1)).all()
    # ref_val is chronologically later than ref_train, so it may contain classes the model never saw;
    # score only flows of trained classes (separable synthetic data)
    km = np.isin(stream.y_ref_val, m.known_)
    assert km.any() and (m.predict(stream.X_ref_val[km]) == stream.y_ref_val[km]).mean() > 0.9


def test_wrapper_rejects_single_class():
    with pytest.raises(ValueError):
        IDSModel("random_forest").fit(np.zeros((10, 3)), np.zeros(10, dtype=int))


def test_clone_and_save_roundtrip(stream, tmp_path):
    m = IDSModel("random_forest", seed=1, params=dict(n_estimators=10)).fit(stream.X_ref_train, stream.y_ref_train)
    c = m.clone()
    assert c.model_ is None and c.seed == 1
    m2 = IDSModel.load(m.save(tmp_path / "m.joblib"))
    np.testing.assert_array_equal(m.predict(stream.X_ref_val), m2.predict(stream.X_ref_val))


def test_binary_metrics_hand_computed():
    yt = np.array([0, 0, 0, 0, 1, 2, 3, 3])
    yp = np.array([0, 1, 0, 0, 1, 0, 3, 2])   # tp=3 (idx4,6,7) fp=1 fn=1 (idx5) tn=3
    m = binary_metrics(yt, yp)
    assert m["precision"] == pytest.approx(3 / 4) and m["recall"] == pytest.approx(3 / 4)
    assert m["fpr"] == pytest.approx(1 / 4) and m["fnr"] == pytest.approx(1 / 4)
    assert m["f1"] == pytest.approx(0.75)


def test_empty_slices_are_not_perfect():
    m = binary_metrics(np.zeros(5, int), np.zeros(5, int))   # no attacks at all
    assert m["recall"] == 0.0 and m["n_attack"] == 0           # support reported so it can be filtered


def test_known_macro_f1_ignores_novel_flows():
    yt = np.array([0, 0, 1, 1, 5, 5]); yp = np.array([0, 0, 1, 0, 0, 0])
    # class 5 is novel (not in known) -> excluded; class0 F1 = 2*(2/3)(1)/(5/3)=0.8, class1 F1 = 2*1*0.5/1.5=2/3
    assert known_class_macro_f1(yt, yp, [0, 1]) == pytest.approx((0.8 + 2 / 3) / 2)


def test_novel_detection():
    yt = np.array([0, 5, 5, 5, 6]); yp = np.array([0, 1, 0, 1, 0])
    d = novel_detection(yt, yp, [0, 1])
    assert d["n_novel"] == 4 and d["n_novel_flagged"] == 2 and d["novel_flag_rate"] == 0.5


def test_report_generator_flags_synthetic_and_reads_numbers(tmp_path):
    import pandas as pd
    from make_phase3_report import build_report

    rows, wrows = [], []
    for m in ["random_forest", "xgboost"]:
        for sd in range(3):
            rows.append(dict(model=m, seed=sd, data_tag="SYNTHETIC", train_protocol="ref_train_only", n_known_classes=4, n_train=100,
                             refval_bin_f1=0.9, refval_bin_fpr=0.01, stream_bin_f1=0.6 + 0.01 * sd + (m == "xgboost") * 0.02,
                             stream_bin_recall=0.5, stream_bin_fpr=0.01, stream_known_only_bin_f1=0.95, stream_known_only_n_attack=10,
                             stream_known_macro_f1=0.9, stream_novel_flag_rate=0.7, stream_novel_share=0.3, fit_seconds=1.0,
                             latency_ms_per_1k_flows=5.0))
            wrows += [dict(model=m, seed=sd, window=w, bin_n_attack=5, bin_recall=1 - 0.2 * w, bin_f1=1 - 0.2 * w) for w in range(4)]
    txt = build_report(pd.DataFrame(rows), pd.DataFrame(wrows))
    assert "NOT REPORTABLE" in txt
    assert "from 1.000 (best window) to 0.400" in txt   # recall range read from the window data
    assert "difference +0.3" in txt                # known-only 0.95 vs whole ~0.61 -> a positive gap is reported
