"""Phase 6 tests: detectors on synthetic step/ramp signals with KNOWN change points; labels never leak."""
import numpy as np
import pytest

from evox.drift.detectors import (ADWINDetector, BatchObs, EdgeTrigger, FusedDetector, MMDDetector,
                                 PageHinkleyDetector, PSIDetector, fuse, normalise_weights)

rng = np.random.default_rng(0)
D = 8


def _obs(t, X=None, conf=None, err=None, n=300):
    X = rng.standard_normal((n, D)).astype(np.float32) if X is None else X
    return BatchObs(t, X, np.full(len(X), 0.95, np.float32) if conf is None else conf, err, None if err is None else t)


def _first(alarms, after=0):
    a = [t for t, v in enumerate(alarms) if v and t >= after]
    return a[0] if a else None


# -- ADWIN -------------------------------------------------------------------
def test_adwin_error_detects_step_with_known_change_point_and_is_quiet_before():
    det, al = ADWINDetector("error"), []
    for t in range(80):
        p = 0.02 if t < 40 else 0.20
        al.append(det.update(_obs(t, err=(rng.random(300) < p).astype(float))))
    assert not any(al[:40]) and _first(al, 40) is not None and _first(al, 40) - 40 <= 5


def test_adwin_conf_is_label_free():
    det = ADWINDetector("conf")
    assert det.needs_labels is False
    al = [det.update(_obs(t, conf=np.full(300, 0.97 if t < 30 else 0.6, np.float32))) for t in range(60)]
    assert _first(al, 30) is not None and not any(al[:30])


def test_adwin_error_ignores_missing_labels():
    det = ADWINDetector("error")
    assert det.update(_obs(0, err=None)) is False and det.score == 0.0


def test_adwin_stationary_false_alarm_rate_is_low():
    det, n = ADWINDetector("error", delta=0.002), 0
    for t in range(300):
        n += det.update(_obs(t, err=(rng.random(300) < 0.03).astype(float)))
    assert n <= 2


# -- Page-Hinkley --------------------------------------------------------------
def test_ph_detects_abrupt_step_not_stationary_noise():
    ph = PageHinkleyDetector(delta=0.005, threshold=0.5)
    x = np.r_[rng.normal(0.03, 0.005, 60), rng.normal(0.25, 0.01, 30)]
    al = [ph.update_value(float(v)) for v in x]
    assert not any(al[:60]) and 60 <= _first(al, 60) <= 66


def test_ph_detects_slow_ramp_later_than_step():
    ph = PageHinkleyDetector(delta=0.005, threshold=0.5)
    x = np.r_[np.full(40, 0.03), 0.03 + np.linspace(0, 0.3, 60)] + rng.normal(0, 0.003, 100)
    al = [ph.update_value(float(v)) for v in x]
    assert _first(al, 40) is not None and _first(al, 40) > 45


def test_ph_one_sided_ignores_decrease():
    ph = PageHinkleyDetector()
    x = np.r_[np.full(40, 0.3), np.full(40, 0.05)] + rng.normal(0, 0.005, 80)
    assert not any(ph.update_value(float(v)) for v in x)


def test_ph_resets_after_alarm_and_needs_labels():
    ph = PageHinkleyDetector(threshold=0.2, min_instances=3)
    assert ph.needs_labels and any(ph.update_value(v) for v in [0.0] * 5 + [1.0] * 5) and ph.n < 5


# -- PSI -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ref():
    return np.random.default_rng(1).standard_normal((20_000, D)).astype(np.float32)


def test_psi_known_value_against_hand_computation():
    r = np.random.default_rng(2).standard_normal((50_000, 1)).astype(np.float32)
    det = PSIDetector(r, n_bins=10, topk=1)
    x = r[:, 0] + 1.0                                          # shift of one sigma
    got = det.psi_vector(x[:, None])[0]
    e = det.edges[0]; pe = np.full(10, 0.1)
    pa = np.bincount(np.searchsorted(e, x, side="right"), minlength=10) / len(x)
    assert got == pytest.approx(float(np.sum((pa - pe) * np.log(pa / pe))), rel=0.05)
    assert got > 0.25                                           # a 1-sigma shift is 'major' by convention


def test_psi_same_distribution_is_below_threshold(ref):
    det = PSIDetector(ref, window_batches=4, threshold=0.25)
    for t in range(10):
        det.update(_obs(t, n=500))
    assert det.stat < 0.05 and det.score < 0.25


def test_psi_detects_feature_shift_and_names_the_feature(ref):
    det, al = PSIDetector(ref, window_batches=3), []
    for t in range(30):
        X = rng.standard_normal((500, D)).astype(np.float32)
        if t >= 15: X[:, 2] += 1.5
        al.append(det.update(_obs(t, X=X)))
    assert not any(al[:15]) and 15 <= _first(al, 15) <= 17
    assert int(np.argmax(det.vector)) == 2


def test_psi_alarm_is_edge_triggered_and_rearms(ref):
    det, al = PSIDetector(ref, window_batches=2), []
    for t in range(60):
        X = rng.standard_normal((500, D)).astype(np.float32)
        if 15 <= t < 30: X[:, 0] += 2
        al.append(det.update(_obs(t, X=X)))
    assert sum(al) == 1                                         # one alarm for the whole shifted segment
    for t in range(60, 80):
        X = rng.standard_normal((500, D)).astype(np.float32)
        if t >= 70: X[:, 0] += 2
        al.append(det.update(_obs(t, X=X)))
    assert sum(al) == 2                                         # re-armed after the revert


def test_psi_handles_constant_and_tied_features():
    r = np.zeros((2000, 3), np.float32); r[:, 1] = np.random.default_rng(3).integers(0, 2, 2000); r[:, 2] = np.random.default_rng(4).standard_normal(2000)
    det = PSIDetector(r)
    assert np.isfinite(det.psi_vector(r[:500])).all()


# -- MMD ---------------------------------------------------------------------------
def test_mmd_null_is_quiet_and_shift_is_loud(ref):
    det = MMDDetector(ref, window_batches=2, n_sub=300, n_ref=300, n_null=40, seed=0)
    quiet = [det.observe(rng.standard_normal((300, D)).astype(np.float32)) for _ in range(10)]
    det.window.clear()
    X = rng.standard_normal((300, D)).astype(np.float32); X[:, :3] += 1.0
    loud = [det.observe(X) for _ in range(3)]
    assert max(quiet) < 3 * det.null_q99 and min(loud) > 5 * det.null_q99


def test_mmd_detects_multivariate_shift_that_marginals_miss():
    """Correlation flip: every marginal stays N(0,1) (PSI blind), but the joint changes."""
    r = np.random.default_rng(5)
    cov0 = np.array([[1, .9], [.9, 1]]); cov1 = np.array([[1, -.9], [-.9, 1]])
    ref2 = r.multivariate_normal([0, 0], cov0, 20_000).astype(np.float32)
    new = r.multivariate_normal([0, 0], cov1, 600).astype(np.float32)
    psi, mmd = PSIDetector(ref2, window_batches=1), MMDDetector(ref2, window_batches=1, n_sub=400, n_ref=400, n_null=40)
    psi.observe(new); mmd.observe(new)
    assert psi.stat < 0.05 and mmd.stat > 3 * mmd.null_q99


def test_mmd_reproducible_with_seed(ref):
    X = np.random.default_rng(9).standard_normal((300, D)).astype(np.float32)
    a, b = MMDDetector(ref, seed=3, n_null=10), MMDDetector(ref, seed=3, n_null=10)
    assert a.observe(X) == b.observe(X)


# -- fusion / trigger ------------------------------------------------------------------
def test_fuse_weights_and_clipping():
    w = normalise_weights({"a": 1, "b": 1, "c": 2})
    assert sum(w.values()) == pytest.approx(1.0)
    assert fuse({"a": 5.0, "b": 0.0, "c": 0.0}, w) == pytest.approx(0.25)     # clipped at 1 -> 0.25*1
    assert fuse({"a": 1, "b": 1, "c": 1}, w) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalise_weights({"a": 0})


def test_edge_trigger_hysteresis():
    tr = EdgeTrigger(rearm=0.5)
    seq = [0.2, 1.2, 1.5, 0.8, 1.1, 0.3, 1.0]
    assert [tr.update(s) for s in seq] == [False, True, False, False, False, False, True]


def test_fused_detector_requires_enough_weight_to_alarm(ref):
    psi, mmd = PSIDetector(ref, window_batches=2), MMDDetector(ref, window_batches=2, n_sub=300, n_ref=300, n_null=30)
    fd = FusedDetector([psi, mmd], {"psi": 1, "mmd": 1}, theta=0.9)             # both must agree
    al = []
    for t in range(20):
        X = rng.standard_normal((400, D)).astype(np.float32)
        if t >= 10: X[:, :3] += 1.5
        al.append(fd.update(_obs(t, X=X)))
    assert not any(al[:10]) and sum(al) == 1 and 10 <= _first(al) <= 13


# -- no detector sees labels it should not -------------------------------------------------
def test_label_free_detectors_work_when_labels_are_none_and_fail_loudly_if_touched(ref):
    class Poison:                       # any access to labels raises
        def __len__(self): raise AssertionError("label-free detector touched labels")
        def __iter__(self): raise AssertionError("label-free detector touched labels")
        def __array__(self, *a, **k): raise AssertionError("label-free detector touched labels")
    for det in (PSIDetector(ref), MMDDetector(ref, n_null=10), ADWINDetector("conf")):
        assert det.needs_labels is False
        o = _obs(0, n=300); o.err = Poison()
        det.update(o)                   # must not raise
    for det in (ADWINDetector("error"), PageHinkleyDetector()):
        assert det.needs_labels is True
