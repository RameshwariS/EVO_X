"""Phase 6 tests: scoring rules, aggregation, offline==online equivalence, analyzer on scenario truth."""
import numpy as np
import pytest

from src.drift_analyzer import DriftAnalyzer, SignatureDB
from src.drift_detectors import BatchObs, MMDDetector, PSIDetector
from src.drift_eval import (MemberConfig, aggregate, compute_signals, criterion, fused_alarms, run_member,
                            run_members, score_alarms, static_run, train_static)
from src.drift_scenarios import LabSource, ScenarioSpec, generate_scenario
from src.stream_builder import build_stream
from src.synthetic_data import generate_synthetic_week


def T(events, n=100):
    return dict(n_batches=n, events=[dict(id=i, **e) for i, e in enumerate(events)])


ONSET = dict(type="onset", start_batch=40, end_batch=40, scored=True)


# -- scoring rules ---------------------------------------------------------------
def test_score_detection_delay_and_window():
    r = score_alarms([43], T([ONSET]), tol=10)
    assert r["n_detected"] == 1 and r["events"][0]["delay"] == 3 and r["n_false"] == 0


def test_alarm_before_event_is_false_after_window_is_false():
    r = score_alarms([10, 52], T([ONSET]), tol=10)           # window is [40, 50]
    assert r["n_false"] == 2 and r["n_missed"] == 1


def test_repeat_alarms_in_matched_window_are_not_false():
    r = score_alarms([41, 44, 48], T([ONSET]), tol=10)
    assert r["n_detected"] == 1 and r["n_repeat"] == 2 and r["n_false"] == 0


def test_window_boundaries_are_inclusive():
    assert score_alarms([40], T([ONSET]), 10)["n_detected"] == 1
    assert score_alarms([50], T([ONSET]), 10)["n_detected"] == 1
    assert score_alarms([51], T([ONSET]), 10)["n_false"] == 1


def test_alarm_at_unscored_revert_is_neutral():
    ev = [ONSET, dict(type="revert", start_batch=60, end_batch=60, scored=False)]
    r = score_alarms([41, 62], T(ev), 10)
    assert r["n_false"] == 0 and r["n_neutral"] == 1 and r["n_detected"] == 1


def test_recurring_second_onset_needs_its_own_alarm():
    ev = [ONSET, dict(type="revert", start_batch=65, end_batch=65, scored=False),
          dict(type="onset", start_batch=90, end_batch=90, scored=True)]
    r = score_alarms([41], T(ev), 10)
    assert r["n_detected"] == 1 and r["n_missed"] == 1


def test_control_stream_counts_only_false_alarms_and_rate():
    r = score_alarms([5, 50], T([], n=200), 10)
    agg = aggregate([(T([], 200), r)])
    assert agg["n_false"] == 2 and agg["fa_per_1000"] == pytest.approx(10.0) and np.isnan(agg["detection_rate"])


def test_aggregate_and_criterion():
    a = score_alarms([42], T([ONSET]), 10); b = score_alarms([], T([ONSET]), 10)
    agg = aggregate([(T([ONSET]), a), (T([ONSET]), b)])
    assert agg["detection_rate"] == 0.5 and agg["mean_delay"] == 2 and agg["n_missed"] == 1
    assert criterion(agg) == pytest.approx(0.5 - 0.0 - 0.002 * 2)


# -- end-to-end on generated scenarios --------------------------------------------------
@pytest.fixture(scope="module")
def world():
    s = build_stream(generate_synthetic_week(50_000, seed=2), reference_fraction=0.5, batch_size=500)
    src = LabSource.from_evostream(s)
    sc = generate_scenario(src, ScenarioSpec("abrupt", "high", 1, "validation"))
    model = train_static(sc)
    run = static_run(sc, model)
    return src, sc, model, run, compute_signals(sc, run)


def test_batchwise_and_whole_stream_scoring_agree(world):
    _, sc, model, run, _ = world
    yp = np.concatenate([model.predict(b.X) for b in sc.stream.iter_batches()])
    assert np.array_equal(yp, run.y_pred)


def test_label_free_detectors_alarm_at_the_change_on_abrupt_high(world):
    _, sc, _, _, sig = world
    out = run_members(sig, MemberConfig())
    c = sc.truth["change_batch"]
    for nm in ("psi", "mmd", "adwin_conf"):
        r = score_alarms(out[nm].alarms, sc.truth)
        assert r["n_detected"] == 1 and r["events"][0]["delay"] <= 3, nm


def test_error_detectors_are_delayed_by_label_delay(world):
    _, sc, _, run, sig = world
    k0 = run_member("adwin_err", sig, MemberConfig(), label_delay=0).alarms
    k5 = run_member("adwin_err", sig, MemberConfig(), label_delay=5).alarms
    c = sc.truth["change_batch"]
    f0 = [a for a in k0 if a >= c]; f5 = [a for a in k5 if a >= c]
    if f0 and f5:                                   # (this scenario barely moves the error rate; both may be silent)
        assert f5[0] - f0[0] == 5
    # before any label has arrived nothing can fire
    assert not [a for a in k5 if a < 5]


def test_offline_psi_mmd_alarms_equal_online_detector_classes(world):
    _, sc, _, run, sig = world
    ref = sc.stream.X_ref_train
    psi = PSIDetector(ref, window_batches=sig.window_batches, seed=0)
    mmd = MMDDetector(ref, window_batches=sig.window_batches, n_sub=400, n_ref=400, n_null=60, seed=0)
    cfg = MemberConfig(); off = run_members(sig, cfg, ("psi", "mmd"))
    a_psi, a_mmd = [], []
    for b in sc.stream.iter_batches():
        o = BatchObs(b.index, b.X, run.conf_batches[b.index])
        if psi.update(o): a_psi.append(b.index)
        if mmd.update(o): a_mmd.append(b.index)
    assert a_psi == off["psi"].alarms and a_mmd == off["mmd"].alarms


def test_stationary_control_is_almost_quiet(world):
    src, *_ = world
    sc = generate_scenario(src, ScenarioSpec("stationary", "low", 7, "validation", n_batches=120))
    sig = compute_signals(sc, static_run(sc, train_static(sc)))
    out = run_members(sig, MemberConfig(), ("psi", "mmd", "adwin_conf", "adwin_err"))
    # MMD's threshold is a null 99th percentile x1.5 on overlapping windows, so ~1 spurious alarm / ~120 batches is
    # expected (measured: ~2.5 per 1000 batches on the test controls); everything else must be silent.
    assert all(len(o.alarms) == 0 for k, o in out.items() if k != "mmd") and len(out["mmd"].alarms) <= 1


def test_fusion_of_agreeing_members_alarms_once_near_change(world):
    _, sc, _, _, sig = world
    out = run_members(sig, MemberConfig())
    f = fused_alarms(out, {"psi": 1, "mmd": 1, "adwin_err": 0, "ph_err": 0}, theta=0.9)
    r = score_alarms(f.alarms, sc.truth)
    assert r["n_detected"] == 1 and r["n_false"] == 0


# -- analyzer --------------------------------------------------------------------------
def _analyzer(src, sc, sig):
    cfg = MemberConfig()
    model = train_static(sc)
    return DriftAnalyzer(sig.ref_psi_detector, cfg.psi_threshold, cfg.mmd_mult * sig.mmd_null_q99,
                         ref_val_conf=model.predict_with_confidence(sc.stream.X_ref_val)[1])


def _sig(an, sc, sig, alarm, ahead=8, register=True):
    nb, bs = sc.spec.n_batches, sc.spec.batch_size
    return an.characterize(alarm, alarm + ahead, sig.psi_stat, sig.mmd_stat,
                           sc.stream.X_stream.reshape(nb, bs, -1), sig.run.conf_batches, register)


def test_signature_has_all_fields_and_onset_is_sensible(world):
    src, sc, _, _, sig = world
    an = _analyzer(src, sc, sig); c = sc.truth["change_batch"]
    s = _sig(an, sc, sig, c + 1)
    d = s.to_dict()
    assert {"magnitude", "persistence", "duration", "type_guess", "novelty", "per_feature_psi", "recurrence"} <= set(d)
    assert c - 1 <= s.onset_estimate <= c + 1 and s.persistence >= 6 and not s.ended


def test_analyzer_magnitude_increases_with_scenario_magnitude(world):
    src, *_ = world
    mags = []
    for m in ("low", "med", "high"):
        sc = generate_scenario(src, ScenarioSpec("abrupt", m, 1, "validation"))
        sig = compute_signals(sc, static_run(sc, train_static(sc)))
        an = _analyzer(src, sc, sig)
        mags.append(_sig(an, sc, sig, sc.truth["change_batch"] + 1).magnitude)
    assert mags[0] < mags[1] < mags[2]


def test_per_feature_psi_points_at_shifted_features(world):
    src, sc, _, _, sig = world
    s = _sig(_analyzer(src, sc, sig), sc, sig, sc.truth["change_batch"] + 1)
    shifted = [sc.stream.feature_names.index(f) for f in sc.truth["affected_features"]]
    top = np.argsort(s.per_feature_psi)[-len(shifted) - 2:]
    assert set(shifted) <= set(top.tolist())


def test_temporary_is_recognised_as_ended_and_short(world):
    src, *_ = world
    sc = generate_scenario(src, ScenarioSpec("temporary", "high", 1, "validation"))
    sig = compute_signals(sc, static_run(sc, train_static(sc)))
    s = _sig(_analyzer(src, sc, sig), sc, sig, sc.truth["change_batch"] + 1, ahead=12)
    assert s.ended and s.type_guess == "temporary"


def test_recurring_second_onset_matches_stored_signature(world):
    src, *_ = world
    sc = generate_scenario(src, ScenarioSpec("recurring", "high", 1, "validation"))
    sig = compute_signals(sc, static_run(sc, train_static(sc)))
    an = _analyzer(src, sc, sig)
    t = sc.truth; first = _sig(an, sc, sig, t["change_batch"] + 1)
    second = _sig(an, sc, sig, t["change_batch"] + 2 * sc.spec.recurring_len + 1)
    assert first.recurrence == 0.0 and second.recurrence > 0.9 and second.recurrence_match == first.signature_id
    assert second.type_guess == "recurring"


def test_different_drift_is_not_matched_as_recurrence(world):
    src, *_ = world
    a = generate_scenario(src, ScenarioSpec("abrupt", "high", 1, "validation"))
    b = generate_scenario(src, ScenarioSpec("abrupt", "high", 2, "validation"))   # different features/classes
    sa = compute_signals(a, static_run(a, train_static(a))); sb = compute_signals(b, static_run(b, train_static(b)))
    an = _analyzer(src, a, sa)
    _sig(an, a, sa, a.truth["change_batch"] + 1)
    s2 = _sig(an, b, sb, b.truth["change_batch"] + 1)
    assert s2.recurrence < 0.9


def test_confidence_novelty_is_positive_but_NOT_specific_to_novel_classes(world):
    """FINDING (kept as a test so it cannot be forgotten): confidence-based novelty fires for any drift that
    moves flows away from the training data -- a 3-sigma covariate shift on known classes gives MORE low-confidence
    flows than a new attack class does.  So `novelty` alone cannot tell the controller 'new class' from
    'covariate shift'; the Phase-B rules must not treat high novelty as proof of an unseen class."""
    src, *_ = world
    out = {}
    for k in ("novel_class", "abrupt"):
        sc = generate_scenario(src, ScenarioSpec(k, "high", 1, "validation"))
        sig = compute_signals(sc, static_run(sc, train_static(sc)))
        out[k] = _sig(_analyzer(src, sc, sig), sc, sig, sc.truth["change_batch"] + 1).novelty
    assert out["novel_class"] > 0.05 and out["abrupt"] > 0.05


def test_no_lookahead_beyond_now(world):
    src, sc, _, _, sig = world
    an1, an2 = _analyzer(src, sc, sig), _analyzer(src, sc, sig)
    c = sc.truth["change_batch"] + 1
    s1 = _sig(an1, sc, sig, c, ahead=5)
    ps2, ms2 = sig.psi_stat.copy(), sig.mmd_stat.copy(); ps2[c + 6:] = 123.0; ms2[c + 6:] = 123.0   # corrupt the future
    nb, bs = sc.spec.n_batches, sc.spec.batch_size
    X = sc.stream.X_stream.reshape(nb, bs, -1).copy(); X[c + 6:] = 1e6
    s2 = an2.characterize(c, c + 5, ps2, ms2, X, sig.run.conf_batches)
    assert s1.magnitude == s2.magnitude and s1.type_guess == s2.type_guess and s1.persistence == s2.persistence


def test_signature_db_cosine():
    db = SignatureDB(); db.add(0, np.array([1.0, 0, 0])); db.add(1, np.array([0, 1.0, 0]))
    sim, i = db.best_match(np.array([0.9, 0.1, 0])); assert i == 0 and sim > 0.99
    assert db.best_match(np.zeros(3)) == (0.0, None)
