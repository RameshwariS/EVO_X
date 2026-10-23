"""Phase 5 tests: truth consistent with the stream, recurring really repeats, reproducible, pools disjoint."""
import json

import numpy as np
import pytest

from src.drift_scenarios import (KINDS, MAGNITUDES, LabSource, ScenarioSpec, batch_class_composition,
                                 generate_scenario, scenario_library)
from src.stream_builder import EvoStream, build_stream
from src.synthetic_data import generate_synthetic_week


@pytest.fixture(scope="module")
def source():
    s = build_stream(generate_synthetic_week(50_000, seed=2), reference_fraction=0.5, batch_size=500)
    return LabSource.from_evostream(s, seed=0)


def _gen(source, kind, mag="med", seed=1, fam="validation", **kw):
    return generate_scenario(source, ScenarioSpec(kind, mag, seed, fam, **kw))


def _share(comp, rows, ids):
    c = comp[rows].sum(axis=0).astype(float)
    return c[ids].sum() / c.sum()


def test_all_six_kinds_generate_with_complete_truth(source):
    for k in KINDS:
        sc = _gen(source, k)
        t = sc.truth
        assert sc.stream.X_stream.shape[0] == t["n_flows"] == t["n_batches"] * t["batch_size"]
        assert t["events"] and any(e["scored"] for e in t["events"])
        assert {"concepts", "segments", "affected_classes", "magnitude_detail", "shift_family"} <= set(t)
        assert len(t["per_batch"]["p_b"]) == t["n_batches"]
        json.dumps(t)                                              # serialisable
        assert isinstance(sc.stream, EvoStream) and sc.stream.n_batches() == t["n_batches"]


def test_reproducible_from_seed_and_different_across_seeds(source):
    a, b, c = _gen(source, "gradual", seed=5), _gen(source, "gradual", seed=5), _gen(source, "gradual", seed=6)
    assert np.array_equal(a.stream.X_stream, b.stream.X_stream) and np.array_equal(a.stream.y_stream, b.stream.y_stream)
    assert a.truth == b.truth
    assert not np.array_equal(a.stream.y_stream, c.stream.y_stream)


def test_no_drift_before_change_point(source):
    """Pre-change class mix == reference prior (clean ground truth): TV distance should be tiny."""
    sc = _gen(source, "abrupt", "high")
    comp = batch_class_composition(sc)
    pre = comp[:sc.truth["change_batch"]].sum(axis=0).astype(float); pre /= pre.sum()
    ref = np.bincount(sc.stream.y_ref_train.astype(int), minlength=comp.shape[1]).astype(float); ref /= ref.sum()
    assert 0.5 * np.abs(pre - ref).sum() < 0.03


def test_abrupt_switches_exactly_at_change_point(source):
    sc = _gen(source, "abrupt", "high")
    c, p = sc.truth["change_batch"], np.array(sc.truth["per_batch"]["p_b"])
    assert (p[:c] == 0).all() and (p[c:] == 1).all()
    comp = batch_class_composition(sc)
    aff = [sc.stream.classes.index(x) for x in sc.truth["affected_classes"] if x != "BENIGN"]
    pre, post = _share(comp, slice(0, c), aff), _share(comp, slice(c, None), aff)
    assert post > pre + 0.2                                       # attack mix really moved


def test_gradual_is_monotone_blend(source):
    sc = _gen(source, "gradual")
    p = np.array(sc.truth["per_batch"]["p_b"]); e = sc.truth["events"][0]
    assert (np.diff(p) >= -1e-9).all() and p[e["start_batch"] - 1] == 0 and p[e["end_batch"] + 1] == 1
    assert 0.2 < p[(e["start_batch"] + e["end_batch"]) // 2] < 0.8      # genuinely in between mid-transition


def test_temporary_reverts(source):
    sc = _gen(source, "temporary", "high")
    t = sc.truth; c, L = t["change_batch"], sc.spec.temporary_len
    p = np.array(t["per_batch"]["p_b"])
    assert (p[c:c + L] == 1).all() and (p[:c] == 0).all() and (p[c + L:] == 0).all()
    rev = [e for e in t["events"] if e["type"] == "revert"][0]
    assert rev["start_batch"] == c + L and not rev["scored"]
    # data after revert looks like data before (feature means), unlike the shifted segment
    X = sc.stream.X_stream; bs = t["batch_size"]; f = [sc.stream.feature_names.index(n) for n in t["affected_features"]]
    m = lambda a, b: X[a * bs:b * bs][:, f].mean(axis=0)
    assert np.abs(m(c, c + L) - m(0, c)).max() > 1.0 and np.abs(m(c + L + 2, c + L + 30) - m(0, c)).max() < 0.5


def test_recurring_really_repeats(source):
    sc = _gen(source, "recurring", "high")
    t = sc.truth; bs = t["batch_size"]
    segs = t["segments"]; assert [s["concept"] for s in segs] == ["A", "B", "A", "B"]
    f = [sc.stream.feature_names.index(n) for n in t["affected_features"]]
    X = sc.stream.X_stream
    mean = lambda s: X[s["start_batch"] * bs:(s["end_batch"] + 1) * bs][:, f].mean(axis=0)
    A1, B1, A2, B2 = (mean(s) for s in segs)
    d = lambda u, v: np.abs(u - v).max()
    assert d(A1, A2) < 0.3 and d(B1, B2) < 0.3 and d(A1, B1) > 1.0       # A~A, B~B, A!~B
    onsets = [e for e in t["events"] if e["type"] == "onset"]
    assert len(onsets) == 2 and onsets[1].get("recurrence") is True


def test_incremental_shift_grows_cumulatively(source):
    sc = _gen(source, "incremental", "high")
    t = sc.truth; bs = t["batch_size"]; c, w = t["change_batch"], sc.spec.incremental_width
    f = [sc.stream.feature_names.index(n) for n in t["affected_features"]]
    sgn = np.sign(np.array([0.0] * len(f)) + 1)                        # magnitude only
    dev = lambda a, b: np.abs(sc.stream.X_stream[a * bs:b * bs][:, f].mean(axis=0) - sc.stream.X_ref_train[:, f].mean(axis=0)).mean()
    d0, d1, d2 = dev(0, c), dev(c + w // 4, c + w // 2), dev(c + w, c + w + 10)
    assert d0 < d1 < d2
    assert t["slope_sigma_per_batch"] == pytest.approx(t["magnitude_detail"]["offset_sigma"] / w)


def test_novel_class_absent_from_reference_and_first_appears_at_truth(source):
    sc = _gen(source, "novel_class", "high")
    t = sc.truth; nid = t["novel_class_ids"][0]
    assert nid not in set(sc.stream.y_ref_train) | set(sc.stream.y_ref_val)
    assert nid in sc.stream.novel_mask().nonzero()[0].tolist() or sc.stream.novel_mask().any()
    comp = batch_class_composition(sc)
    first = int(np.flatnonzero(comp[:, nid] > 0)[0])
    assert first == t["first_batch_of_novel_class"] == t["change_batch"]
    assert nid not in t["known_class_ids"]


def test_offset_is_bounded_and_only_on_declared_features(source):
    sc = _gen(source, "abrupt", "high")
    sig = sc.truth["magnitude_detail"]["offset_sigma"]
    assert sig == 3.0 and len(sc.truth["affected_features"]) == sc.truth["magnitude_detail"]["n_features_shifted"] >= 3


def test_magnitude_levels_are_ordered(source):
    tv = [_gen(source, "abrupt", m).truth["magnitude_detail"]["mix_tv_distance"] for m in MAGNITUDES]
    sg = [_gen(source, "abrupt", m).truth["magnitude_detail"]["offset_sigma"] for m in MAGNITUDES]
    assert tv[0] < tv[1] < tv[2] and sg[0] < sg[1] < sg[2]


def test_validation_and_test_pools_are_disjoint(source):
    for c in source.pools["validation"]:
        v = {r.tobytes() for r in source.pools["validation"][c]}
        t = {r.tobytes() for r in source.pools["test"][c]}
        # identical flows can exist in the raw data (duplicates are removed upstream, so this must be empty)
        assert not (v & t)


def test_train_is_disjoint_from_pools(source):
    tr = {r.tobytes() for r in source.X_train}
    for fam in ("validation", "test"):
        for c, X in source.pools[fam].items():
            assert not (tr & {r.tobytes() for r in X})


def test_spec_validation_rejects_too_short_streams():
    with pytest.raises(ValueError):
        ScenarioSpec("recurring", n_batches=60).validate()
    with pytest.raises(ValueError):
        ScenarioSpec("nope").validate()


def test_library_keeps_families_apart():
    lib = scenario_library((1, 2), (10, 11, 12))
    assert len(lib) == 6 * 3 * (2 + 3)
    assert not ({s.seed for s in lib if s.family == "validation"} & {s.seed for s in lib if s.family == "test"})


def test_save_writes_truth_and_loadable_stream(source, tmp_path):
    sc = _gen(source, "abrupt")
    sc.save(tmp_path)
    assert json.loads((tmp_path / "drift_truth.json").read_text())["kind"] == "abrupt"
    s2 = EvoStream.load(tmp_path)
    assert np.array_equal(s2.X_stream, sc.stream.X_stream)


def test_stationary_control_has_no_events_and_no_shift(source):
    sc = generate_scenario(source, ScenarioSpec("stationary", "low", 3, "validation", n_batches=100))
    assert sc.truth["events"] == [] and not any(sc.truth["per_batch"]["p_b"])
    X = sc.stream.X_stream; bs = sc.spec.batch_size
    assert np.abs(X[:50 * bs].mean(0) - X[50 * bs:].mean(0)).max() < 0.15


def test_library_with_controls():
    lib = scenario_library((1,), (2,), controls_per_family=1, control_batches=200)
    assert sum(s.kind == "stationary" for s in lib) == 2 and len(lib) == 2 + 6 * 3 * 2


def test_concept_choice_is_independent_of_magnitude(source):
    """low/med/high must be the SAME drift at different strengths (same classes, same features, same signs)."""
    t = [_gen(source, "abrupt", m).truth for m in MAGNITUDES]
    assert t[0]["affected_features"] == t[1]["affected_features"] == t[2]["affected_features"]
    assert t[0]["affected_classes"] == t[1]["affected_classes"] == t[2]["affected_classes"]
    n = [_gen(source, "novel_class", m).truth["novel_classes"] for m in MAGNITUDES]
    assert n[0] == n[1] == n[2]
