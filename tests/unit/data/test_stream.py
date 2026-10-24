"""Phase 2 tests: ordering, leakage, real-data quirks, persistence."""
import numpy as np
import pandas as pd
import pytest

from evox.data.loading import load_raw_data
from evox.data.stream import (EvoStream, build_stream, normalize_label, parse_cicids_timestamps)
from evox.evaluation.stream import assert_valid, class_timeline, validate_stream
from evox.data.synthetic import generate_synthetic_week, write_synthetic_week_csvs


@pytest.fixture(scope="module")
def stream():
    return build_stream(generate_synthetic_week(20_000, seed=1), batch_size=500)


def test_validation_passes(stream):
    assert_valid(stream)


def test_pm_quirk_restores_true_order():
    # "1:30" is really 13:30; naive sort would put it before 9:00
    s = pd.Series(["3/7/2017 9:00", "3/7/2017 11:59", "3/7/2017 1:30", "3/7/2017 4:45"])
    ts = parse_cicids_timestamps(s)
    assert ts.is_monotonic_increasing
    assert ts.iloc[2].hour == 13 and ts.iloc[0].day == 3 and ts.iloc[0].month == 7


def test_iso_timestamps_untouched():
    s = pd.Series(pd.to_datetime(["2017-07-03 14:00", "2017-07-03 15:00"]))
    assert parse_cicids_timestamps(s).equals(s)


def test_label_normalisation():
    assert normalize_label("Web Attack \x96 Brute Force") == "Web Attack - Brute Force"
    assert normalize_label("FTP-Patator") == "FTP-Patator"


def test_stream_is_time_contiguous_and_unshuffled(stream):
    assert stream.meta["ordering_source"] == "timestamp"
    assert stream.t_ref.max() <= stream.t_stream.min()
    assert (np.diff(stream.t_stream.astype("int64")) >= 0).all()


def test_scaler_fit_on_reference_only(stream):
    # live stream must NOT be standardised to mean 0 (that would mean leakage)
    assert np.abs(stream.X_stream.mean(axis=0)).max() > 0.05


def test_novel_classes_found_in_stream(stream):
    tl = class_timeline(stream).set_index("label")
    # Friday-only attacks cannot be in a 30% (Mon-Tue) reference window
    assert tl.loc["DDoS", "status"].startswith("NOVEL")
    assert tl.loc["BENIGN", "status"] == "known"
    assert stream.novel_mask().any()


def test_batches_cover_stream_exactly(stream):
    batches = list(stream.iter_batches())
    assert sum(len(b) for b in batches) == len(stream.y_stream)
    assert len(batches) == stream.n_batches()
    assert batches[0].t_start <= batches[-1].t_end


def test_save_load_roundtrip(stream, tmp_path):
    stream.save(tmp_path)
    s2 = EvoStream.load(tmp_path)
    np.testing.assert_array_equal(stream.X_stream, s2.X_stream)
    np.testing.assert_array_equal(stream.y_ref_val, s2.y_ref_val)
    assert s2.classes == stream.classes and s2.feature_names == stream.feature_names
    assert s2.t_stream is not None
    assert_valid(s2)


def test_file_order_fallback_without_timestamp(tmp_path):
    """MachineLearningCVE-style files: no Timestamp. Alphabetical order would
    start with Friday; we must start with Monday."""
    write_synthetic_week_csvs(tmp_path, n_rows=12_000, with_timestamp=False)
    raw = load_raw_data(raw_dir=tmp_path, add_provenance=True)
    s = build_stream(raw)
    assert s.meta["ordering_source"] == "file_order"
    # Monday is benign-only, so the very first reference flows must be BENIGN
    assert set(np.unique(s.y_ref_train[:200])) == {0}
    # Friday attacks must land in the stream, not the reference
    tl = class_timeline(s).set_index("label")
    assert tl.loc["DDoS", "ref_flows"] == 0


def test_timestamped_csvs_with_leading_space_headers(tmp_path):
    write_synthetic_week_csvs(tmp_path, n_rows=12_000, with_timestamp=True)
    s = build_stream(load_raw_data(raw_dir=tmp_path, add_provenance=True))
    assert s.meta["ordering_source"] == "timestamp"
    assert_valid(s)


# ---------------- Part B additions ----------------
from evox.data.adapters import (UNSW_RAW_COLUMNS, adapt_cse2018_frame, adapt_unsw_nb15_frame,
                                  normalize_unsw_attack_cat)
from evox.data.stream import prepare_ordered


def _fake_unsw(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.random((n, len(UNSW_RAW_COLUMNS))), columns=UNSW_RAW_COLUMNS)
    d["srcip"], d["dstip"] = "10.0.0.1", "10.0.0.2"
    d["proto"], d["state"], d["service"] = "tcp", "FIN", "-"
    d["Stime"] = 1421927414 + np.sort(rng.integers(0, 80_000, n))
    cats = np.array(["", "Fuzzers ", "Backdoors", "Exploits", "Generic"])
    d["attack_cat"] = cats[np.minimum(4, (d.Stime - d.Stime.min()) // 16_500)]  # classes appear over time
    d["Label"] = (d.attack_cat != "").astype(int)
    return d


def test_unsw_label_normalisation():
    assert normalize_unsw_attack_cat(np.nan) == "BENIGN"
    assert normalize_unsw_attack_cat("Fuzzers ") == "Fuzzers"
    assert normalize_unsw_attack_cat("Backdoors") == "Backdoor"


def test_unsw_adapter_builds_valid_stream():
    s = build_stream(adapt_unsw_nb15_frame(_fake_unsw()), dataset_name="UNSW-NB15")
    assert s.meta["ordering_source"] == "timestamp"
    assert_valid(s)
    assert "Backdoor" in s.classes and "proto" not in s.feature_names


def test_cse2018_adapter_handles_repeated_header_and_case():
    d = generate_synthetic_week(3000, timestamp_style="iso")
    d["Timestamp"] = d["Timestamp"].dt.strftime("%d/%m/%Y %H:%M:%S")
    d["Label"] = d["Label"].replace({"BENIGN": "Benign"})
    hdr = pd.DataFrame([{c: c for c in d.columns}])
    n0 = len(d)
    d = pd.concat([d.iloc[:100], hdr, d.iloc[100:]], ignore_index=True)
    out = adapt_cse2018_frame(d)
    assert len(out) == n0 and out["Timestamp"].notna().all()
    s = build_stream(out)
    assert s.classes[0] == "BENIGN"
    assert_valid(s)


def test_reference_sweep_known_classes_grow():
    import importlib
    sweep = importlib.import_module("evox.tools.reference_sweep").sweep
    df, _ = prepare_ordered(generate_synthetic_week(20_000, seed=2))
    t = sweep(df, [0.2, 0.5, 0.8])
    assert (t.attack_classes_in_ref.diff().dropna() >= 0).all()
    # NB: novel share of stream attacks is NOT monotone (the denominator shrinks as
    # the reference grows), so only the known-class count is asserted.
