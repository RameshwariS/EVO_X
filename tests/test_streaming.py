"""Phase 4 tests: codec, transports, idempotency, ordering, equivalence.  No broker needed."""
import numpy as np
import pytest

from src.ids_model import IDSModel
from src.stream_builder import build_stream
from src.stream_codec import decode_batch, decode_prediction, encode_batch, encode_prediction
from src.stream_consumer import ModelHolder, PredictionCollector, StreamGapError, StreamScorer
from src.stream_pipeline import replay_in_memory, run_pipeline
from src.stream_producer import StreamProducer
from src.stream_transport import InMemoryTransport, KafkaTransport
from src.synthetic_data import generate_synthetic_week


@pytest.fixture(scope="module")
def stream():
    return build_stream(generate_synthetic_week(9_000, seed=5), reference_fraction=0.5, batch_size=300)


@pytest.fixture(scope="module")
def model(stream):
    return IDSModel("random_forest", seed=0, params=dict(n_estimators=15)).fit(stream.X_ref_train, stream.y_ref_train)


# -- codec ---------------------------------------------------------------
def test_codec_roundtrip_is_bit_exact():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((37, 11)).astype(np.float32)
    X[0, 0], X[1, 1] = np.float32(1e-38), np.float32(-3.4e38)           # extremes survive
    y = rng.integers(0, 12, 37).astype(np.int16)
    d = decode_batch(encode_batch(4, X, y, 100, 137, "2017-07-03 09:00", None))
    assert d.X.tobytes() == X.tobytes() and (d.y == y).all()
    assert (d.index, d.row_start, d.row_end, d.t_start, d.t_end) == (4, 100, 137, "2017-07-03 09:00", None)


def test_labels_can_be_withheld():
    X, y = np.zeros((5, 2), np.float32), np.arange(5, dtype=np.int16)
    d = decode_batch(encode_batch(0, X, y, 0, 5), with_labels=False)
    assert d.y is None and d.X.shape == (5, 2)


def test_codec_rejects_garbage_and_shape_mismatch():
    with pytest.raises(ValueError):
        decode_batch(b"NOPE" + b"\x00" * 20)
    with pytest.raises(ValueError):
        encode_batch(0, np.zeros((3, 2), np.float32), np.zeros(4, np.int16), 0, 3)


def test_prediction_codec_roundtrip():
    yp, cf = np.array([0, 3, 7], np.int16), np.array([0.5, 0.25, 1.0], np.float32)
    h, yp2, cf2 = decode_prediction(encode_prediction(9, yp, cf, produced_ns=123, model_version="v7"))
    assert h["index"] == 9 and h["model_version"] == "v7" and h["produced_ns"] == 123
    assert (yp2 == yp).all() and cf2.tobytes() == cf.tobytes()


# -- the key acceptance test --------------------------------------------
def test_pipeline_predictions_are_bit_identical_to_inmemory(stream, model):
    res = run_pipeline(stream, model, InMemoryTransport())
    yp, cf = replay_in_memory(stream, model)
    assert np.array_equal(res.y_pred, yp)
    assert res.conf.tobytes() == cf.tobytes()
    assert res.n_batches == stream.n_batches() and len(res.y_pred) == len(stream.y_stream)


def test_idempotent_under_at_least_once_redelivery(stream, model):
    tr = InMemoryTransport(duplicate_every=3)            # every 3rd batch delivered twice
    res = run_pipeline(stream, model, tr)
    yp, cf = replay_in_memory(stream, model)
    assert res.n_duplicates_skipped > 0
    assert np.array_equal(res.y_pred, yp) and res.conf.tobytes() == cf.tobytes()


def test_gap_is_an_error_not_silently_scored(stream, model):
    tr = InMemoryTransport()
    batches = list(stream.iter_batches())
    for b in (batches[0], batches[2]):                    # batch 1 is missing
        tr.send("network-events", encode_batch(b.index, b.X, b.y, b.row_start, b.row_end), key=b.index)
    sc = StreamScorer(tr, ModelHolder(model))
    assert sc.step() == 0
    with pytest.raises(StreamGapError):
        sc.step()


def test_scorer_never_receives_labels(stream, model, monkeypatch):
    import src.stream_consumer as sc_mod
    seen = []
    orig = sc_mod.decode_batch
    monkeypatch.setattr(sc_mod, "decode_batch", lambda m, with_labels=True: (seen.append(with_labels), orig(m, with_labels))[1])
    run_pipeline(stream, model, InMemoryTransport(), limit=3)
    assert seen and not any(seen)


def test_model_swap_is_atomic_and_versioned(stream, model):
    other = IDSModel("random_forest", seed=1, params=dict(n_estimators=5)).fit(stream.X_ref_train, stream.y_ref_train)
    tr, holder = InMemoryTransport(), ModelHolder(model, "v1")
    sc, col = StreamScorer(tr, holder), PredictionCollector(tr)
    batches = list(stream.iter_batches())[:4]
    for b in batches:
        tr.send("network-events", encode_batch(b.index, b.X, b.y, b.row_start, b.row_end), key=b.index)
    sc.step(); sc.step()
    old = holder.swap(other, "v2")
    assert old[1] == "v1"
    sc.step(); sc.step()
    while col.step():
        pass
    assert [col.versions[i] for i in range(4)] == ["v1", "v1", "v2", "v2"]


def test_producer_respects_limit_and_order(stream):
    tr = InMemoryTransport()
    assert StreamProducer(tr).publish(stream, limit=5) == 5
    got = [tr.poll("network-events").key for _ in range(5)]
    assert got == [0, 1, 2, 3, 4] and tr.poll("network-events") is None


def test_latency_fields_are_populated(stream, model):
    res = run_pipeline(stream, model, InMemoryTransport(), limit=4)
    s = res.summary()
    assert s["e2e_ms_mean"] > 0 and s["model_ms_mean"] > 0 and s["batches_per_s"] > 0
    assert (res.e2e_ms >= res.scored_ms - 1e-6).all()    # received after created


# -- Kafka glue with injected fakes (no broker) ---------------------------
class _FakeProducer:
    def __init__(self): self.sent = []; self.polls = 0
    def produce(self, topic, value=None, key=None): self.sent.append((topic, key, value))
    def poll(self, t): self.polls += 1
    def flush(self): pass


class _FakeMsg:
    def __init__(self, v, k, off): self._v, self._k, self._o = v, k, off
    def error(self): return None
    def value(self): return self._v
    def key(self): return self._k
    def offset(self): return self._o


class _FakeConsumer:
    def __init__(self, items): self.items = list(items); self.committed = []
    def poll(self, t): return self.items.pop(0) if self.items else None
    def commit_offset(self, msg): self.committed.append(msg.offset)
    def close(self): pass


def test_kafka_transport_glue_with_fakes():
    fp = _FakeProducer()
    kt = KafkaTransport(producer=fp, consumer_factory=lambda t: _FakeConsumer([_FakeMsg(b"abc", b"7", 0)]))
    kt.send("network-events", b"xyz", key=3)
    assert fp.sent == [("network-events", b"3", b"xyz")]
    m = kt.poll("network-events")
    assert (m.key, m.value, m.offset) == (7, b"abc", 0)
    kt.commit(m)
    assert kt._c("network-events").committed == [0]
