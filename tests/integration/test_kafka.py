"""Needs a broker:  docker compose up -d   then   pytest -m integration"""
import socket

import numpy as np
import pytest

from evox.config import KAFKA_BOOTSTRAP
from evox.models.ids import IDSModel
from evox.data.stream import build_stream
from evox.streaming.pipeline import replay_in_memory, run_pipeline
from evox.streaming.transport import KafkaTransport
from evox.data.synthetic import generate_synthetic_week


def _broker_up():
    host, port = KAFKA_BOOTSTRAP.split(":")
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
        return True
    except OSError:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _broker_up(), reason="no Kafka broker on localhost:9092 (docker compose up -d)")
def test_kafka_roundtrip_bit_identical():
    import uuid
    s = build_stream(generate_synthetic_week(9_000, seed=5), reference_fraction=0.5, batch_size=300)
    m = IDSModel("random_forest", seed=0, params=dict(n_estimators=15)).fit(s.X_ref_train, s.y_ref_train)
    kt = KafkaTransport(group_id=f"test-{uuid.uuid4().hex[:8]}")
    kt.create_topics()
    res = run_pipeline(s, m, kt)
    yp, cf = replay_in_memory(s, m)
    assert np.array_equal(res.y_pred, yp) and res.conf.tobytes() == cf.tobytes()
