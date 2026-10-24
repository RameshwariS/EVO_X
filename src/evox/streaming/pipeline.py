"""EVO-X Phase 4 -- wire producer -> scorer -> collector together and measure it.

``run_pipeline`` works with ANY Transport; with ``InMemoryTransport`` it is the
no-Docker fallback.  ``replay_in_memory`` is the reference the Kafka path must
match bit-for-bit (this is exactly the Phase 3 per-batch loop)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from evox.data.stream import EvoStream
from evox.streaming.consumer import ModelHolder, PredictionCollector, StreamScorer
from evox.streaming.producer import StreamProducer
from evox.streaming.transport import Transport


@dataclass
class PipelineResult:
    y_pred: np.ndarray
    conf: np.ndarray
    n_batches: int
    wall_seconds: float
    e2e_ms: np.ndarray               # per batch, producer timestamp -> prediction received
    scored_ms: np.ndarray            # per batch, producer timestamp -> prediction created
    model_ms: np.ndarray             # per batch, pure model time
    n_duplicates_skipped: int = 0

    def summary(self) -> dict:
        return dict(batches=self.n_batches, batches_per_s=self.n_batches / self.wall_seconds,
                    e2e_ms_mean=float(self.e2e_ms.mean()), e2e_ms_p50=float(np.percentile(self.e2e_ms, 50)),
                    e2e_ms_p95=float(np.percentile(self.e2e_ms, 95)),
                    model_ms_mean=float(self.model_ms.mean()), duplicates_skipped=self.n_duplicates_skipped)


def replay_in_memory(stream: EvoStream, model, batch_size: Optional[int] = None):
    """The Phase 3 loop: score each batch directly. Reference for the equivalence test."""
    yp = np.empty(len(stream.y_stream), dtype=np.int16)
    cf = np.empty(len(stream.y_stream), dtype=np.float32)
    for b in stream.iter_batches(batch_size):
        p, c = model.predict_with_confidence(b.X)
        yp[b.row_start:b.row_end], cf[b.row_start:b.row_end] = p, c
    return yp, cf


def run_pipeline(stream: EvoStream, model, transport: Transport, rate: float = 0.0,
                 batch_size: Optional[int] = None, limit: Optional[int] = None, version: str = "v1",
                 transport_consumer: Optional[Transport] = None) -> PipelineResult:
    """Producer, scorer and collector run as three threads over the transport."""
    n_batches = stream.n_batches(batch_size) if limit is None else min(limit, stream.n_batches(batch_size))
    holder = ModelHolder(model, version)
    scorer = StreamScorer(transport_consumer or transport, holder)
    collector = PredictionCollector(transport_consumer or transport)
    errors = []

    def _guard(fn):
        def w():
            try:
                fn()
            except BaseException as e:      # surface thread errors to the caller
                errors.append(e)
        return w

    t0 = time.perf_counter()
    th = [threading.Thread(target=_guard(lambda: StreamProducer(transport).publish(stream, rate, batch_size, limit))),
          threading.Thread(target=_guard(lambda: scorer.run(n_batches)))]
    for t in th:
        t.start()
    try:
        yp, cf = collector.collect(n_batches)
    finally:
        for t in th:
            t.join(timeout=60)
    wall = time.perf_counter() - t0
    if errors:
        raise errors[0]
    idx = range(n_batches)
    return PipelineResult(yp, cf, n_batches, wall,
                          np.array([collector.e2e_ns[i] for i in idx]) / 1e6,
                          np.array([collector.scored_ns[i] for i in idx]) / 1e6,
                          np.array(scorer.score_seconds) * 1e3, scorer.n_duplicates)
