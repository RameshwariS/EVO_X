"""
EVO-X Phase 4 -- consumer / scorer (Module 4)
=============================================
``ModelHolder``   deployed model behind one reference, swapped atomically (Phase C3
                  promote/rollback will call ``swap``; readers never see a half-built model).
``StreamScorer``  consumes ``network-events``, scores with the deployed model, publishes
                  predictions + confidence to ``predictions``.

Scoring never sees labels: batches are decoded with ``with_labels=False``.
Idempotent on batch index: a re-delivered batch is skipped (counted), so
at-least-once delivery has exactly-once effect.  A *gap* (index jumps ahead) is an
error on a single ordered partition and is raised, never silently scored.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from evox.config import TOPIC_EVENTS, TOPIC_PREDICTIONS
from evox.models.ids import IDSModel
from evox.streaming.codec import decode_batch, decode_prediction, encode_prediction
from evox.streaming.transport import Transport


class StreamGapError(RuntimeError):
    pass


class ModelHolder:
    def __init__(self, model: IDSModel, version: str = "v1"):
        self._lock = threading.Lock()
        self._state: Tuple[IDSModel, str] = (model, version)

    def get(self) -> Tuple[IDSModel, str]:
        with self._lock:
            return self._state

    def swap(self, model: IDSModel, version: str) -> Tuple[IDSModel, str]:
        with self._lock:
            old, self._state = self._state, (model, version)
            return old


class StreamScorer:
    def __init__(self, transport: Transport, holder: ModelHolder,
                 in_topic: str = TOPIC_EVENTS, out_topic: str = TOPIC_PREDICTIONS):
        self.transport, self.holder = transport, holder
        self.in_topic, self.out_topic = in_topic, out_topic
        self.next_index = 0
        self.n_scored = self.n_duplicates = 0
        self.score_seconds: List[float] = []        # pure model time per batch

    def step(self, timeout: float = 0.0) -> Optional[int]:
        """Process at most one message.  Returns the scored batch index, or None."""
        msg = self.transport.poll(self.in_topic, timeout)
        if msg is None:
            return None
        b = decode_batch(msg.value, with_labels=False)
        if b.index < self.next_index:               # duplicate (at-least-once redelivery)
            self.n_duplicates += 1
            self.transport.commit(msg)
            return None
        if b.index > self.next_index:
            raise StreamGapError(f"expected batch {self.next_index}, got {b.index}")
        model, version = self.holder.get()
        t0 = time.perf_counter()
        pred, conf = model.predict_with_confidence(b.X)
        self.score_seconds.append(time.perf_counter() - t0)
        self.transport.send(self.out_topic,
                            encode_prediction(b.index, pred, conf, b.produced_ns, model_version=version), key=b.index)
        self.next_index += 1
        self.n_scored += 1
        self.transport.commit(msg)                  # commit AFTER the prediction is published
        return b.index

    def run(self, n_batches: int, idle_timeout: float = 30.0):
        last = time.perf_counter()
        while self.next_index < n_batches:
            if self.step(timeout=0.2) is not None:
                last = time.perf_counter()
            elif time.perf_counter() - last > idle_timeout:
                raise TimeoutError(f"scorer idle for {idle_timeout}s at batch {self.next_index}/{n_batches}")


class PredictionCollector:
    """Reassembles the ``predictions`` topic into per-flow arrays; de-duplicates on index."""

    def __init__(self, transport: Transport, topic: str = TOPIC_PREDICTIONS):
        self.transport, self.topic = transport, topic
        self.by_index = {}
        self.e2e_ns: dict = {}                      # index -> receive_ns - produced_ns
        self.scored_ns: dict = {}                   # index -> scored_ns - produced_ns
        self.versions: dict = {}

    def step(self, timeout: float = 0.0) -> bool:
        m = self.transport.poll(self.topic, timeout)
        if m is None:
            return False
        hdr, yp, cf = decode_prediction(m.value)
        i = hdr["index"]
        if i not in self.by_index:
            self.by_index[i] = (yp, cf)
            self.e2e_ns[i] = time.time_ns() - hdr["produced_ns"]
            self.scored_ns[i] = hdr["scored_ns"] - hdr["produced_ns"]
            self.versions[i] = hdr.get("model_version")
        self.transport.commit(m)
        return True

    def collect(self, n_batches: int, idle_timeout: float = 30.0):
        last = time.perf_counter()
        while len(self.by_index) < n_batches:
            if self.step(0.2):
                last = time.perf_counter()
            elif time.perf_counter() - last > idle_timeout:
                raise TimeoutError("collector idle")
        return self.arrays(n_batches)

    def arrays(self, n_batches: int):
        idx = range(n_batches)
        return (np.concatenate([self.by_index[i][0] for i in idx]),
                np.concatenate([self.by_index[i][1] for i in idx]))
