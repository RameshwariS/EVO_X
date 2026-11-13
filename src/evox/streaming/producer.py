"""EVO-X Phase 4 -- stream producer (Module 4).  Publishes each replay batch of an
``EvoStream`` (real, or a Phase-5 scenario, which is EvoStream-compatible) to
``network-events`` as ONE message.  Labels travel in their own field."""
from __future__ import annotations

import time
from typing import Optional

from evox.config import TOPIC_EVENTS
from evox.data.stream import EvoStream
from evox.streaming.codec import encode_batch
from evox.streaming.transport import Transport


class StreamProducer:
    def __init__(self, transport: Transport, topic: str = TOPIC_EVENTS):
        self.transport, self.topic = transport, topic

    def publish(self, stream: EvoStream, rate: float = 0.0, batch_size: Optional[int] = None,
                limit: Optional[int] = None) -> int:
        """rate = batches/second (0 = as fast as possible).  Returns #batches sent."""
        period = 1.0 / rate if rate and rate > 0 else 0.0
        t_next, n = time.perf_counter(), 0
        for b in stream.iter_batches(batch_size):
            if limit is not None and n >= limit:
                break
            if period:
                time.sleep(max(0.0, t_next - time.perf_counter()))
                t_next += period
            self.transport.send(self.topic, encode_batch(b.index, b.X, b.y, b.row_start, b.row_end, b.t_start, b.t_end),
                                key=b.index)
            n += 1
        self.transport.flush()
        return n
