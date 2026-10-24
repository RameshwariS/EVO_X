"""
EVO-X Phase 4 -- message transports (Synopsis §7.3 Module 4)
============================================================
``Transport`` is the only thing the producer/consumer know about.  Two
implementations share the interface:

* :class:`InMemoryTransport`  -- in-process topics.  This is the *fallback* that
  guarantees later phases never block on Docker/Kafka, and it can inject
  at-least-once duplicates so idempotency is tested without a broker.
* :class:`KafkaTransport`     -- confluent-kafka.  Single partition for the data
  topic, ``acks=all`` + idempotent producer, consumer with manual commit AFTER
  processing  =>  at-least-once delivery.

Delivery semantics (document in the report):
    at-least-once on the wire, exactly-once *effect* because the scorer is
    idempotent on batch index (see stream_consumer.StreamScorer).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from evox.config import (KAFKA_BOOTSTRAP, TOPIC_DRIFT, TOPIC_EVENTS, TOPIC_EVOLUTION, TOPIC_MODEL,
                        TOPIC_PREDICTIONS)

ALL_TOPICS = [TOPIC_EVENTS, TOPIC_PREDICTIONS, TOPIC_DRIFT, TOPIC_EVOLUTION, TOPIC_MODEL]


@dataclass
class Message:
    topic: str
    offset: int
    key: Optional[int]
    value: bytes
    ts_ns: int


class Transport:
    def send(self, topic: str, value: bytes, key: Optional[int] = None) -> None: raise NotImplementedError
    def poll(self, topic: str, timeout: float = 0.0) -> Optional[Message]: raise NotImplementedError
    def commit(self, msg: Message) -> None: pass
    def flush(self) -> None: pass
    def close(self) -> None: pass


class InMemoryTransport(Transport):
    """Thread-safe in-process topics.

    ``duplicate_every=k`` re-delivers every k-th message once more (simulating a
    producer retry / consumer restart after a crash) so idempotency can be tested.
    """

    def __init__(self, duplicate_every: int = 0):
        self._q: Dict[str, Deque[Message]] = defaultdict(deque)
        self._offset: Dict[str, int] = defaultdict(int)
        self._cv = threading.Condition()
        self.duplicate_every = duplicate_every
        self.sent: Dict[str, int] = defaultdict(int)

    def send(self, topic, value, key=None):
        with self._cv:
            m = Message(topic, self._offset[topic], key, bytes(value), time.time_ns())
            self._offset[topic] += 1
            self._q[topic].append(m)
            self.sent[topic] += 1
            if self.duplicate_every and self.sent[topic] % self.duplicate_every == 0:
                self._q[topic].append(m)            # same offset/key/value delivered twice
            self._cv.notify_all()

    def poll(self, topic, timeout=0.0):
        with self._cv:
            if not self._q[topic] and timeout > 0:
                self._cv.wait_for(lambda: bool(self._q[topic]), timeout)
            return self._q[topic].popleft() if self._q[topic] else None


class KafkaTransport(Transport):
    """confluent-kafka transport.  Not exercised by the unit tests that need a broker
    (see tests/test_kafka_integration.py, marked ``integration``); clients can be
    injected so the glue logic is unit-tested with fakes."""

    def __init__(self, bootstrap: str = KAFKA_BOOTSTRAP, group_id: str = "evox-scorer",
                 producer=None, consumer_factory=None, max_message_bytes: int = 10 * 1024 * 1024):
        self.bootstrap, self.group_id, self.max_message_bytes = bootstrap, group_id, max_message_bytes
        self._producer = producer
        self._consumer_factory = consumer_factory
        self._consumers: Dict[str, object] = {}

    # -- clients ----------------------------------------------------------
    def _p(self):
        if self._producer is None:
            from confluent_kafka import Producer
            self._producer = Producer({
                "bootstrap.servers": self.bootstrap,
                "enable.idempotence": True,          # no duplicates from producer retries
                "acks": "all",
                "max.in.flight.requests.per.connection": 1,   # keep strict ordering even on retry
                "linger.ms": 0,
                "message.max.bytes": self.max_message_bytes,
            })
        return self._producer

    def _c(self, topic):
        if topic not in self._consumers:
            if self._consumer_factory is not None:
                c = self._consumer_factory(topic)
            else:
                from confluent_kafka import Consumer
                c = Consumer({
                    "bootstrap.servers": self.bootstrap,
                    "group.id": f"{self.group_id}-{topic}",
                    "auto.offset.reset": "earliest",
                    "enable.auto.commit": False,      # commit only AFTER processing => at-least-once
                    "fetch.message.max.bytes": self.max_message_bytes,
                    "max.partition.fetch.bytes": self.max_message_bytes,
                })
                c.subscribe([topic])
            self._consumers[topic] = c
        return self._consumers[topic]

    # -- Transport --------------------------------------------------------
    def create_topics(self, partitions_events: int = 1):
        from confluent_kafka.admin import AdminClient, NewTopic
        adm = AdminClient({"bootstrap.servers": self.bootstrap})
        cfg = {"max.message.bytes": str(self.max_message_bytes), "retention.ms": str(24 * 3600 * 1000)}
        # ONE partition everywhere: ordering of a time-ordered stream is the point
        futs = adm.create_topics([NewTopic(t, num_partitions=partitions_events, replication_factor=1, config=cfg)
                                  for t in ALL_TOPICS])
        for t, f in futs.items():
            try:
                f.result()
            except Exception as e:           # already exists is fine
                if "TOPIC_ALREADY_EXISTS" not in str(e):
                    raise

    def send(self, topic, value, key=None):
        k = None if key is None else str(key).encode()
        p = self._p()
        while True:
            try:
                p.produce(topic, value=value, key=k)
                break
            except BufferError:               # local queue full -> serve callbacks and retry
                p.poll(0.1)
        p.poll(0)

    def poll(self, topic, timeout=0.0):
        m = self._c(topic).poll(timeout)
        if m is None:
            return None
        if m.error():
            raise RuntimeError(f"kafka error: {m.error()}")
        key = None if m.key() is None else int(m.key())
        return Message(topic, m.offset(), key, m.value(), time.time_ns())

    def commit(self, msg):
        c = self._c(msg.topic)
        # commit the *raw* message position; fakes may not implement commit
        if hasattr(c, "commit_offset"):
            c.commit_offset(msg)
        else:
            from confluent_kafka import TopicPartition
            c.commit(offsets=[TopicPartition(msg.topic, 0, msg.offset + 1)], asynchronous=False)

    def flush(self):
        self._p().flush()

    def close(self):
        for c in self._consumers.values():
            c.close()
