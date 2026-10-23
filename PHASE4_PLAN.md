# Phase 4 — Kafka-based streaming simulation (Synopsis §7.9 Phase 4, §7.3 Module 4)

**Goal:** replace the in-memory `iter_batches()` loop with a producer/consumer pipeline, so the system is a
decoupled stream processor (as in Figure 7.1), without changing any experimental numbers.

## Tasks
1. **Local broker.** `docker-compose.yml` with a single-node Kafka (KRaft mode, no ZooKeeper) and nothing else.
   Topics from the synopsis: `network-events`, `predictions`, `drift-events`, `evolution-events`, `model-events`
   (create the first two now; the rest are stubs until Phases 6–10). Library: `confluent-kafka` (or `kafka-python`).
2. **Producer** (`src/stream_producer.py`): reads `EvoStream`, publishes each replay batch to `network-events` as one
   message (compressed float32 bytes + metadata: batch index, row range, timestamps; **labels go in a separate
   field so the consumer can hold them back** — in deployment labels arrive late or never; this matters for Phase 7).
   Options: `--rate` (batches/s, 0 = as fast as possible) and a `--seed` for scenario replay later.
3. **Consumer / scorer** (`src/stream_consumer.py`): subscribes, scores with the deployed `IDSModel`, publishes
   predictions + confidence to `predictions`. Keep the deployed model behind a tiny `ModelHolder` so Phase 10
   can swap it atomically.
4. **Equivalence test (the key acceptance test):** predictions through Kafka must be bit-identical to the in-memory
   Phase 3 run (same seed). If not identical, find out why before building on it (ordering, batching, dtype casts).
5. **Ordering and delivery:** one partition for `network-events` (ordering is the point of a time-ordered stream);
   document at-least-once semantics and make the consumer idempotent on batch index.
6. **Throughput/latency measurement:** batches/s and end-to-end latency per batch for RF and XGBoost. These feed
   the Latency term in the Evolution Controller utility (§7.4).
7. **Tests:** producer/consumer round-trip against a Docker-launched broker (mark as `integration`, skippable in CI);
   unit tests for the (de)serialisation with no broker.
8. **Report:** architecture figure update (Stream Simulator box), deployment note, equivalence result.

## Definition of done
`docker compose up` + two commands replay the stream through Kafka; equivalence test passes; latency table recorded.

## Then
Phase 5: synthetic drift generator (6 scenario types) on top of the same `EvoStream` — it is where recurring,
gradual, incremental and temporary drift of *known* classes come from, since the natural CICIDS2017 stream is
mostly novel-class drift (see Phase 2 findings).
