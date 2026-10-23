"""
EVO-X Phase 4 -- replay the stream through a producer/consumer pipeline (Synopsis §7.9 Phase 4)
==============================================================================================
    # no Docker needed (in-memory fallback transport; same code path, same wire format):
    python phase4_replay.py run --transport memory --synthetic --check-equivalence --bench

    # with Kafka:
    docker compose up -d
    python phase4_replay.py run --transport kafka --create-topics --check-equivalence --bench
    #   ... or as two processes (the "two commands" of the definition of done):
    python phase4_replay.py consume --transport kafka --create-topics     # terminal 1
    python phase4_replay.py produce --transport kafka --rate 50           # terminal 2

Acceptance test: predictions that came through the transport must be BIT-IDENTICAL to the
in-memory Phase-3 loop with the same model/seed.  Exit code 1 if they are not.
Outputs: results/metrics/phase4_latency.csv  (the Latency term of the controller utility, §7.4)
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from phase3_static_ids import cap_per_class
from src.config import KAFKA_BOOTSTRAP, METRICS_DIR, RANDOM_SEED, STREAM_DIR
from src.ids_model import IDSModel
from src.stream_builder import EvoStream, build_stream
from src.stream_consumer import ModelHolder, PredictionCollector, StreamScorer
from src.stream_pipeline import replay_in_memory, run_pipeline
from src.stream_producer import StreamProducer
from src.stream_transport import InMemoryTransport, KafkaTransport
from src.synthetic_data import generate_synthetic_week

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("phase4")


def load_stream(a):
    if a.synthetic or not (STREAM_DIR / "stream.npz").exists():
        log.warning("SYNTHETIC stream in use -- validates the pipeline only; latency numbers are indicative, not results.")
        return build_stream(generate_synthetic_week(30_000), reference_fraction=0.5, batch_size=500), "SYNTHETIC"
    return EvoStream.load(a.stream_dir), "REAL"


def make_transport(a):
    if a.transport == "memory":
        return InMemoryTransport(duplicate_every=a.duplicate_every)
    t = KafkaTransport(a.bootstrap)
    if a.create_topics:
        t.create_topics()
    return t


def train(kind, s, seed, cap):
    X, y = cap_per_class(s.X_ref_train, s.y_ref_train, cap, seed)
    return IDSModel(kind, seed=seed, params=dict(n_estimators=a_est(kind))).fit(X, y)


def a_est(kind):  # smaller ensembles keep the smoke test fast; the controller profile uses the real configs
    return 100 if kind == "random_forest" else 150


def check_equivalence(s, model, res, bs) -> bool:
    yp, cf = replay_in_memory(s, model, bs)
    n = len(res.y_pred)
    same_p = np.array_equal(res.y_pred, yp[:n])
    same_c = np.array_equal(res.conf.view(np.uint32), cf[:n].view(np.uint32))   # bit-level on float32
    log.info("equivalence: predictions identical=%s  confidence bit-identical=%s  (%d flows)", same_p, same_c, n)
    return same_p and same_c


def cmd_run(a) -> int:
    s, tag = load_stream(a)
    bs = a.batch_size or int(s.meta["batch_size"])
    rows, ok = [], True
    for kind in a.models:
        model = train(kind, s, RANDOM_SEED, a.max_train_per_class)
        tr = make_transport(a)
        res = run_pipeline(s, model, tr, rate=a.rate, batch_size=bs, limit=a.limit)
        tr.close()
        if a.check_equivalence:
            ok &= check_equivalence(s, model, res, bs)
        row = dict(model=kind, transport=a.transport, batch_size=bs, data_tag=tag,
                   throughput_batches_per_s=res.summary()["batches_per_s"], model_ms_mean=res.summary()["model_ms_mean"],
                   duplicates_skipped=res.n_duplicates_skipped)
        if a.bench:
            # latency is only meaningful below saturation: an unpaced producer just builds a backlog and the
            # "latency" becomes queue time.  Re-run paced at 50% of the measured throughput.
            paced_rate = 0.5 * row["throughput_batches_per_s"]
            tr2 = make_transport(a)
            lat = run_pipeline(s, model, tr2, rate=paced_rate, batch_size=bs, limit=min(a.limit or 10**9, 60))
            tr2.close()
            ls = lat.summary()
            row.update(paced_rate=paced_rate, e2e_ms_mean=ls["e2e_ms_mean"], e2e_ms_p50=ls["e2e_ms_p50"], e2e_ms_p95=ls["e2e_ms_p95"],
                       e2e_ms_per_1k_flows=ls["e2e_ms_mean"] * 1000 / bs)
        rows.append(row)
        log.info("%s: %s", kind, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    if a.bench:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        out = METRICS_DIR / f"phase4_latency_{a.transport}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        log.info("wrote %s", out)
    return 0 if ok else 1


def cmd_produce(a) -> int:
    s, _ = load_stream(a)
    n = StreamProducer(make_transport(a)).publish(s, rate=a.rate, batch_size=a.batch_size, limit=a.limit)
    log.info("published %d batches", n)
    return 0


def cmd_consume(a) -> int:
    s, _ = load_stream(a)
    bs = a.batch_size or int(s.meta["batch_size"])
    n_batches = s.n_batches(bs) if a.limit is None else min(a.limit, s.n_batches(bs))
    model = train(a.models[0], s, RANDOM_SEED, a.max_train_per_class)
    tr = make_transport(a)
    scorer, coll = StreamScorer(tr, ModelHolder(model)), PredictionCollector(tr)
    scorer.run(n_batches, idle_timeout=120)
    yp, cf = coll.collect(n_batches)
    ok = check_equivalence(s, model, type("R", (), dict(y_pred=yp, conf=cf))(), bs) if a.check_equivalence else True
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="EVO-X Phase 4: streaming replay")
    ap.add_argument("cmd", choices=["run", "produce", "consume"])
    ap.add_argument("--transport", choices=["memory", "kafka"], default="memory")
    ap.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    ap.add_argument("--create-topics", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--stream-dir", default=str(STREAM_DIR))
    ap.add_argument("--models", nargs="+", default=["random_forest", "xgboost"])
    ap.add_argument("--rate", type=float, default=0.0, help="batches/s (0 = as fast as possible)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N batches")
    ap.add_argument("--max-train-per-class", type=int, default=20_000)
    ap.add_argument("--duplicate-every", type=int, default=0, help="(memory) redeliver every k-th message")
    ap.add_argument("--check-equivalence", action="store_true")
    ap.add_argument("--bench", action="store_true", help="write the latency table")
    a = ap.parse_args()
    return dict(run=cmd_run, produce=cmd_produce, consume=cmd_consume)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
