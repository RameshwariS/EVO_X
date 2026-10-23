"""
EVO-X Phase 4 -- wire format for replay batches (Synopsis §7.3 Module 4)
=======================================================================
One Kafka message = one replay batch:

    [4B magic 'EVX1'][4B header_len][header JSON][4B X_len][X blob][4B y_len][y blob]

* X blob  : zlib-compressed little-endian float32, C-order  -> bit-exact round trip
* y blob  : zlib-compressed little-endian int16, in its OWN field.  A consumer that
            must not see labels (label-delay policy, Phase B) calls
            ``decode_batch(msg, with_labels=False)`` and the label bytes are never
            even decompressed.
* header  : batch index, row range, shape, stream timestamps, producer wall-clock
            (ns) for end-to-end latency measurement.

No pickle anywhere (safe to consume from an untrusted topic).
"""
from __future__ import annotations

import json
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

MAGIC = b"EVX1"


@dataclass
class DecodedBatch:
    index: int
    X: np.ndarray
    y: Optional[np.ndarray]          # None when with_labels=False
    row_start: int
    row_end: int
    t_start: Optional[str]
    t_end: Optional[str]
    produced_ns: int
    header: Dict


def encode_batch(index: int, X: np.ndarray, y: np.ndarray, row_start: int, row_end: int,
                 t_start=None, t_end=None, produced_ns: Optional[int] = None, level: int = 1) -> bytes:
    X = np.ascontiguousarray(X, dtype="<f4")
    y = np.ascontiguousarray(y, dtype="<i2")
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D and match y in length")
    hdr = dict(index=int(index), n=int(X.shape[0]), d=int(X.shape[1]), row_start=int(row_start),
               row_end=int(row_end), t_start=None if t_start is None else str(t_start),
               t_end=None if t_end is None else str(t_end),
               produced_ns=int(time.time_ns() if produced_ns is None else produced_ns),
               x_dtype="<f4", y_dtype="<i2")
    h = json.dumps(hdr).encode()
    xb, yb = zlib.compress(X.tobytes(), level), zlib.compress(y.tobytes(), level)
    return b"".join([MAGIC, struct.pack("<I", len(h)), h, struct.pack("<I", len(xb)), xb,
                     struct.pack("<I", len(yb)), yb])


def decode_batch(msg: bytes, with_labels: bool = True) -> DecodedBatch:
    if msg[:4] != MAGIC:
        raise ValueError("bad magic: not an EVO-X batch message")
    o = 4
    (hl,) = struct.unpack_from("<I", msg, o); o += 4
    hdr = json.loads(msg[o:o + hl]); o += hl
    (xl,) = struct.unpack_from("<I", msg, o); o += 4
    xb = msg[o:o + xl]; o += xl
    (yl,) = struct.unpack_from("<I", msg, o); o += 4
    yb = msg[o:o + yl]
    X = np.frombuffer(zlib.decompress(xb), dtype="<f4").reshape(hdr["n"], hdr["d"]).astype(np.float32)
    y = np.frombuffer(zlib.decompress(yb), dtype="<i2").astype(np.int16) if with_labels else None
    return DecodedBatch(hdr["index"], X, y, hdr["row_start"], hdr["row_end"], hdr["t_start"], hdr["t_end"],
                        hdr["produced_ns"], hdr)


def encode_prediction(index: int, y_pred: np.ndarray, conf: np.ndarray, produced_ns: int,
                      scored_ns: Optional[int] = None, model_version: str = "v1") -> bytes:
    """Prediction message.  ``produced_ns`` is copied from the input batch so the
    end-to-end latency can be read from the prediction alone."""
    yp = np.ascontiguousarray(y_pred, dtype="<i2")
    cf = np.ascontiguousarray(conf, dtype="<f4")
    hdr = dict(index=int(index), n=int(len(yp)), produced_ns=int(produced_ns),
               scored_ns=int(time.time_ns() if scored_ns is None else scored_ns), model_version=model_version)
    h = json.dumps(hdr).encode()
    return b"".join([MAGIC, struct.pack("<I", len(h)), h, yp.tobytes(), cf.tobytes()])


def decode_prediction(msg: bytes):
    if msg[:4] != MAGIC:
        raise ValueError("bad magic")
    (hl,) = struct.unpack_from("<I", msg, 4)
    hdr = json.loads(msg[8:8 + hl])
    n, o = hdr["n"], 8 + hl
    y_pred = np.frombuffer(msg, dtype="<i2", count=n, offset=o).astype(np.int16)
    conf = np.frombuffer(msg, dtype="<f4", count=n, offset=o + 2 * n).astype(np.float32)
    return hdr, y_pred, conf
