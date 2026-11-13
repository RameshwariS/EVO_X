"""Phase B data contracts: stable sample IDs, delayed labels and bounded memory."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from evox.config import LABEL_DELAY_BATCHES


def priority(sample_id: str, seed: int = 0) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{sample_id}".encode(), digest_size=8).digest(), "big")


@dataclass(frozen=True)
class Samples:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    batches: np.ndarray             # -1 means initially available reference labels

    def __post_init__(self):
        X, y = np.asarray(self.X, dtype=np.float32), np.asarray(self.y, dtype=np.int64)
        ids, batches = np.asarray(self.ids, dtype=str), np.asarray(self.batches, dtype=np.int64)
        if X.ndim != 2 or y.ndim != 1 or ids.ndim != 1 or batches.ndim != 1:
            raise ValueError("Samples need a feature matrix and one-dimensional metadata")
        if not (len(X) == len(y) == len(ids) == len(batches)):
            raise ValueError("Sample arrays have different lengths")
        if len(set(ids)) != len(ids) or (batches < -1).any() or not np.isfinite(X).all():
            raise ValueError("Sample IDs must be unique, batches >= -1, and features finite")
        for name, a in (("X", X), ("y", y), ("ids", ids), ("batches", batches)):
            a = a.copy()
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    def __len__(self):
        return len(self.y)

    def take(self, indices) -> "Samples":
        return Samples(self.X[indices], self.y[indices], self.ids[indices], self.batches[indices])

    def snapshot(self) -> str:
        h = hashlib.sha256()
        for a in (self.X, self.y, self.batches):
            h.update(str(a.shape).encode()); h.update(a.tobytes())
        for sid in self.ids:
            b = sid.encode(); h.update(len(b).to_bytes(8, "big")); h.update(b)
        return h.hexdigest()

    @classmethod
    def empty(cls, n_features: int):
        return cls(np.empty((0, n_features)), np.array([], int), np.array([], str), np.array([], int))


def combine(*parts: Samples) -> Samples:
    """Union by ID; duplicate IDs with conflicting contents are an error."""
    if not parts:
        raise ValueError("Need at least one sample set")
    X = np.concatenate([p.X for p in parts])
    y = np.concatenate([p.y for p in parts])
    ids = np.concatenate([p.ids for p in parts])
    batches = np.concatenate([p.batches for p in parts])
    seen, keep = {}, []
    for i, sid in enumerate(ids):
        if sid in seen:
            j = seen[sid]
            if y[i] != y[j] or batches[i] != batches[j] or not np.array_equal(X[i], X[j]):
                raise ValueError(f"Conflicting sample ID: {sid}")
        else:
            seen[sid] = i; keep.append(i)
    return Samples(X[keep], y[keep], ids[keep], batches[keep])


@dataclass(frozen=True)
class LabelPolicy:
    delay_batches: int = LABEL_DELAY_BATCHES
    labelled_fraction: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.delay_batches < 0 or not 0 <= self.labelled_fraction <= 1:
            raise ValueError("Label delay must be nonnegative; fraction must be in [0, 1]")

    def available(self, samples: Samples, now_batch: int) -> Samples:
        reference = samples.batches == -1
        ready = samples.batches + self.delay_batches <= now_batch
        budget = np.array([priority(s, self.seed) / 2**64 < self.labelled_fraction for s in samples.ids], dtype=bool)
        return samples.take(reference | (ready & budget))


def balanced_subset(samples: Samples, capacity: int, seed: int = 0) -> Samples:
    """Deterministic class-balanced priority sampling, with redistribution of spare slots."""
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    groups = [sorted(np.flatnonzero(samples.y == c), key=lambda i: priority(samples.ids[i], seed))
              for c in np.unique(samples.y)]
    keep, depth = [], 0
    while len(keep) < min(capacity, len(samples)):
        for group in groups:
            if depth < len(group) and len(keep) < capacity:
                keep.append(group[depth])
        depth += 1
    return samples.take(np.array(keep, dtype=int))


class ReplayBuffer:
    """Bounded class-balanced memory; only policy-released labels enter storage.

    Priority sampling is reproducible. Evicted examples are not retained elsewhere.
    On strategic adaptation, stale entries can be explicitly expired by batch age.
    """
    def __init__(self, capacity: int, n_features: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.seed = capacity, seed
        self.samples = Samples.empty(n_features)

    def add(self, samples: Samples, policy: LabelPolicy, now_batch: int):
        self.samples = balanced_subset(combine(self.samples, policy.available(samples, now_batch)),
                                       self.capacity, self.seed)

    def expire(self, min_batch: int):
        self.samples = self.samples.take((self.samples.batches == -1) | (self.samples.batches >= min_batch))


@dataclass(frozen=True)
class AdaptationData:
    recent: Samples
    memory: Samples
    now_batch: int
    parent_version: str
    excluded_ids: tuple[str, ...] = ()

    def released(self, policy: LabelPolicy) -> tuple[Samples, Samples]:
        recent, memory = policy.available(self.recent, self.now_batch), policy.available(self.memory, self.now_batch)
        excluded = set(self.excluded_ids)
        if excluded.intersection(recent.ids) or excluded.intersection(memory.ids):
            raise ValueError("Training data overlaps a held-out evaluation set")
        return recent, memory
