"""Phase B candidate generation. Nothing in this module promotes a model."""
from __future__ import annotations

import copy
import hashlib
import pickle
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from evox.adaptation.data import AdaptationData, LabelPolicy, Samples, balanced_subset, combine
from evox.models.ids import IDSModel


class Action(str, Enum):
    NO_OP = "no_op"
    INCREMENTAL = "incremental"
    REPLAY = "replay"
    STRATEGIC = "strategic_selection"
    RETRAIN = "full_retrain"


@dataclass(frozen=True)
class StrategyConfig:
    recent_capacity: int = 2048
    memory_capacity: int = 2048
    selection_capacity: int = 1024
    stale_after_batches: int = 30
    retain_fraction: float = 0.5
    boosting_rounds: int = 20

    def __post_init__(self):
        if min(self.recent_capacity, self.memory_capacity, self.selection_capacity, self.stale_after_batches,
               self.boosting_rounds) < 1 or not 0 < self.retain_fraction < 1:
            raise ValueError("Invalid strategy capacities or update parameters")


@dataclass
class CandidateModel:
    candidate_id: str
    parent_version: str
    action: str
    model: IDSModel
    training_snapshot_id: str
    training_ids: list[str]
    profile: dict
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    status: str = "CANDIDATE"

    def record(self):
        return {k: v for k, v in self.__dict__.items() if k != "model"}


class AdaptationStrategy:
    action: Action

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()

    def training_data(self, base: IDSModel, recent: Samples, memory: Samples) -> Samples:
        raise NotImplementedError

    def apply(self, base_model: IDSModel, data: AdaptationData, label_policy: LabelPolicy) -> CandidateModel:
        recent, memory = data.released(label_policy)
        if self.action == Action.NO_OP:
            return CandidateModel(data.parent_version, data.parent_version, self.action.value, base_model,
                                  recent.take(np.array([], int)).snapshot(), [],
                                  dict(training_seconds=0.0, peak_python_bytes=0, model_bytes=0,
                                       latency_ms_per_1k=0.0, n_train=0),
                                  metadata={"reason": "parent unchanged"}, status="NO_OP")
        recent = balanced_subset(recent, self.config.recent_capacity, base_model.seed)
        memory = balanced_subset(memory, self.config.memory_capacity, base_model.seed)
        train = self.training_data(base_model, recent, memory)
        if not len(train):
            raise ValueError("No released labelled training samples")
        if self.action != Action.INCREMENTAL and len(np.unique(train.y)) < 2:
            raise ValueError("Training a fresh classifier requires at least two classes")
        # Include parent identity, action, config and decision time in the candidate identity.
        snapshot = train.snapshot()
        identity = (f"{data.parent_version}:{base_model.kind}:{base_model.seed}:{base_model.params}:"
                    f"{self.action.value}:{snapshot}:{asdict(self.config)}:{data.now_batch}")
        cid = "candidate-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        tracking = tracemalloc.is_tracing()
        if not tracking:
            tracemalloc.start()
        before, _ = tracemalloc.get_traced_memory()
        t0 = time.perf_counter()
        try:
            if self.action == Action.INCREMENTAL:
                model = copy.deepcopy(base_model)
                model.update(train.X, train.y, self.config.retain_fraction, self.config.boosting_rounds)
            else:
                model = base_model.clone().fit(train.X, train.y)
            seconds = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not tracking:
                tracemalloc.stop()
        probe = train.X[:min(256, len(train))]
        model.predict(probe)  # warm-up outside the measurement
        timings = []
        for _ in range(3):
            start = time.perf_counter(); model.predict(probe)
            timings.append((time.perf_counter() - start) * 1e6 / len(probe))
        profile = dict(training_seconds=seconds, peak_python_bytes=max(0, peak - before),
                       model_bytes=len(pickle.dumps(model)), latency_ms_per_1k=float(np.median(timings)),
                       n_train=len(train))
        return CandidateModel(cid, data.parent_version, self.action.value, model, snapshot, train.ids.tolist(), profile,
                              metadata=dict(now_batch=data.now_batch, label_policy=asdict(label_policy),
                                            strategy_config=asdict(self.config), algorithm=base_model.kind,
                                            memory_measure="tracemalloc Python peak; excludes native allocations"))


class NoOpStrategy(AdaptationStrategy):
    action = Action.NO_OP


class IncrementalStrategy(AdaptationStrategy):
    action = Action.INCREMENTAL

    def training_data(self, base, recent, memory):
        return recent


class ReplayStrategy(AdaptationStrategy):
    action = Action.REPLAY

    def training_data(self, base, recent, memory):
        return combine(recent, memory)


class RetrainStrategy(AdaptationStrategy):
    action = Action.RETRAIN

    def training_data(self, base, recent, memory):
        return recent


class StrategicSelectionStrategy(AdaptationStrategy):
    """SSF-inspired uncertainty/error selection, plus age-based memory expiry.

    Selection uses released labels only. This is an inspired baseline, not an
    implementation of the published SSF algorithm.
    """
    action = Action.STRATEGIC

    def training_data(self, base, recent, memory):
        latest = int(recent.batches.max()) if len(recent) else -1
        memory = memory.take((memory.batches == -1) |
                             (memory.batches >= latest - self.config.stale_after_batches))
        pool = combine(recent, memory)
        if not len(pool):
            return pool
        pred, confidence = base.predict_with_confidence(pool.X)
        scores = (pred != pool.y).astype(float) + 1.0 - confidence
        # Keep each available class represented before filling by informativeness.
        anchors = [int(idx[np.argmax(scores[idx])]) for c in np.unique(pool.y)
                   for idx in [np.flatnonzero(pool.y == c)]]
        capacity = max(self.config.selection_capacity, len(anchors))
        chosen = set(anchors)
        ranked = sorted(range(len(pool)), key=lambda i: (-scores[i], pool.ids[i]))
        chosen.update([i for i in ranked if i not in chosen][:max(0, capacity - len(chosen))])
        return pool.take(np.array(sorted(chosen), int))


STRATEGIES = {s.action: s for s in (NoOpStrategy, IncrementalStrategy, ReplayStrategy,
                                    StrategicSelectionStrategy, RetrainStrategy)}


class AdaptationEngine:
    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()

    def apply(self, action: Action | str, base_model: IDSModel, data: AdaptationData,
              label_policy: LabelPolicy | None = None) -> CandidateModel:
        return STRATEGIES[Action(action)](self.config).apply(base_model, data, label_policy or LabelPolicy())
