"""Cheap shadow candidates and disjoint held-out measurements for Phase B.

This is a controller estimator, not Phase C's final evaluator or safety gate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

from evox.adaptation.strategies import Action, AdaptationEngine, StrategyConfig
from evox.adaptation.data import AdaptationData, LabelPolicy, Samples
from evox.evolution.controller import ActionEstimate
from evox.models.ids import IDSModel, UnsupportedUpdate
from evox.evaluation.metrics import binary_metrics


@dataclass(frozen=True)
class ProxyHoldouts:
    recent: Samples
    historical: Samples

    def validate(self, data: AdaptationData, policy: LabelPolicy):
        a, b = set(self.recent.ids), set(self.historical.ids)
        if not a or not b or a & b:
            raise ValueError("Proxy holdouts must be nonempty and disjoint")
        recent, memory = data.released(policy)
        if (a | b) & (set(recent.ids) | set(memory.ids)):
            raise ValueError("Proxy holdout overlaps training data")
        for samples in (self.recent, self.historical):
            if len(policy.available(samples, data.now_batch)) != len(samples):
                raise ValueError("Proxy evaluation attempted to use unreleased labels")


def measure(model: IDSModel, holdouts: ProxyHoldouts):
    scores = {}
    predictions = []
    for name, s in (("recent", holdouts.recent), ("historical", holdouts.historical)):
        yp = model.predict(s.X)
        scores[name + "_f1"] = float(f1_score(s.y, yp, labels=np.unique(s.y), average="macro", zero_division=0))
        predictions.append(yp)
    combined_y = np.concatenate([holdouts.recent.y, holdouts.historical.y])
    scores["fpr"] = binary_metrics(combined_y, np.concatenate(predictions))["fpr"]
    return scores


def parent_latency(model: IDSModel, X):
    model.predict(X)
    t0 = time.perf_counter(); model.predict(X)
    return (time.perf_counter() - t0) * 1e6 / len(X)


class ProxyEvaluator:
    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig(recent_capacity=256, memory_capacity=256, selection_capacity=256)

    def evaluate(self, base: IDSModel, data: AdaptationData, policy: LabelPolicy,
                 holdouts: ProxyHoldouts, audit: ProxyHoldouts | None = None):
        holdouts.validate(data, policy)
        if audit is not None:
            audit.validate(data, policy)
            if (set(audit.recent.ids) | set(audit.historical.ids)) & (set(holdouts.recent.ids) | set(holdouts.historical.ids)):
                raise ValueError("Tuning audit data overlaps proxy holdout")
        baseline = measure(base, holdouts)
        latency = parent_latency(base, holdouts.recent.X[:128])
        engine, estimates, profiles = AdaptationEngine(self.config), {Action.NO_OP: ActionEstimate()}, []
        outcomes = {Action.NO_OP.value: measure(base, audit or holdouts)}
        unavailable = {}
        for action in Action:
            if action == Action.NO_OP:
                continue
            try:
                c = engine.apply(action, base, data, policy)
            except (UnsupportedUpdate, ValueError) as exc:
                unavailable[action.value] = str(exc)
                continue
            metrics = measure(c.model, holdouts)
            estimates[action] = ActionEstimate(metrics["recent_f1"] - baseline["recent_f1"],
                                               max(0.0, baseline["historical_f1"] - metrics["historical_f1"]),
                                               c.profile["training_seconds"], metrics["fpr"] - baseline["fpr"],
                                               max(0.0, c.profile["latency_ms_per_1k"] - latency))
            outcomes[action.value] = measure(c.model, audit or holdouts)
            profiles.append(dict(action=action.value, algorithm=base.kind, **c.profile,
                                 parent_latency_ms_per_1k=latency))
        return estimates, outcomes, profiles, unavailable, baseline
