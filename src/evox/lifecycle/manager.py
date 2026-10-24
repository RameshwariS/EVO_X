"""Candidate -> evaluator -> policy -> durable registry -> active model."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from evox.adaptation.data import LabelPolicy, Samples, combine
from evox.evaluation.candidate import CandidateEvaluator, EvaluationSets, model_metrics
from evox.evolution.safety import GateDecision, SafetyGate
from evox.lifecycle.registry import FileModelRegistry, RegistryError
from evox.streaming.consumer import ModelHolder


@dataclass(frozen=True)
class LifecycleResult:
    version: str
    status: str
    evaluation: dict
    decision: dict
    duplicate: bool = False


class EvolutionLifecycle:
    def __init__(self, registry: FileModelRegistry, gate: SafetyGate):
        if registry.active_version is None:
            raise RegistryError("Initialize the registry before starting the lifecycle")
        self.registry, self.gate = registry, gate
        self.holder = ModelHolder(registry.load(), registry.active_version)
        self.evaluator = CandidateEvaluator()

    def process(self, candidate, sets: EvaluationSets, policy: LabelPolicy, now_batch: int,
                best_historical=None, initial_historical=None, forbidden_ids=(), context=None,
                bypass_gate=False):
        parent, version = self.holder.get()
        if candidate.status == "NO_OP":
            if candidate.parent_version != version:
                raise RegistryError("Stale No-Op parent")
            self.registry.record_event("NO_OP", dict(parent_version=version, context=context or {}))
            return LifecycleResult(version, "NO_OP", {}, {})
        existing = self.registry.candidate_version(candidate.candidate_id)
        if existing and self.registry.versions[existing]["status"] != "CANDIDATE":
            r = self.registry.versions[existing]
            return LifecycleResult(existing, r["status"], r["metrics"], r["gate"], duplicate=True)
        if candidate.parent_version != version or version != self.registry.active_version:
            raise RegistryError("Candidate/holder/registry versions are inconsistent")
        registered = self.registry.register(candidate, dict(context or {}, selection_ids=list(forbidden_ids)))
        try:
            evaluation = self.evaluator.evaluate(candidate, parent, sets, policy, now_batch, best_historical,
                                                 initial_historical,
                                                 set(forbidden_ids) | self.registry.ancestor_training_ids())
            decision = self.gate.decide(evaluation)
            if bypass_gate:
                decision = GateDecision(True, decision.checks, decision.thresholds, "unconditional_baseline_or_ablation")
        except ValueError as exc:
            # Data-integrity failures never become unconditional baseline deployments.
            evaluation = dict(error=str(exc), candidate_id=candidate.candidate_id)
            decision = GateDecision(False, {}, asdict(self.gate.thresholds), "invalid_evaluation:" + str(exc))
        if decision.promote:
            self.registry.promote(registered, evaluation, decision.to_dict())
            self.holder.swap(candidate.model, registered)
            return LifecycleResult(registered, "DEPLOYED", evaluation, decision.to_dict())
        self.registry.reject(registered, evaluation, decision.to_dict())
        return LifecycleResult(registered, "REJECTED", evaluation, decision.to_dict())

    def rollback(self, reason, evidence=None):
        version = self.registry.rollback(reason, evidence)
        self.holder.swap(self.registry.load(version), version)
        return version


@dataclass(frozen=True)
class MonitorConfig:
    min_batches: int = 2
    min_rows: int = 32
    max_new_drop: float = 0.03
    max_historical_drop: float = 0.05
    max_fpr_increase: float = 0.02

    def __post_init__(self):
        values = (self.max_new_drop, self.max_historical_drop, self.max_fpr_increase)
        if min(self.min_batches, self.min_rows) < 1 or any(not np.isfinite(x) or x < 0 for x in values):
            raise ValueError("Invalid rollback-monitor settings")


class RollbackMonitor:
    """Compare the deployed candidate to its frozen parent on fresh monitor rows.

    Caller supplies only batches actually scored by this deployed version. Gate,
    proxy and final-report rows must be different source-flow groups.
    """
    def __init__(self, lifecycle: EvolutionLifecycle, parent_model, activated_at: int,
                 historical: Samples, config: MonitorConfig | None = None):
        self.lifecycle, self.parent, self.activated_at = lifecycle, parent_model, activated_at
        self.version = lifecycle.holder.get()[1]
        self.historical, self.config = historical, config or MonitorConfig()
        self.window = Samples.empty(historical.X.shape[1])
        self.finished = False
        self._last_observed_batch = activated_at

    def observe(self, samples: Samples, policy: LabelPolicy, now_batch: int, scored_version: str):
        if self.finished or scored_version != self.version or self.lifecycle.holder.get()[1] != self.version:
            return None
        samples = samples.take(samples.batches > self.activated_at)
        if len(policy.available(samples, now_batch)) != len(samples):
            raise ValueError("Rollback monitor attempted to use unreleased labels")
        if len(policy.available(self.historical, now_batch)) != len(self.historical):
            raise ValueError("Historical monitoring labels are not available")
        record = self.lifecycle.registry.versions[self.version]
        forbidden = (self.lifecycle.registry.ancestor_training_ids() |
                     set(record.get("metrics", {}).get("evaluation_ids", [])) |
                     set(record.get("context", {}).get("selection_ids", [])))
        if set(samples.ids) & forbidden or set(self.historical.ids) & forbidden:
            raise ValueError("Rollback monitor overlaps model training")
        samples = samples.take(samples.batches > self._last_observed_batch)
        if not len(samples):
            return None
        self._last_observed_batch = int(samples.batches.max())
        self.window = combine(self.window, samples)
        c = self.config
        if len(self.window) < c.min_rows or len(np.unique(self.window.batches)) < c.min_batches:
            return None
        current = self.lifecycle.holder.get()[0]
        new, old = model_metrics(current, self.window), model_metrics(self.parent, self.window)
        historical_new, historical_old = model_metrics(current, self.historical), model_metrics(self.parent, self.historical)
        evidence = dict(version=self.version, evaluated_at_batch=now_batch, n=len(self.window),
                        delta_f1_new=new["macro_f1"] - old["macro_f1"],
                        delta_f1_historical=historical_new["macro_f1"] - historical_old["macro_f1"],
                        delta_fpr=new["binary"]["fpr"] - old["binary"]["fpr"])
        harmful = (evidence["delta_f1_new"] < -c.max_new_drop or
                   evidence["delta_f1_historical"] < -c.max_historical_drop or
                   evidence["delta_fpr"] > c.max_fpr_increase)
        self.lifecycle.registry.record_event("MONITOR_EVALUATED", evidence)
        # Use a bounded non-overlapping monitoring window, and keep monitoring a
        # good promotion until superseded or the stream ends.
        self.window = Samples.empty(self.window.X.shape[1])
        if harmful:
            self.finished = True
            evidence["restored_version"] = self.lifecycle.rollback("post_promotion_degradation", evidence)
        evidence["rolled_back"] = harmful
        return evidence
