"""Explicit, fail-closed candidate promotion policy and validation-only tuning."""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np

from evox.config import TEST_SEEDS, VALIDATION_SEEDS


@dataclass(frozen=True)
class GateThresholds:
    min_new_gain: float = 0.01
    max_historical_drop: float = 0.03
    max_forgetting: float = 0.05
    max_fpr_increase: float = 0.01
    max_training_seconds: float = 60.0
    max_latency_ms_per_1k: float = 100.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Gate thresholds must be finite and nonnegative")


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    checks: dict
    thresholds: dict
    reason: str

    def to_dict(self):
        return asdict(self)


class SafetyGate:
    def __init__(self, thresholds: GateThresholds | None = None):
        self.thresholds = thresholds or GateThresholds()

    def decide(self, evaluation: dict) -> GateDecision:
        t = self.thresholds
        requirements = {
            "new_gain": ("delta_f1_new", t.min_new_gain, True),
            "historical_retention": ("delta_f1_historical", -t.max_historical_drop, True),
            "forgetting": ("forgetting", t.max_forgetting, False),
            "false_positives": ("delta_fpr", t.max_fpr_increase, False),
            "training_budget": ("training_seconds", t.max_training_seconds, False),
            "latency_budget": ("latency_ms_per_1k", t.max_latency_ms_per_1k, False),
        }
        checks = {}
        for name, (key, bound, lower) in requirements.items():
            value = evaluation.get(key)
            finite = isinstance(value, (int, float)) and math.isfinite(value)
            if finite and key in ("forgetting", "training_seconds", "latency_ms_per_1k") and value < 0:
                finite = False
            if finite and key in ("delta_f1_new", "delta_f1_historical", "delta_fpr") and not -1 <= value <= 1:
                finite = False
            valid = finite and (value >= bound - 1e-12 if lower else value <= bound + 1e-12)
            checks[name] = dict(value=value if finite else None, bound=bound, passed=bool(valid))
        failed = [name for name, check in checks.items() if not check["passed"]]
        return GateDecision(not failed, checks, asdict(t), "all_checks_passed" if not failed else ",".join(failed))


def tune_gate(records, budget_seconds=60.0, latency_budget=100.0):
    """One-step validation pilot: audit outcomes select the gate, not its gate holdout.

    Pilots keep the initial parent fixed. The final experiment is closed-loop;
    pilot estimates must not be reported as deployed performance.
    """
    if not records:
        raise ValueError("Gate tuning requires validation candidates")
    for r in records:
        if r["family"] != "validation" or r["seed"] not in VALIDATION_SEEDS or r["seed"] in TEST_SEEDS:
            raise ValueError("Gate tuning accepts validation scenarios/seeds only")
    cost_scale = max([r["evaluation"]["training_seconds"] for r in records] + [1e-6])
    best, rows = None, []
    grid = itertools.product((0.0, 0.01, 0.02), (0.02, 0.05), (0.03, 0.1), (0.005, 0.02),
                             (budget_seconds / 2, budget_seconds))
    for gain, old, forgetting, fpr, cost in grid:
        thresholds = GateThresholds(gain, old, forgetting, fpr, cost, latency_budget)
        gate = SafetyGate(thresholds)
        objectives, harmful, accepted, total_cost = [], 0, 0, 0.0
        for r in records:
            passed = gate.decide(r["evaluation"]).promote
            if passed:
                a = r["audit"]
                objectives.append(.5 * (a["delta_f1_new"] + a["delta_f1_historical"]) -
                                  .05 * r["evaluation"]["training_seconds"] / cost_scale)
                harmful += a["delta_f1_new"] < -.03 or a["delta_f1_historical"] < -.03
                accepted += 1; total_cost += r["evaluation"]["training_seconds"]
            else:
                objectives.append(0.0)
        row = dict(**asdict(thresholds), objective=float(np.mean(objectives)), n_accepted=accepted,
                   harmful_accepted=int(harmful), accepted_cost_seconds=total_cost, n_candidates=len(records))
        rows.append(row)
        # Predeclared conservative tie break, never test outcomes.
        key = (row["objective"], -harmful, -total_cost, gain, -old, -forgetting, -fpr, -cost)
        if best is None or key > best[0]:
            best = (key, thresholds)
    return best[1], rows
