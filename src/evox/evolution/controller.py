"""Phase B transparent rules + multi-objective ranking, with shadow-only events."""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from evox.adaptation.strategies import Action, CandidateModel
from evox.config import TOPIC_EVOLUTION
from evox.drift.analyzer import DriftSignature


@dataclass(frozen=True)
class ControllerThresholds:
    low_magnitude: float = 1.0
    high_magnitude: float = 3.0
    short_persistence: int = 2
    long_persistence: int = 10
    recurrence: float = 0.9
    novelty: float = 0.2

    def __post_init__(self):
        if not 0 <= self.low_magnitude < self.high_magnitude:
            raise ValueError("Magnitude thresholds must be ordered")
        if not 0 <= self.short_persistence < self.long_persistence:
            raise ValueError("Persistence thresholds must be ordered")
        if not 0 <= self.recurrence <= 1 or not 0 <= self.novelty <= 1:
            raise ValueError("Recurrence/novelty thresholds must be in [0, 1]")


@dataclass(frozen=True)
class UtilityWeights:
    gain: float = 0.5
    forgetting: float = 0.25
    cost: float = 0.1
    fpr: float = 0.1
    latency: float = 0.05

    def __post_init__(self):
        v = np.array(list(asdict(self).values()), dtype=float)
        if not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0:
            raise ValueError("Utility weights must be finite, nonnegative, and have positive sum")

    def normalized(self):
        d = asdict(self); total = sum(d.values())
        return {k: v / total for k, v in d.items()}


@dataclass(frozen=True)
class ActionEstimate:
    gain: float = 0.0
    forgetting: float = 0.0
    cost_seconds: float = 0.0
    delta_fpr: float = 0.0
    latency_ms_per_1k: float = 0.0       # incremental latency over parent, clipped at zero

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("Utility estimates must be finite")
        if min(self.forgetting, self.cost_seconds, self.latency_ms_per_1k) < 0:
            raise ValueError("Forgetting, cost, and added latency must be nonnegative")


@dataclass
class DecisionRecord:
    event_id: str
    parent_version: str
    signature: dict
    mode: str
    rule: str
    inputs: dict
    eligible: list[str]
    excluded: dict
    estimates: dict
    utilities: dict
    chosen_action: str
    weights: dict
    thresholds: dict
    candidate: dict | None = None
    deployment_status: str = "SHADOW_ONLY"

    def to_dict(self):
        return asdict(self)


class EvolutionController:
    def __init__(self, weights: UtilityWeights | None = None, thresholds: ControllerThresholds | None = None,
                 cost_scale: float = 1.0, latency_scale: float = 1.0, mode: str = "rules_utility"):
        if mode not in ("rules_only", "utility_only", "rules_utility"):
            raise ValueError("Unknown controller ablation")
        if not math.isfinite(cost_scale) or not math.isfinite(latency_scale) or min(cost_scale, latency_scale) <= 0:
            raise ValueError("Utility normalization scales must be finite and positive")
        self.weights = weights or UtilityWeights()
        self.thresholds = thresholds or ControllerThresholds()
        self.cost_scale, self.latency_scale, self.mode = cost_scale, latency_scale, mode

    def rule_actions(self, sig: DriftSignature):
        t = self.thresholds
        if sig.magnitude < t.low_magnitude or sig.persistence < t.short_persistence or sig.ended:
            return [Action.NO_OP], "insignificant_short_or_ended"
        if sig.recurrence >= t.recurrence:
            return [Action.REPLAY, Action.INCREMENTAL, Action.NO_OP], "recurring"
        # Phase A: low confidence is NOT proof of a novel attack. This only marks
        # eligibility; held-out performance and utility still decide the action.
        if sig.novelty >= t.novelty and sig.magnitude >= t.high_magnitude:
            return [Action.RETRAIN, Action.STRATEGIC, Action.NO_OP], "high_unfamiliarity_and_magnitude"
        if sig.magnitude >= t.high_magnitude and sig.persistence >= t.long_persistence:
            return [Action.RETRAIN, Action.REPLAY, Action.NO_OP], "major_persistent"
        return [Action.INCREMENTAL, Action.STRATEGIC, Action.REPLAY, Action.NO_OP], "moderate"

    def utility(self, e: ActionEstimate):
        w = self.weights.normalized()
        terms = dict(gain=w["gain"] * e.gain, forgetting=-w["forgetting"] * e.forgetting,
                     cost=-w["cost"] * e.cost_seconds / self.cost_scale, fpr=-w["fpr"] * e.delta_fpr,
                     latency=-w["latency"] * e.latency_ms_per_1k / self.latency_scale)
        return dict(terms=terms, total=sum(terms.values()))

    def decide(self, sig: DriftSignature, estimates: dict[Action, ActionEstimate], parent_version: str,
               event_id: str, budget_seconds: float = 60.0, context: dict | None = None,
               unavailable: dict | None = None) -> DecisionRecord:
        if not math.isfinite(budget_seconds) or budget_seconds < 0:
            raise ValueError("Budget must be finite and nonnegative")
        estimates = {Action(k): v for k, v in estimates.items()}
        estimates[Action.NO_OP] = ActionEstimate()  # always a valid, zero-cost fallback
        allowed, rule = self.rule_actions(sig)
        if self.mode == "utility_only":
            allowed, rule = list(Action), "utility_only"
        eligible, excluded = [], {}
        for action in Action:
            if action not in allowed:
                excluded[action.value] = "rule_filter"
            elif action not in estimates:
                excluded[action.value] = (unavailable or {}).get(action.value, "no_valid_proxy_estimate")
            elif estimates[action].cost_seconds > budget_seconds:
                excluded[action.value] = "adaptation_budget"
            else:
                eligible.append(action)
        utilities = {a.value: self.utility(estimates[a]) for a in eligible}
        if self.mode == "rules_only":
            chosen = next(a for a in allowed if a in eligible)
        else:
            # Stable tie break: prefer No-Op, then lower cost, then action name.
            chosen = min(eligible, key=lambda a: (-utilities[a.value]["total"], a != Action.NO_OP,
                                                  estimates[a].cost_seconds, a.value))
        return DecisionRecord(event_id, parent_version, sig.to_dict(), self.mode, rule,
                              dict(budget_seconds=budget_seconds, cost_scale=self.cost_scale,
                                   latency_scale=self.latency_scale, context=context or {}),
                              [a.value for a in eligible], excluded,
                              {a.value: asdict(v) for a, v in estimates.items()}, utilities,
                              chosen.value, self.weights.normalized(), asdict(self.thresholds))


def publish_decision(record: DecisionRecord, candidate: CandidateModel | None = None,
                     path: Path | None = None, transport=None):
    if candidate is not None:
        if candidate.parent_version != record.parent_version or candidate.action != record.chosen_action:
            raise ValueError("Candidate does not match controller decision")
        record.candidate = candidate.record()
    payload = json.dumps(record.to_dict(), allow_nan=False, sort_keys=True)
    if path is not None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(payload + "\n")
    if transport is not None:
        transport.send(TOPIC_EVOLUTION, payload.encode("utf-8"), key=record.signature["signature_id"])


def weight_grid(values=(0.0, 0.25, 0.5, 1.0)):
    """Pre-registered normalized five-dimensional grid; deduplicate equivalent vectors."""
    seen = set()
    for v in itertools.product(values, repeat=5):
        if sum(v) <= 0:
            continue
        key = tuple(round(x / sum(v), 12) for x in v)
        if key not in seen:
            seen.add(key)
            yield UtilityWeights(*v)


def tune_weights(episodes, cost_scale: float, latency_scale: float, cost_penalty: float = 0.05,
                 thresholds: ControllerThresholds | None = None):
    """Select weights on VALIDATION episodes, using a second independent holdout.

    episodes contain signature, estimates (proxy holdout) and outcomes (audit
    holdout). Test-family episodes and seeds are refused, even if renamed.
    """
    from evox.config import TEST_SEEDS, VALIDATION_SEEDS

    if not episodes:
        raise ValueError("Cannot select weights without validation episodes")
    for ep in episodes:
        if ep["family"] != "validation" or ep["seed"] not in VALIDATION_SEEDS or ep["seed"] in TEST_SEEDS:
            raise ValueError("Weight selection accepts validation scenarios/seeds only")
    rows, best = [], None
    for weights in weight_grid():
        controller = EvolutionController(weights, thresholds, cost_scale, latency_scale)
        objectives, costs = [], []
        for i, ep in enumerate(episodes):
            rec = controller.decide(ep["signature"], ep["estimates"], "v1", str(i), ep["budget_seconds"])
            out = ep["outcomes"][rec.chosen_action]
            cost = ep["estimates"].get(Action(rec.chosen_action), ActionEstimate()).cost_seconds
            objectives.append(0.5 * (out["recent_f1"] + out["historical_f1"]) - cost_penalty * cost / cost_scale)
            costs.append(cost)
        row = dict(**weights.normalized(), objective=float(np.mean(objectives)), mean_cost=float(np.mean(costs)),
                   n_episodes=len(episodes))
        rows.append(row)
        key = (row["objective"], -row["mean_cost"])
        if best is None or key > best[0]:
            best = (key, weights)
    return best[1], rows
