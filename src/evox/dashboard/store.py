"""Read-only access to experiment run directories for the dashboard.

Run identifiers are resolved ONLY from the discovered set, never from raw URL text, so a request
cannot read files outside the runs root.  Everything read from disk is treated as data: the UI
escapes it, and nothing here executes or evaluates file content.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from evox.lifecycle.registry import canonical

MAX_EVENTS = 400          # cap rows returned per request so a huge run cannot freeze the browser
HISTORY = "history.jsonl"


def _clean(value):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_csv(path: Path, limit: int | None = None) -> list[dict]:
    try:
        frame = pd.read_csv(path, nrows=limit)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return []
    return _clean(frame.astype(object).where(frame.notna(), None).to_dict("records"))


def verify_history(path: Path) -> dict:
    """Recompute the append-only hash chain without raising.  Reports, never repairs."""
    previous, count = "0" * 64, 0
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:
        return dict(ok=False, events=0, detail=str(exc))
    for number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
            digest = event.pop("hash")
            if not line.endswith("\n"):
                return dict(ok=False, events=count, detail="incomplete journal tail")
            if event["sequence"] != number or event["previous_hash"] != previous:
                return dict(ok=False, events=count, detail=f"sequence/hash chain broken at event {number}")
            if hashlib.sha256(canonical(event).encode()).hexdigest() != digest:
                return dict(ok=False, events=count, detail=f"checksum mismatch at event {number}")
        except (KeyError, ValueError):
            return dict(ok=False, events=count, detail=f"malformed event {number}")
        previous, count = digest, number
    return dict(ok=True, events=count, detail="hash chain verified")


def _trim(value, depth=0):
    """Drop bulky fields (training ids) and long lists before sending to a browser."""
    if isinstance(value, dict):
        return {k: _trim(v, depth + 1) for k, v in value.items() if k not in ("training_ids", "snapshot")}
    if isinstance(value, list):
        return f"<{len(value)} items>" if len(value) > 12 else [_trim(v, depth + 1) for v in value]
    return value


def summarize_history(path: Path) -> dict:
    """Events (compact) and the final state of every version from a registry journal."""
    events, versions, order = [], {}, []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return dict(events=[], versions=[], active=None, integrity=dict(ok=False, events=0, detail="unreadable"))
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for u in event.get("updates", []):
            if u["version"] not in versions:
                order.append(u["version"])
            versions[u["version"]] = u
        events.append(dict(sequence=event.get("sequence"), kind=event.get("kind"), timestamp=event.get("timestamp"),
                           active_before=event.get("active_before"), active_after=event.get("active_after"),
                           payload=_trim(event.get("payload", {}))))
    rows = []
    for v in order:
        u = versions[v]
        gate = u.get("gate") or {}
        ctx = u.get("context") or {}
        controller = ctx.get("controller") or {}
        rows.append(dict(
            version=v, status=u.get("status"), action=u.get("action"), parent_version=u.get("parent_version"),
            candidate_id=u.get("candidate_id"), model_kind=u.get("model_kind"), n_train=(u.get("profile") or {}).get("n_train"),
            training_seconds=(u.get("profile") or {}).get("training_seconds"),
            gate=dict(promote=gate.get("promote"), reason=gate.get("reason"), checks=gate.get("checks", {})),
            metrics=_trim(u.get("metrics") or {}),
            signature=_trim(ctx.get("signature") or controller.get("signature")),
            controller=_trim({k: controller.get(k) for k in ("rule", "eligible", "excluded", "chosen_action", "utilities",
                                                                "estimates", "weights") if controller}),
            sha256=u.get("sha256")))
    return _clean(dict(events=events[-MAX_EVENTS:], n_events=len(events), versions=rows,
                       active=events[-1]["active_after"] if events else None, integrity=verify_history(path)))


def classify(path: Path) -> str | None:
    if (path / "demo.json").is_file() and (path / "registry" / HISTORY).is_file():
        return "demo"
    if (path / "controllers.json").is_file() and (path / "per_scenario.csv").is_file():
        return "phase_d_transfer"
    if (path / "portability.csv").is_file():
        return "phase_d_generalization"
    if (path / "per_scenario.csv").is_file() and (path / "registries").is_dir():
        return "phase_c"
    if (path / "evolution_events.jsonl").is_file():
        return "phase_b"
    if (path / "registry" / HISTORY).is_file() or (path / HISTORY).is_file():
        return "lifecycle"
    return None


class RunStore:
    """Discovers runs under ``root`` (a run may also be ``root`` itself)."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._runs: dict[str, tuple[str, Path]] = {}
        self.refresh()

    def refresh(self) -> None:
        found: dict[str, tuple[str, Path]] = {}
        # Depth-limited walk (a Phase C run holds thousands of registry folders; never descend into them).
        candidates, level = [self.root], [self.root]
        for _ in range(2):
            nxt = []
            for parent in level:
                try:
                    children = sorted(p for p in parent.iterdir() if p.is_dir() and p.name not in ("registries", "models", "registry", "__pycache__"))
                except OSError:
                    continue
                nxt.extend(children)
            candidates.extend(nxt)
            level = nxt
        for p in candidates:
            kind = classify(p)
            if kind:
                rid = "." if p == self.root else p.relative_to(self.root).as_posix()
                found[rid] = (kind, p)
        self._runs = found

    def _get(self, rid: str) -> tuple[str, Path]:
        if rid not in self._runs:
            raise KeyError(f"Unknown run {rid!r}")
        return self._runs[rid]

    # -- listings ----------------------------------------------------------
    def runs(self) -> list[dict]:
        out = []
        for rid, (kind, path) in sorted(self._runs.items()):
            summary = read_json(path / "summary.json", {}) or {}
            protocol = read_json(path / "preregistered_protocol.json", {}) or {}
            out.append(dict(id=rid, kind=kind, data_tag=summary.get("data_tag") or protocol.get("data_tag"),
                            algorithm=summary.get("algorithm") or protocol.get("algorithm"),
                            quick=bool(summary.get("quick", protocol.get("quick", False)))))
        return _clean(out)

    def overview(self, rid: str) -> dict:
        kind, path = self._get(rid)
        protocol = read_json(path / "preregistered_protocol.json", {}) or {}
        keep = ("data_tag", "quick", "algorithm", "source_algorithm", "label_policy", "validation_seeds", "test_seeds",
                "magnitudes", "systems", "role_split", "harmful", "unnecessary", "rollback", "design", "hypotheses")
        summary = read_json(path / "summary.json", {}) or {}
        hypotheses = read_json(path / "hypotheses.json", {}) or {}
        if kind == "phase_d_transfer" and isinstance(summary.get("H5"), dict):
            hypotheses = dict(H5=dict(claim="A controller tuned on one algorithm transfers to another without re-tuning",
                                      **summary["H5"]))
        return _clean(dict(id=rid, kind=kind, summary={k: v for k, v in summary.items() if k != "H5"},
                           hypotheses=hypotheses,
                           protocol={k: protocol[k] for k in keep if k in protocol},
                           demo_steps=(read_json(path / "demo.json", {}) or {}).get("steps"),
                           report=(path / "report.md").read_text(encoding="utf-8")[:6000]
                           if (path / "report.md").is_file() else None))

    def registries(self, rid: str) -> list[dict]:
        kind, path = self._get(rid)
        out = []
        if (path / "registry" / HISTORY).is_file():
            out.append(dict(id="registry", scenario="", system="lifecycle demo"))
        elif (path / HISTORY).is_file():
            out.append(dict(id=".", scenario="", system=""))
        reg = path / "registries"
        if reg.is_dir():
            for scenario in sorted(p for p in reg.iterdir() if p.is_dir()):
                for system in sorted(p for p in scenario.iterdir() if (p / HISTORY).is_file()):
                    out.append(dict(id=f"{scenario.name}/{system.name}", scenario=scenario.name, system=system.name))
        return out

    def timeline(self, rid: str, registry: str) -> dict:
        _, path = self._get(rid)
        allowed = {r["id"]: r for r in self.registries(rid)}
        if registry not in allowed:
            raise KeyError(f"Unknown registry {registry!r}")
        history = (path / HISTORY if registry == "." else path / registry / HISTORY)
        return dict(registry=allowed[registry], **summarize_history(history))

    def decisions(self, rid: str) -> dict:
        kind, path = self._get(rid)
        records = []
        events = path / "evolution_events.jsonl"
        if events.is_file():
            for line in events.read_text(encoding="utf-8").splitlines()[:MAX_EVENTS]:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(dict(
                    event_id=d.get("event_id"), mode=d.get("mode"), rule=d.get("rule"), chosen=d.get("chosen_action"),
                    eligible=d.get("eligible"), excluded=d.get("excluded"), utilities=d.get("utilities"),
                    estimates=d.get("estimates"), weights=d.get("weights"), thresholds=d.get("thresholds"),
                    signature={k: v for k, v in (d.get("signature") or {}).items() if k != "per_feature_psi"},
                    scenario=(d.get("inputs") or {}).get("context", {}).get("scenario"),
                    deployment_status=d.get("deployment_status")))
        candidates = read_csv(path / "candidate_events.csv", MAX_EVENTS)
        return _clean(dict(controller_decisions=records, candidate_events=candidates))

    def results(self, rid: str) -> dict:
        kind, path = self._get(rid)
        table = path / "per_scenario.csv"
        out: dict = dict(systems=[], scenarios=[])
        if table.is_file():
            frame = pd.read_csv(table)
            cols = [c for c in ("macro_f1", "forgetting", "updates", "rejected", "rollbacks", "training_seconds",
                                "harmful_promotions", "unnecessary_updates") if c in frame.columns]
            if "system" in frame.columns and cols:
                means = frame.groupby("system")[cols].mean().reset_index()
                out["systems"] = _clean(means.to_dict("records"))
            keep = [c for c in ("scenario", "system", "kind", "magnitude", "seed", "macro_f1", "forgetting", "updates",
                                "rejected", "rollbacks") if c in frame.columns]
            out["scenarios"] = _clean(frame[keep].head(MAX_EVENTS).to_dict("records"))
        for name in ("paired_statistics.csv", "portability.csv", "decision_agreement.csv"):
            if (path / name).is_file():
                out[name[:-4]] = read_csv(path / name, MAX_EVENTS)
        return _clean(out)

    def batches(self, rid: str, scenario: str, system: str) -> list[dict]:
        _, path = self._get(rid)
        rows = read_csv(path / "batch_metrics.csv")
        return [r for r in rows if r.get("scenario") == scenario and r.get("system") == system][:5000]
