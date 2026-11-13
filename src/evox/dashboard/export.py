"""Self-contained HTML export: one file with the UI and a snapshot of the data (works from file://)."""
from __future__ import annotations

import json
from pathlib import Path

from evox.dashboard.api import call, snapshot_key
from evox.dashboard.page import PAGE
from evox.dashboard.store import RunStore


def build_snapshot(store: RunStore, run_ids: list[str] | None = None, max_registries: int = 12) -> dict:
    snap = {snapshot_key("runs"): store.runs()}
    for run in store.runs():
        rid = run["id"]
        if run_ids and rid not in run_ids:
            continue
        for endpoint in ("overview", "registries", "decisions", "results"):
            snap[snapshot_key(endpoint, rid)] = call(store, endpoint, rid)
        registries = snap[snapshot_key("registries", rid)]
        # Prefer the EVO-X registries of a large comparative run so the export stays small.
        ranked = sorted(registries, key=lambda r: (0 if r["system"].startswith("evox") or r["system"] in ("lifecycle demo", "") else 1, r["id"]))
        kept = ranked[:max_registries]
        # List only what the file contains, so the offline page never offers a view it cannot show.
        snap[snapshot_key("registries", rid)] = sorted(kept, key=lambda r: r["id"])
        for reg in kept:
            snap[snapshot_key("timeline", rid, reg["id"])] = call(store, "timeline", rid, reg["id"])
            if reg["scenario"]:
                snap[snapshot_key("batches", rid, reg["scenario"], reg["system"])] = call(store, "batches", rid, reg["scenario"], reg["system"])
    return snap


def export_html(runs_dir: Path, output: Path, run_ids: list[str] | None = None, max_registries: int = 12) -> Path:
    store = RunStore(runs_dir)
    snapshot = build_snapshot(store, run_ids, max_registries)
    # "</" is escaped so run text can never terminate the inline script element.
    data = json.dumps(snapshot, allow_nan=False).replace("</", "<\\/")
    html = PAGE.replace("/*__SNAPSHOT__*/null", data)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
