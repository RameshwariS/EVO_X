"""Phase D2 demo script: drift -> controller decision -> candidate -> gate (promote / reject) -> rollback.

The registry events (candidates, gate checks, promotion, rejection, post-promotion rollback, hash-chained
journal) come from the real Phase C lifecycle code in ``lifecycle_demo``.  The controller decision is made by
the real :class:`EvolutionController`, but its *inputs* (drift signature and proxy estimates) are scripted so
the story is deterministic.  They are labelled as scripted in the record and in ``demo.json``; they are not
measured results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evox.adaptation.strategies import Action
from evox.config import RESULTS_DIR
from evox.drift.analyzer import DriftSignature
from evox.evolution.controller import ActionEstimate, EvolutionController, publish_decision
from evox.experiments.lifecycle_demo import run_demo


def scripted_decision(out: Path) -> dict:
    """Controller decision for the scripted drift (real controller, scripted inputs)."""
    signature = DriftSignature(0, 40, 48, 40, 9.3, 4.1, 80.0, 12, 12, False, "abrupt", 0.0, 0.05, 0.3, 0.0, None,
                               [0.2, 0.1, 0.8])
    estimates = {
        Action.REPLAY: ActionEstimate(gain=0.80, forgetting=0.02, cost_seconds=4.0, delta_fpr=-0.02, latency_ms_per_1k=1.0),
        Action.RETRAIN: ActionEstimate(gain=0.85, forgetting=0.95, cost_seconds=3.0, delta_fpr=0.40, latency_ms_per_1k=1.0)}
    controller = EvolutionController(cost_scale=10.0, latency_scale=5.0)
    record = controller.decide(signature, estimates, "v1", "demo:alarm0:batch40", 60.0,
                               context=dict(scenario="scripted_abrupt_drift", data_tag="SYNTHETIC",
                                            scripted_inputs=True,
                                            note="signature and proxy estimates are scripted for the demonstration"))
    if record.chosen_action != Action.REPLAY.value:
        raise RuntimeError("Scripted demo decision no longer selects Memory Replay; update the scripted estimates")
    publish_decision(record, path=out / "evolution_events.jsonl")
    return record.to_dict()


def run_scripted_demo(output: Path) -> dict:
    output = Path(output)
    summary = run_demo(output)                      # real promote / reject / rollback (raises if the dir is not empty)
    decision = scripted_decision(output)
    steps = [
        dict(step=1, what="Concept drift appears", detail="An abrupt feature-regime change with a flipped label relationship (batch 40)."),
        dict(step=2, what="Controller decides", detail=f"Rule '{decision['rule']}' leaves {decision['eligible']} eligible; "
                                                       f"utility ranking chooses '{decision['chosen_action']}' (scripted inputs)."),
        dict(step=3, what="Candidate v2 is built and gated", detail=f"Memory Replay candidate passes every Safety Gate check and is promoted as {summary['promoted_version']}."),
        dict(step=4, what="An unsafe recent-only candidate is rejected", detail=f"A full retrain on recent data only forgets historical traffic; the gate rejects it ({summary['rejected_version']}); the active model does not change."),
        dict(step=5, what="Post-promotion monitor rolls back", detail=f"Traffic keeps its regime but the old label relationship returns; two monitor batches show degradation and the registry restores {summary['restored_version']}."),
        dict(step=6, what="Rollback is exact", detail=f"Restored parent predictions identical to the original: {summary['parent_predictions_identical']}.")]
    (output / "demo.json").write_text(json.dumps(dict(data_tag="SYNTHETIC", scripted_controller_inputs=True, steps=steps),
                                                 indent=2), encoding="utf-8")
    return dict(summary=summary, decision=decision, steps=steps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "runs" / "phase_d_demo")
    ap.add_argument("--no-html", action="store_true", help="do not write the offline dashboard.html")
    ap.add_argument("--serve", action="store_true", help="serve the dashboard after the demo")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    try:
        result = run_scripted_demo(a.output_dir)
    except ValueError as exc:
        print(str(exc))
        return 2
    for s in result["steps"]:
        print(f"{s['step']}. {s['what']}: {s['detail']}")
    if not a.no_html:
        from evox.dashboard.export import export_html
        page = export_html(a.output_dir, a.output_dir / "dashboard.html")
        print(f"\nOffline dashboard: {page}")
    if a.serve:
        from evox.dashboard.server import make_server
        server = make_server(a.output_dir, "127.0.0.1", a.port)
        print(f"Serving http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
