"""Phase D2: read-only dashboard store, routing, integrity check, safe export and the scripted demo."""
import json
import shutil
import threading
import urllib.error
import urllib.request

import pytest

from evox.dashboard.api import handle, snapshot_key
from evox.dashboard.export import export_html
from evox.dashboard.server import make_server
from evox.dashboard.store import RunStore, verify_history
from evox.experiments.demo import run_scripted_demo


@pytest.fixture(scope="module")
def demo_root():
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp()) / "runs"
    run_scripted_demo(root / "demo")
    return root


def get(store, url):
    status, ctype, body = handle(store, url)
    return status, (json.loads(body) if ctype == "application/json" else body)


def test_demo_run_is_discovered_with_controller_decision_and_story(demo_root):
    store = RunStore(demo_root)
    assert [(r["id"], r["kind"], r["data_tag"]) for r in store.runs()] == [("demo", "demo", "SYNTHETIC")]
    status, overview = get(store, "/api/overview?run=demo")
    assert status == 200 and len(overview["demo_steps"]) == 6
    status, decisions = get(store, "/api/decisions?run=demo")
    record = decisions["controller_decisions"][0]
    assert record["chosen"] == "replay" and record["rule"] == "major_persistent"
    assert record["deployment_status"] == "SHADOW_ONLY" and set(record["utilities"]) == {"no_op", "replay", "full_retrain"}


def test_timeline_shows_promotion_rejection_rollback_and_verified_chain(demo_root):
    store = RunStore(demo_root)
    status, regs = get(store, "/api/registries?run=demo")
    assert status == 200 and [r["id"] for r in regs] == ["registry"]
    status, tl = get(store, "/api/timeline?run=demo&registry=registry")
    kinds = [e["kind"] for e in tl["events"]]
    assert kinds.index("PROMOTED") < kinds.index("REJECTED") < kinds.index("ROLLED_BACK")
    assert tl["active"] == "v1" and tl["integrity"]["ok"] and tl["integrity"]["events"] == len(kinds)
    statuses = {v["version"]: v["status"] for v in tl["versions"]}
    assert statuses["v3"] == "REJECTED" and statuses["v2"] == "ROLLED_BACK"
    v3 = next(v for v in tl["versions"] if v["version"] == "v3")
    assert v3["gate"]["promote"] is False and any(not c["passed"] for c in v3["gate"]["checks"].values())
    assert all("training_ids" not in json.dumps(v) for v in tl["versions"])        # bulky ids are never sent


def test_tampered_journal_is_reported_not_trusted(demo_root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(demo_root / "demo", copy)
    history = copy / "registry" / "history.jsonl"
    lines = history.read_text().splitlines(keepends=True)
    lines[2] = lines[2].replace('"PROMOTED"', '"REJECTED"')
    history.write_text("".join(lines))
    result = verify_history(history)
    assert result["ok"] is False and "event 3" in result["detail"]
    assert verify_history(copy / "registry" / "history.jsonl.missing")["ok"] is False
    history.write_text("".join(lines[:2]) + lines[2].rstrip("\n"))               # incomplete tail
    assert verify_history(history)["ok"] is False


def test_routing_rejects_unknown_traversal_and_missing_parameters(demo_root):
    store = RunStore(demo_root)
    assert handle(store, "/api/overview?run=../../etc")[0] == 404
    assert handle(store, "/api/overview?run=%2E%2E%2F%2E%2E")[0] == 404
    assert handle(store, "/api/timeline?run=demo&registry=../../x")[0] == 404
    assert handle(store, "/api/timeline?run=demo")[0] == 400
    assert handle(store, "/api/unknown")[0] == 404 and handle(store, "/nope")[0] == 404
    assert handle(store, "/healthz")[0] == 200
    assert handle(store, "/")[0] == 200 and b"EVO-X dashboard" in handle(store, "/")[2]


def test_server_is_read_only_and_local(demo_root):
    server = make_server(demo_root, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert urllib.request.urlopen(base + "/healthz", timeout=5).read() == b"ok"
        data = json.loads(urllib.request.urlopen(base + "/api/runs", timeout=5).read())
        assert data[0]["id"] == "demo"
        reply = urllib.request.urlopen(base + "/", timeout=5)
        assert "default-src 'self'" in reply.headers["Content-Security-Policy"]
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(urllib.request.Request(base + "/api/runs", data=b"{}", method="POST"), timeout=5)
        assert err.value.code == 405
    finally:
        server.shutdown()
        server.server_close()


def test_export_is_self_contained_and_matches_the_page_key_format(demo_root, tmp_path):
    page = export_html(demo_root, tmp_path / "out" / "d.html")
    html = page.read_text(encoding="utf-8")
    assert "http://" not in html.split("<script>")[0] and "src=" not in html          # no external resources
    snapshot = json.loads(html.split("const SNAP=")[1].split(";\nconst $=")[0])
    for key in (snapshot_key("runs"), snapshot_key("overview", "demo"), snapshot_key("timeline", "demo", "registry")):
        assert key in snapshot


def test_export_cannot_be_broken_by_hostile_run_text(tmp_path):
    run = tmp_path / "runs" / "evil"
    run.mkdir(parents=True)
    (run / "evolution_events.jsonl").write_text("")
    (run / "summary.json").write_text('{"data_tag": "</script><img src=x onerror=alert(1)>", "bad": NaN, "inf": Infinity}')
    html = export_html(tmp_path / "runs", tmp_path / "o.html").read_text(encoding="utf-8")
    assert "</script><img" not in html and "<\\/script>" in html
    snapshot = json.loads(html.split("const SNAP=")[1].split(";\nconst $=")[0])
    overview = snapshot[snapshot_key("overview", "evil")]
    assert overview["summary"]["bad"] is None and overview["summary"]["inf"] is None   # non-finite numbers become null


def test_page_never_assigns_data_through_innerhtml():
    from evox.dashboard.page import PAGE
    assert "innerHTML" not in PAGE and "document.write" not in PAGE and "eval(" not in PAGE


def test_demo_refuses_a_nonempty_directory(tmp_path):
    (tmp_path / "x").write_text("x")
    with pytest.raises(ValueError):
        run_scripted_demo(tmp_path)
