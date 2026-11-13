import json

from evox.experiments.lifecycle_demo import run_demo


def test_demo_persists_promote_reject_rollback_and_exact_restoration(tmp_path):
    result=run_demo(tmp_path)
    assert result["promoted_version"]=="v2" and result["rejected_version"]=="v3"
    assert result["restored_version"]=="v1" and result["parent_predictions_identical"]
    records=[json.loads(line) for line in (tmp_path/"registry"/"history.jsonl").read_text().splitlines()]
    kinds=[r["kind"] for r in records]
    assert kinds.index("PROMOTED")<kinds.index("REJECTED")<kinds.index("ROLLED_BACK")
