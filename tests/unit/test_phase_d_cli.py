"""Phase D command wiring and the reproducibility manifest."""
import json

import pytest

from evox.cli import COMMANDS, main
from evox.experiments import reproduce


def test_phase_d_commands_are_registered_beside_the_earlier_ones():
    for name in ("generalize", "dashboard", "demo", "reproduce"):
        assert name in COMMANDS
    for name in ("baseline", "build-stream", "static-ids", "replay", "scenarios", "detect", "evolve", "compare", "lifecycle-demo"):
        assert name in COMMANDS                      # nothing from Phases 1-C was displaced


def test_reproduce_dry_run_lists_every_step_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "r"
    assert main(["reproduce", "--dry-run", "--output-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    for step in ("phase_b_evolve", "phase_c_compare", "phase_d_demo", "phase_d_transfer_selfcheck",
                 "phase_d_portability", "phase_d_dashboard_export"):
        assert step in printed
    assert not out.exists()


def test_reproduce_real_variant_needs_both_inputs_and_has_no_synthetic_flag(tmp_path, capsys):
    with pytest.raises(SystemExit):
        reproduce.main(["--stream-dir", "x"])
    assert main(["reproduce", "--dry-run", "--stream-dir", "s", "--detector-config", "d.json", "--output-dir", str(tmp_path / "r")]) == 0
    assert "--synthetic" not in capsys.readouterr().out


def test_reproduce_refuses_nonempty_output(tmp_path):
    (tmp_path / "x").write_text("x")
    assert main(["reproduce", "--output-dir", str(tmp_path)]) == 2


def test_reproduce_records_a_failed_step_in_the_manifest(tmp_path):
    from unittest.mock import patch
    steps = [("only_step", "evox.experiments.demo", ["--output-dir", str(tmp_path / "out" / "demo"), "--no-html"]),
             ("never_runs", "evox.experiments.demo", [])]
    with patch.object(reproduce, "plan", return_value=steps), \
            patch.object(reproduce, "run_step", side_effect=[dict(name="only_step", module="m", argv=[], returncode=3, seconds=0.0)]):
        assert reproduce.main(["--output-dir", str(tmp_path / "out")]) == 1
    manifest = json.loads((tmp_path / "out" / "MANIFEST.json").read_text())
    assert manifest["status"] == "failed" and manifest["failed_step"] == "only_step" and len(manifest["steps"]) == 1
    assert manifest["data_tag"] == "SYNTHETIC" and manifest["test_seeds"] and "pipeline only" in manifest["note"]


def test_artefact_hashes_cover_key_files_and_change_with_content(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "summary.json").write_text("{}")
    (tmp_path / "a" / "ignored.txt").write_text("x")
    first = reproduce.artefact_hashes(tmp_path)
    assert list(first) == ["a/summary.json"]
    (tmp_path / "a" / "summary.json").write_text('{"x": 1}')
    assert reproduce.artefact_hashes(tmp_path) != first
