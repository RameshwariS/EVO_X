"""Protect installation boundaries, path configuration and saved-model migration."""
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest

from evox.cli import COMMANDS, main
from evox import __version__
from evox.models.compat import legacy_model_modules
from evox.models.ids import IDSModel, IncrementalForest


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_help_without_executing_a_workflow(command, capsys):
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_installed_cli_works_outside_repository_and_imports_do_not_create_paths(tmp_path):
    workspace = tmp_path / "workspace"
    env = {**os.environ, "EVOX_PROJECT_ROOT": str(workspace), "EVOX_KAFKA_BOOTSTRAP": "broker:1234"}
    for name in ("EVOX_DATA_DIR", "EVOX_RESULTS_DIR"):
        env.pop(name, None)
    command = [sys.executable, "-c", (
        "import json; import evox.config as c; import evox.tools.check_data; "
        "print(json.dumps([str(c.PROJECT_ROOT), str(c.DATA_DIR), c.KAFKA_BOOTSTRAP]))")]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    paths = json.loads(result.stdout)
    assert paths == [str(workspace), str(workspace / "data"), "broker:1234"]
    assert not workspace.exists()
    result = subprocess.run([sys.executable, "-m", "evox", "--version"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == __version__ and not workspace.exists()


def test_old_pickle_class_paths_load_with_identical_predictions(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = IDSModel("random_forest", params=dict(n_estimators=6, n_jobs=1)).fit(X, y)
    model.update(X, y)
    expected = model.predict(X)
    path = tmp_path / "old-model.joblib"
    # Reproduce the exact pre-migration class names in a trusted test artifact.
    with legacy_model_modules(), patch.object(IDSModel, "__module__", "src.ids_model"), \
            patch.object(IncrementalForest, "__module__", "src.ids_model"):
        joblib.dump(model, path)
    restored = IDSModel.load(path)
    np.testing.assert_array_equal(restored.predict(X), expected)
    assert "src.ids_model" not in sys.modules


def test_preflight_missing_data_is_a_return_code_not_an_import_exit(tmp_path, capsys):
    assert main(["check-data", "--raw-dir", str(tmp_path)]) == 1
    assert "No CSVs" in capsys.readouterr().out
