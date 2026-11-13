"""paper-fill must refuse non-reportable runs and read every number from the run files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evox.tools import paper_fill

SYSTEMS = ["static", "always_retrain", "evox", "evox_no_gate"]


def make_run(root: Path, tag="REAL", quick=False, n_seeds=6) -> Path:
    run = root / "run"
    run.mkdir()
    rows = []
    for seed in range(2001, 2001 + n_seeds):
        for kind in ["abrupt", "recurring"]:
            for i, s in enumerate(SYSTEMS):
                rows.append(dict(scenario=f"{kind}_med_tes{seed}", system=s, family="test", seed=seed, kind=kind,
                                 macro_f1=0.70 + 0.05 * i, f1=0.8, fpr=0.05, forgetting=0.02, updates=i, rejected=0,
                                 rollbacks=0, harmful_promotions=0, unnecessary_updates=0, training_seconds=0.1))
    pd.DataFrame(rows).to_csv(run / "per_scenario.csv", index=False)
    (run / "summary.json").write_text(json.dumps(dict(data_tag=tag, quick=quick, algorithm="random_forest")))
    (run / "hypotheses.json").write_text(json.dumps({"H1": dict(claim="c", status="inconclusive", evidence="e")}))
    pd.DataFrame([dict(baseline="static", metric="macro_f1", n_pairs=n_seeds, mean_difference=0.1, p=0.03, holm_p=0.06)]
                 ).to_csv(run / "paired_statistics.csv", index=False)
    return run


def args(run, **kw):
    ns = dict(compare_dir=str(run), detect_dir=None, transfer_dir=None, portability_dir=None)
    ns.update(kw)
    return type("A", (), ns)


def test_refuses_synthetic_quick_and_few_seeds(tmp_path):
    for kw in (dict(tag="SYNTHETIC"), dict(quick=True), dict(n_seeds=5)):
        d = tmp_path / str(sorted(kw))
        d.mkdir()
        problems, _ = paper_fill.check_compare_run(make_run(d, **kw))
        assert problems
    d = tmp_path / "cli"
    d.mkdir()
    out = tmp_path / "o.md"
    rc = paper_fill.main(["--compare-dir", str(make_run(d, tag="SYNTHETIC")), "--output", str(out)])
    assert rc == 1 and not out.exists()


def test_real_full_run_is_reportable_and_numbers_come_from_files(tmp_path):
    run = make_run(tmp_path)
    assert paper_fill.check_compare_run(run)[0] == []
    text, problems = paper_fill.build_results(args(run))
    assert problems == [] and "NOT REPORTABLE" not in text
    assert "0.700 ± 0.000" in text        # static macro-F1 read from per_scenario.csv
    assert "0.850 ± 0.000" in text        # evox_no_gate (i=3): 0.70 + 0.15
    assert "inconclusive" in text and "0.0600" in text   # hypothesis status and Holm p from the run


def test_allow_synthetic_stamps_every_section_and_inject_keeps_original(tmp_path):
    run = make_run(tmp_path, tag="SYNTHETIC", quick=True, n_seeds=1)
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n## 4. Protocol\nx\n## 5. Results\nold [RESULT: x]\n## 6. Discussion\ny\n")
    out = tmp_path / "res.md"
    rc = paper_fill.main(["--compare-dir", str(run), "--output", str(out), "--paper", str(paper), "--allow-synthetic"])
    assert rc == 0
    text = out.read_text()
    assert text.startswith("> **NOT REPORTABLE") and text.count("[NOT REPORTABLE] 5.") >= 4
    filled = (tmp_path / "res_filled.md").read_text()
    assert "old [RESULT: x]" not in filled and "## 6. Discussion" in filled and "## 4. Protocol" in filled
    assert "old [RESULT: x]" in paper.read_text()      # original untouched
