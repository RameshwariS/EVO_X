"""Phase D3: one-command end-to-end run with a reproducibility manifest.

``evox reproduce`` (default: synthetic, quick) chains the existing workflows in a single fresh directory:

    Phase B  evolve            -> frozen controller (validation seeds only)
    Phase C  compare           -> gated continual comparison (test seeds)
    Phase D  demo              -> scripted promote / reject / rollback
    Phase D  generalize transfer / report
    Phase D  dashboard export  -> one offline HTML file

and writes ``MANIFEST.json`` + ``SHA256SUMS`` (package versions, seeds, git commit when available, the exact
arguments, wall time and a checksum of every artefact).  Synthetic output validates the pipeline only and is
never a research result.  With ``--stream-dir`` and ``--detector-config`` it runs the REAL variant of the
Phase B/C part instead (no synthetic fallback).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from evox.config import PROJECT_ROOT, RESULTS_DIR, TEST_SEEDS, VALIDATION_SEEDS

log = logging.getLogger("reproduce")
PACKAGES = ("evo-x", "numpy", "pandas", "scikit-learn", "xgboost", "scipy", "river", "joblib", "psutil", "matplotlib",
            "confluent-kafka")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def git_commit() -> dict:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain")))
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, dirty=None)


def run_step(name: str, module: str, argv: list[str]) -> dict:
    started = time.perf_counter()
    code = int(importlib.import_module(module).main(argv) or 0)
    return dict(name=name, module=module, argv=argv, returncode=code, seconds=round(time.perf_counter() - started, 2))


KEY_FILES = ("summary.json", "per_scenario.csv", "frozen_controller.json", "frozen_gate.json", "hypotheses.json",
             "preregistered_protocol.json", "report.md", "decision_agreement.csv", "dashboard.html")


def artefact_hashes(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name in KEY_FILES:
            out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def plan(a, out: Path) -> list[tuple[str, str, list[str]]]:
    model = a.model
    steps = []
    if a.stream_dir:
        steps.append(("phase_b_evolve", "evox.experiments.evolve",
                      ["--model", model, "--stream-dir", a.stream_dir, "--detector-config", a.detector_config,
                       "--output-dir", str(out / "phase_b")]))
        steps.append(("phase_c_compare", "evox.experiments.compare",
                      ["--model", model, "--stream-dir", a.stream_dir,
                       "--controller-config", str(out / "phase_b" / "frozen_controller.json"),
                       "--output-dir", str(out / "phase_c")]))
        steps.append(("phase_d_dashboard_export", "evox.dashboard.cli",
                      ["export", "--runs-dir", str(out), "--output", str(out / "dashboard.html")]))
        return steps
    quick = ["--synthetic", "--quick", "--model", model]
    steps += [
        ("phase_b_evolve", "evox.experiments.evolve", quick + ["--output-dir", str(out / "phase_b")]),
        ("phase_c_compare", "evox.experiments.compare",
         quick + ["--controller-config", str(out / "phase_b" / "frozen_controller.json"), "--output-dir", str(out / "phase_c")]),
        ("phase_d_demo", "evox.experiments.demo", ["--output-dir", str(out / "demo"), "--no-html"]),
        ("phase_d_transfer_selfcheck", "evox.experiments.generalize",
         ["transfer", "--synthetic", "--source-controller", str(out / "phase_b" / "frozen_controller.json"),
          "--source-model", model, "--target-run", str(out / "phase_c"), "--model", model, "--limit", "2",
          "--output-dir", str(out / "phase_d_transfer")]),
        ("phase_d_portability", "evox.experiments.generalize",
         ["report", "--run", f"SYNTHETIC/{model}={out / 'phase_c'}", "--output-dir", str(out / "phase_d_generalization")]),
        ("phase_d_dashboard_export", "evox.dashboard.cli",
         ["export", "--runs-dir", str(out), "--output", str(out / "dashboard.html")])]
    return steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "runs" / "phase_d_reproduce")
    ap.add_argument("--model", choices=("random_forest", "xgboost"), default="random_forest")
    ap.add_argument("--stream-dir", help="real processed stream (switches to the REAL Phase B/C variant)")
    ap.add_argument("--detector-config", help="matching REAL detector config (required with --stream-dir)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without running it")
    a = ap.parse_args(argv)
    if bool(a.stream_dir) != bool(a.detector_config):
        ap.error("--stream-dir and --detector-config must be given together")
    out = a.output_dir
    steps = plan(a, out)
    if a.dry_run:
        for name, module, args in steps:
            print(f"{name}: python -m {module} {' '.join(args)}")
        return 0
    if out.exists() and any(out.iterdir()):
        print("Output directory must be empty; choose a fresh --output-dir")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    platform=platform.platform(), packages=package_versions(), git=git_commit(),
                    data_tag="REAL" if a.stream_dir else "SYNTHETIC", quick=not a.stream_dir, algorithm=a.model,
                    validation_seeds=list(VALIDATION_SEEDS), test_seeds=list(TEST_SEEDS), steps=[], status="running")
    note = "SYNTHETIC QUICK RUN: validates the pipeline only; not a research result." if not a.stream_dir else \
        "REAL run: see each step's report for its scope and seeds."
    manifest["note"] = note
    failed = None
    for name, module, args in steps:
        log.info("== %s", name)
        try:
            record = run_step(name, module, args)
        except SystemExit as exc:      # argparse errors inside a step
            record = dict(name=name, module=module, argv=args, returncode=int(exc.code or 1), seconds=0.0)
        except Exception as exc:       # keep the manifest even when a step crashes
            record = dict(name=name, module=module, argv=args, returncode=1, seconds=0.0, error=f"{type(exc).__name__}: {exc}")
        manifest["steps"].append(record)
        if record["returncode"] != 0:
            failed = name
            break
    manifest["status"] = "failed" if failed else "completed"
    manifest["failed_step"] = failed
    manifest["artefact_sha256"] = artefact_hashes(out)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    sums = "".join(f"{digest}  {rel}\n" for rel, digest in manifest["artefact_sha256"].items())
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(f"{manifest['status'].upper()}: {note}\nManifest: {out / 'MANIFEST.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
