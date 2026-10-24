"""Single, lazily loaded command line entry point for experiments and tools."""
from __future__ import annotations

import argparse
import importlib
import logging
import sys

from evox import __version__

COMMANDS = {
    "baseline": ("evox.experiments.baseline", "Train and evaluate the original static baseline"),
    "build-stream": ("evox.experiments.build_stream", "Clean data and construct a time-ordered stream"),
    "static-ids": ("evox.experiments.static_ids", "Evaluate static IDS models over the stream"),
    "replay": ("evox.experiments.replay", "Replay batches through memory or Kafka transport"),
    "scenarios": ("evox.experiments.scenarios", "Construct controlled drift scenarios"),
    "detect": ("evox.experiments.detect", "Tune and evaluate drift detectors"),
    "evolve": ("evox.experiments.evolve", "Run the Phase B controller and shadow candidates"),
    "compare": ("evox.experiments.compare", "Run Phase C gated continual replay and baseline comparisons"),
    "lifecycle-demo": ("evox.experiments.lifecycle_demo", "Demonstrate promotion, rejection and rollback"),
    "check-data": ("evox.tools.check_data", "Inspect raw dataset schema and timestamps"),
    "reference-sweep": ("evox.tools.reference_sweep", "Compare reference-window fractions"),
    "report": ("evox.tools.report", "Generate the Phase 3 report from saved metrics"),
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="evox", description="EVO-X experiment and maintenance commands",
        epilog="Run 'evox COMMAND --help' for the selected workflow's options.")
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument("--version", action="version", version=__version__)
    if not args or args[0] not in COMMANDS:
        parser.parse_args(args)
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    module = importlib.import_module(COMMANDS[args[0]][0])
    return int(module.main(args[1:]) or 0)
