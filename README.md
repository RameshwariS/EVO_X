# EVO-X

An experimental framework for continual intrusion detection under concept drift.
EVO-X combines streaming IDS models, drift detection and analysis, adaptation
strategies, and a controller that governs candidate promotion in replay experiments.

Phases A and B and the Phase C lifecycle/comparison implementation are available.
Phase C adds candidate evaluation, safety gating, versioned promotion, rollback
and continual comparative replay. Full real-data experiments remain outstanding.
Synthetic runs validate the pipeline; they are not evidence of real-world detection performance.

## Repository layout

```text
src/evox/
  data/          # loading, preprocessing, dataset adapters and streams
  models/        # Random Forest, XGBoost, incremental updates and model loading
  streaming/     # Kafka/in-memory transport, producer, scorer and codec
  drift/         # drift detectors, signatures and scenario generation
  adaptation/    # label policy, memory and candidate strategies
  evolution/     # eligibility rules, utility ranking and proxy evaluation
  lifecycle/     # durable registry, promotion, history and rollback
  evaluation/    # detection, streaming and drift metrics
  experiments/   # executable research workflows
  tools/         # preflight checks, reference sweep and report generation
  cli.py         # unified command line interface
  config.py      # paths and experiment defaults

tests/unit/      # tests grouped by domain
tests/integration/
infra/docker/    # local Kafka configuration
docs/            # guides, architecture, roadmap, research and report drafts
data/            # local raw/processed datasets
results/         # models, metrics, figures, reports and experiment runs
```

See the [architecture guide](docs/architecture.md) and
[command migration table](docs/migration.md) for details.

## Install

Python 3.11 or newer is required. From the repository root:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev,kafka]"
```

With pip and an activated virtual environment, use
`python -m pip install -e ".[dev,kafka]"`. Omit the `kafka` extra for in-memory
experiments. Dependencies, package metadata, CLI registration and pytest settings
are defined in `pyproject.toml`; `requirements.txt` is a compatibility installer.
The existing `.venv` in this checkout has already been installed.

## Run

Use the installed `evox` command after activating the virtual environment, or
invoke the same CLI explicitly:

```powershell
.venv/Scripts/python.exe -m evox --help
.venv/Scripts/python.exe -m evox check-data
.venv/Scripts/python.exe -m evox build-stream --synthetic
.venv/Scripts/python.exe -m evox replay run --synthetic --transport memory --check-equivalence
.venv/Scripts/python.exe -m evox evolve --synthetic --quick
.venv/Scripts/python.exe -m evox lifecycle-demo --output-dir results/runs/phase_c_demo_next
```

Run `evox COMMAND --help` for each workflow's options. The Phase B quick run
covers all six drift types plus a stationary control, uses validation-only weight
selection, and leaves every candidate unpromoted. New Phase B outputs default to
`results/runs/`; specify a fresh `--output-dir` for subsequent runs.

Real experiments require CICIDS2017 CSVs in `data/raw/`, a processed stream, and a
matching Phase A detector configuration. Existing datasets and saved results
were preserved in their original locations during the layout migration.

```powershell
# Optional local Kafka broker
docker compose -f infra/docker/compose.yaml up -d

# Unit tests, then all tests (broker integration skips if no broker is running)
.venv/Scripts/python.exe -m pytest tests/unit
.venv/Scripts/python.exe -m pytest tests
```

## Configuration

Defaults use this checkout's `data/` and `results/` directories. Set
`EVOX_PROJECT_ROOT`, `EVOX_DATA_DIR`, `EVOX_RESULTS_DIR` or `EVOX_KAFKA_BOOTSTRAP`
to override them. `.env.example` documents the variables; export them in your
shell or deployment environment. It is not loaded implicitly.

Load pre-migration joblib wrappers through `evox.models.ids.IDSModel.load()`;
that method resolves the old module names while preserving the saved files.

## Documentation

- [Phase A: streaming and drift laboratory](docs/guides/phase-a.md)
- [Phase B: adaptation and Evolution Controller](docs/guides/phase-b.md)
- [Phase C: safety, lifecycle and comparative experiments](docs/guides/phase-c.md)
- [Repository architecture](docs/architecture.md)
- [Layout and command migration](docs/migration.md)
- [Remaining project roadmap](docs/roadmap.md)
- [Controller design notes](docs/design/controller.md)
- [Report source drafts](docs/reports/)
- [Historical phase plans](docs/archive/phases/)

The repository's CI configuration runs unit tests on Windows/Linux and builds
Python distributions. Full real-data validation and live deployment are outside
the synthetic verification completed here.

Verification: **207 tests passed, 1 broker-dependent test skipped**. The
wheel and source distribution build successfully, the installed wheel runs
outside the checkout, old saved models still load, and the packaged Phase B
smoke run completes all 14 scenarios. Phase C checks cover 70 comparative replays
per classifier plus promote/reject/rollback with exact parent restoration.
Existing dataset/result files were preserved.
