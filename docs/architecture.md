# Repository architecture

EVO-X is an installable Python package with a `src/` layout. Application code
imports `evox.*`; experiment runners compose the package rather than importing
files from the repository root. Paths are explicit and configurable, so the CLI
also works outside the checkout's working directory.

```text
EVO_X/
├── pyproject.toml              # metadata, dependencies, CLI, pytest configuration
├── requirements.txt            # compatibility installer for the development extras
├── .env.example                # environment variable reference (not auto-loaded)
├── .github/workflows/tests.yml # Windows/Linux unit tests and distribution builds
├── src/evox/
│   ├── cli.py                  # lazy command routing
│   ├── config.py               # paths and experiment defaults
│   ├── data/                   # loading, adapters, preprocessing, streams, synthetic data
│   ├── models/                 # base classifiers, update interface, artifact compatibility
│   ├── streaming/              # codec, transports, producer, consumer, replay pipeline
│   ├── drift/                  # detectors, analyzer, scenario generator
│   ├── adaptation/             # label policy, bounded memory, candidate strategies
│   ├── evolution/              # controller, proxy evaluation and Safety Gate
│   ├── lifecycle/              # versioned artifacts, event journal and rollback
│   ├── evaluation/             # baseline, stream and detector metrics/reporting
│   ├── experiments/            # executable workflows for Phases 1–B
│   └── tools/                  # data preflight, reference sweep, report generation
├── tests/
│   ├── unit/                   # mirrors package domains
│   └── integration/            # real broker integration tests
├── infra/docker/compose.yaml   # local Kafka broker
├── docs/
│   ├── guides/                 # current Phase A/B instructions
│   ├── design/                 # controller design notes
│   ├── research/               # citation investigation
│   ├── reports/                # report source drafts
│   ├── archive/                # prior phase plans and historical README
│   └── roadmap.md              # Phases A–D plan
├── data/{raw,processed}/       # local datasets; excluded from package distributions
└── results/                    # existing results preserved in their original paths
    ├── {models,metrics,figures}/
    ├── reports/                # generated report output
    └── runs/                   # default destination for new Phase B runs
```

## Responsibilities

`data`, `models`, `streaming`, `drift`, `adaptation`, `evolution` and `evaluation`
contain reusable domain code. `experiments` is the orchestration boundary for
experiment-specific data preparation, sweeps and output files. `tools` contains
maintenance/report workflows. Importing configuration and preflight tools does
not create directories or read the raw dataset; writing commands create their
own output paths when they execute.

The CLI is the supported workflow interface. For example, `evox evolve` runs
Phase B shadow experiments, `evox compare` runs Phase C continual comparisons,
and `evox replay run` uses the transport pipeline.
`python -m evox` invokes exactly the same entry point. No `sys.path` modification
or repository-root Python wrapper scripts are needed.

## Configuration and outputs

`EVOX_PROJECT_ROOT` sets the workspace root. In a source checkout the default is
the checkout root; a wheel installation defaults to the current directory.
`EVOX_DATA_DIR`, `EVOX_RESULTS_DIR` and `EVOX_KAFKA_BOOTSTRAP` override individual
locations. Environment variables must be exported by the shell/deployment;
`.env.example` is documentation, not an implicit configuration loader.

Historical experiment artifacts remain unchanged under `results/`. Their saved
protocols intentionally retain their original commands/paths as provenance.
New Phase B runs default to `results/runs/`; explicit `--output-dir` still works.
The report command writes generated text to `results/reports/phase3.md` rather
than modifying report source drafts in `docs/`.

## Existing saved models

The old module path `src.ids_model` is resolved temporarily inside
`evox.models.ids.IDSModel.load()`. Both old Random Forest/incremental ensembles
and XGBoost wrappers can therefore be loaded without rewriting their joblib
files. New saves use `evox.models.ids`. Use the model's `load()` method for old
artifacts; directly calling `joblib.load()` bypasses the compatibility mapping.

## Extending the project

Add reusable features to their domain package and keep file-writing experiment
logic in `experiments`. Add tests to the matching `tests/unit` group; mark tests
that require external services as integration tests. Phase C evaluation, safety
and registry code lives in the evaluation, evolution and lifecycle packages.
Future dashboard/service code can be added when implemented.
