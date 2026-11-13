# Layout migration

Install the project once with `python -m pip install -e ".[dev,kafka]"` (or the
equivalent `uv pip install`). The old root scripts have moved into the package.

| Previous command | Current command |
|---|---|
| `python main.py` | `evox baseline` |
| `python phase2_build_stream.py` | `evox build-stream` |
| `python phase3_static_ids.py` | `evox static-ids` |
| `python phase4_replay.py run` | `evox replay run` |
| `python phase5_build_scenarios.py` | `evox scenarios` |
| `python phase6_detect_eval.py` | `evox detect` |
| `python phase_b_evolve.py` | `evox evolve` |
| `python check_raw_data.py` | `evox check-data` |
| `python scripts_compare_reference_fractions.py` | `evox reference-sweep` |
| `python make_phase3_report.py` | `evox report` |
| `docker compose up -d` | `docker compose -f infra/docker/compose.yaml up -d` |

Existing workflow flags are preserved. Each `evox` command can also be written
as `python -m evox COMMAND` or `.venv/Scripts/python.exe -m evox COMMAND`.

Python imports now follow domain packages: `src.ids_model` becomes
`evox.models.ids`, `src.stream_builder` becomes `evox.data.stream`, and
`src.evolution_controller` becomes `evox.evolution.controller`. Tests use the
installed package, not direct imports from `src`.

The current guides are [Phase A](guides/phase-a.md) and [Phase B](guides/phase-b.md).
Older phase plans live in `docs/archive/phases/`; report drafts live in
`docs/reports/`. See [the architecture guide](architecture.md) for the full tree.

Data files and historical results keep their original locations and contents.
New runs default to `results/runs/`, and generated Phase 3 reports now go to
`results/reports/phase3.md`. Saved protocol files are historical records and
have not been rewritten. Load older joblib files through `IDSModel.load()`.
