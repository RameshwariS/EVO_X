# Reproducibility checklist

1. Install with `python -m pip install -e ".[dev,kafka]"` (Python 3.11+). Record `pip freeze` with the results.
2. `python -m evox reproduce` must complete on a fresh checkout (synthetic, quick). Its `MANIFEST.json` records
   versions, seeds, git commit and arguments; `SHA256SUMS` records artefact checksums.
3. For real results use fresh output directories, the validation seed family for tuning and the test seed family for
   reporting, at least six test seeds, and archive each run directory (its `source_snapshot.zip` captures the code used).
4. Never report a run whose protocol says `SYNTHETIC` or `quick: true`.
5. Archive the result CSVs (`per_scenario.csv`, `paired_statistics.csv`, `portability.csv`, `decision_agreement.csv`)
   together with `MANIFEST.json` and `preregistered_protocol.json`.
6. Timing columns vary between machines; compare F1, decisions and update counts, not seconds.
