"""
EVO-X Phase 2 (Part B, step 1-2): preflight check of data/raw/ BEFORE the full run.
    python check_raw_data.py
Tells you which CICIDS2017 release you have and whether timestamps will work.
"""
import sys
import pandas as pd
from src.config import LABEL_COLUMN, RAW_DATA_DIR, TIMESTAMP_COLUMN
from src.data_loader import _find_real_csvs
from src.stream_builder import normalize_label, parse_cicids_timestamps

files = _find_real_csvs(RAW_DATA_DIR)
if not files:
    sys.exit(f"No CSVs in {RAW_DATA_DIR}. See PHASE2_PARTB.md step 1.")
print(f"{len(files)} file(s), in the order they will be replayed:")
ok = True
for f in files:
    head = pd.read_csv(f, nrows=2000, low_memory=False, encoding="latin1")
    head.columns = [c.strip() for c in head.columns]
    has_ts, has_label = TIMESTAMP_COLUMN in head.columns, LABEL_COLUMN in head.columns
    line = f"  {f.name:60s} cols={head.shape[1]:3d} Label={'Y' if has_label else 'N'} Timestamp={'Y' if has_ts else 'N'}"
    if has_ts:
        raw = head[TIMESTAMP_COLUMN].astype(str)
        parsed = parse_cicids_timestamps(raw, fix_pm=False)
        line += f"  sample='{raw.iloc[0]}' parsed={parsed.notna().mean():.0%}"
        ok &= parsed.notna().mean() > 0.99
    ok &= has_label
    print(line)
if not any(TIMESTAMP_COLUMN in pd.read_csv(f, nrows=1, encoding="latin1").rename(columns=str.strip).columns for f in files):
    print("\nWARNING: no Timestamp column -> this is the MachineLearningCVE release. Ordering will fall back to "
          "capture-day file order. Prefer GeneratedLabelledFlows (TrafficLabelling).")
print("\nPREFLIGHT:", "OK" if ok else "PROBLEMS FOUND (see above)")
sys.exit(0 if ok else 1)
