"""
EVO-X Phase 2 (Part B, step 1-2): preflight check of data/raw/ BEFORE the full run.
    python -m evox check-data
Tells you which CICIDS2017 release you have and whether timestamps will work.
"""
import argparse
from pathlib import Path
import pandas as pd
from evox.config import LABEL_COLUMN, RAW_DATA_DIR, TIMESTAMP_COLUMN
from evox.data.loading import _find_real_csvs
from evox.data.stream import normalize_label, parse_cicids_timestamps

def check_data(raw_dir: Path) -> int:
    files = _find_real_csvs(raw_dir)
    if not files:
        print(f"No CSVs in {raw_dir}. See docs/archive/phases/PHASE2_PARTB.md.")
        return 1
    print(f"{len(files)} file(s), in the order they will be replayed:")
    ok, any_timestamp = True, False
    for f in files:
        head = pd.read_csv(f, nrows=2000, low_memory=False, encoding="latin1")
        head.columns = [c.strip() for c in head.columns]
        has_ts, has_label = TIMESTAMP_COLUMN in head.columns, LABEL_COLUMN in head.columns
        any_timestamp |= has_ts
        line = f"  {f.name:60s} cols={head.shape[1]:3d} Label={'Y' if has_label else 'N'} Timestamp={'Y' if has_ts else 'N'}"
        if has_ts:
            raw = head[TIMESTAMP_COLUMN].astype(str)
            parsed = parse_cicids_timestamps(raw, fix_pm=False)
            line += f"  sample='{raw.iloc[0]}' parsed={parsed.notna().mean():.0%}"
            ok &= parsed.notna().mean() > 0.99
        ok &= has_label
        print(line)
    if not any_timestamp:
        print("\nWARNING: no Timestamp column -> MachineLearningCVE release. Ordering falls back to "
              "capture-day file order. Prefer GeneratedLabelledFlows (TrafficLabelling).")
    print("\nPREFLIGHT:", "OK" if ok else "PROBLEMS FOUND (see above)")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check raw CICIDS2017 CSV schemas and timestamps")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    return check_data(parser.parse_args(argv).raw_dir)


if __name__ == "__main__":
    raise SystemExit(main())
