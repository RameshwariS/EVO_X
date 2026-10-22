"""
EVO-X Phase 1 — Synthetic CICIDS2017-style Data Generator
============================================================
The real CICIDS2017 dataset (Sharafaldin et al., 2018) is ~7GB of CSVs and
requires a manual download from the University of New Brunswick's CIC
registry (https://www.unb.ca/cic/datasets/ids-2017.html). It cannot be
fetched automatically in this environment.

This module generates a *statistically plausible stand-in* dataset with the
same schema, class list, and rough class-imbalance profile as CICIDS2017, so
that:

  1. The Phase 1 pipeline (data_loader -> preprocessing -> baseline_ids ->
     evaluate) can be run and unit-tested end-to-end right now.
  2. When the real CSVs are dropped into data/raw/, data_loader.py picks
     them up automatically and this module is bypassed entirely.

DO NOT report results produced on this synthetic data as CICIDS2017 findings
in the dissertation — it exists purely to validate the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMN, TIMESTAMP_COLUMN, RANDOM_SEED

# Approximate real CICIDS2017 class list + rough relative prevalence
# (order matters for label encoding reproducibility)
ATTACK_PROFILE = {
    "BENIGN": 0.80,
    "DoS Hulk": 0.08,
    "PortScan": 0.05,
    "DDoS": 0.03,
    "DoS GoldenEye": 0.015,
    "FTP-Patator": 0.01,
    "SSH-Patator": 0.008,
    "DoS slowloris": 0.005,
    "DoS Slowhttptest": 0.0005,
    "Bot": 0.0005,
    "Web Attack Brute Force": 0.0003,
    "Infiltration": 0.0001,
    "Heartbleed": 0.00002,
}

NUMERIC_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Active Mean",
    "Idle Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Average Packet Size",
]


def _simulate_class_features(rng: np.random.Generator, label: str, n: int) -> pd.DataFrame:
    """Draw feature vectors from a class-specific Gaussian-ish generator so
    that classes are separable-but-noisy, similar in spirit to real traffic
    (benign traffic has low/regular volume; attacks show distinct footprints).
    """
    base = {
        "BENIGN": dict(dur=(2000, 800), pkts=(8, 3), bps=(500, 200), syn=0.05),
        "DoS Hulk": dict(dur=(50, 20), pkts=(120, 40), bps=(50000, 15000), syn=0.7),
        "PortScan": dict(dur=(10, 5), pkts=(2, 1), bps=(100, 50), syn=0.95),
        "DDoS": dict(dur=(30, 10), pkts=(200, 60), bps=(80000, 20000), syn=0.8),
        "DoS GoldenEye": dict(dur=(300, 100), pkts=(60, 20), bps=(20000, 6000), syn=0.6),
        "FTP-Patator": dict(dur=(1500, 500), pkts=(15, 5), bps=(1000, 300), syn=0.3),
        "SSH-Patator": dict(dur=(1200, 400), pkts=(12, 4), bps=(900, 250), syn=0.3),
        "DoS slowloris": dict(dur=(5000, 1500), pkts=(5, 2), bps=(200, 80), syn=0.4),
        "DoS Slowhttptest": dict(dur=(4500, 1200), pkts=(6, 2), bps=(250, 90), syn=0.4),
        "Bot": dict(dur=(2500, 900), pkts=(10, 4), bps=(700, 250), syn=0.2),
        "Web Attack Brute Force": dict(dur=(1800, 600), pkts=(20, 6), bps=(1500, 400), syn=0.2),
        "Infiltration": dict(dur=(6000, 2000), pkts=(30, 10), bps=(3000, 900), syn=0.15),
        "Heartbleed": dict(dur=(800, 300), pkts=(4, 2), bps=(150, 60), syn=0.1),
    }[label]

    dur = np.clip(rng.normal(*base["dur"], n), 1, None)
    fwd_pkts = np.clip(rng.normal(base["pkts"][0], base["pkts"][1], n), 0, None)
    bwd_pkts = np.clip(fwd_pkts * rng.uniform(0.3, 1.2, n), 0, None)
    fwd_len = np.clip(rng.normal(base["pkts"][0] * 60, base["pkts"][1] * 20, n), 0, None)
    bwd_len = np.clip(fwd_len * rng.uniform(0.2, 1.0, n), 0, None)
    bps = np.clip(rng.normal(*base["bps"], n), 0, None)
    pps = np.clip(bps / (dur + 1) * 100, 0, None)
    iat_mean = np.clip(rng.normal(dur / (fwd_pkts + 1), 5, n), 0, None)
    iat_std = np.clip(iat_mean * rng.uniform(0.1, 0.6, n), 0, None)
    syn = rng.binomial(1, base["syn"], n)
    ack = rng.binomial(1, 1 - base["syn"] * 0.5, n)
    psh = rng.binomial(1, 0.3, n)
    avg_pkt_size = np.clip((fwd_len + bwd_len) / (fwd_pkts + bwd_pkts + 1), 0, None)

    return pd.DataFrame(
        {
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_len,
            "Total Length of Bwd Packets": bwd_len,
            "Fwd Packet Length Max": fwd_len * rng.uniform(1.0, 1.5, n),
            "Fwd Packet Length Mean": fwd_len / (fwd_pkts + 1),
            "Bwd Packet Length Max": bwd_len * rng.uniform(1.0, 1.5, n),
            "Bwd Packet Length Mean": bwd_len / (bwd_pkts + 1),
            "Flow Bytes/s": bps,
            "Flow Packets/s": pps,
            "Flow IAT Mean": iat_mean,
            "Flow IAT Std": iat_std,
            "Fwd IAT Mean": iat_mean * rng.uniform(0.8, 1.2, n),
            "Bwd IAT Mean": iat_mean * rng.uniform(0.8, 1.2, n),
            "Active Mean": dur * rng.uniform(0.4, 0.9, n),
            "Idle Mean": dur * rng.uniform(0.0, 0.3, n),
            "SYN Flag Count": syn,
            "ACK Flag Count": ack,
            "PSH Flag Count": psh,
            "Average Packet Size": avg_pkt_size,
        }
    )


def generate_synthetic_cicids(
    n_rows: int = 50_000,
    start_time: str = "2017-07-03 09:00:00",
    duration_hours: float = 8.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic, CICIDS2017-schema-compatible flow dataset,
    time-ordered over a single simulated working day (matching the real
    dataset's Monday-Friday working-hours capture structure, Synopsis §7.6).
    """
    rng = np.random.default_rng(seed)
    labels = list(ATTACK_PROFILE.keys())
    probs = np.array(list(ATTACK_PROFILE.values()))
    probs = probs / probs.sum()

    label_counts = rng.multinomial(n_rows, probs)
    frames = []
    for label, count in zip(labels, label_counts):
        if count == 0:
            continue
        feats = _simulate_class_features(rng, label, count)
        feats[LABEL_COLUMN] = label
        frames.append(feats)

    df = pd.concat(frames, ignore_index=True)
    # Shuffle row order FIRST so that timestamp assignment below interleaves
    # classes realistically across the day, instead of each class occupying
    # its own contiguous time block (which would make a chronological
    # train/test split degenerate into a single-class training set).
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Assign each (now-shuffled) row an independent random timestamp within
    # the working day, then sort by it — this interleaves classes across
    # time the way real concurrent network traffic would.
    start = pd.Timestamp(start_time)
    span_seconds = duration_hours * 3600
    offsets = rng.uniform(0, span_seconds, len(df))
    df[TIMESTAMP_COLUMN] = [start + pd.Timedelta(seconds=s) for s in offsets]

    # A touch of realistic missingness / inf values (CICIDS2017 is known for
    # these artefacts — Synopsis §9.3 "Dataset age and known biases")
    bad_idx = rng.choice(df.index, size=max(1, len(df) // 2000), replace=False)
    df.loc[bad_idx, "Flow Bytes/s"] = np.inf
    nan_idx = rng.choice(df.index, size=max(1, len(df) // 2000), replace=False)
    df.loc[nan_idx, "Flow Packets/s"] = np.nan

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return df


if __name__ == "__main__":
    from src.config import RAW_DATA_DIR

    df = generate_synthetic_cicids(n_rows=50_000)
    out_path = RAW_DATA_DIR / "synthetic_cicids2017_day1.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote synthetic dataset: {out_path}  shape={df.shape}")
    print(df[LABEL_COLUMN].value_counts())
