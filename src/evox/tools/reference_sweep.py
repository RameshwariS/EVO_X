"""
EVO-X Phase 2 (Part B, step 3-4): evidence for the reference-window decision.

For several reference fractions, report which classes are known vs novel and
how much of the live stream is novel-class traffic.  Label-only, so it is
fast even on the full dataset (clean+order runs once).

    python -m evox reference-sweep            # real CSVs, else synthetic
    python -m evox reference-sweep --fractions 0.2 0.4 0.6 0.8
Writes results/metrics/phase2_reference_sweep.csv and results/figures/phase2_reference_sweep.png
"""
import argparse, logging
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evox.config import ensure_output_dirs
from evox.config import BENIGN_LABEL, FIGURES_DIR, LABEL_COLUMN, METRICS_DIR, RAW_DATA_DIR
from evox.data.loading import _find_real_csvs, load_raw_data
from evox.data.stream import prepare_ordered
from evox.data.synthetic import generate_synthetic_week

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def sweep(df: pd.DataFrame, fractions) -> pd.DataFrame:
    labels = df[LABEL_COLUMN].to_numpy()
    all_attacks = sorted(set(labels) - {BENIGN_LABEL})
    rows, n = [], len(df)
    for f in fractions:
        k = int(n * f)
        ref, live = pd.Series(labels[:k]), pd.Series(labels[k:])
        ref_cls = set(ref) - {BENIGN_LABEL}
        live_att = live[live != BENIGN_LABEL]
        novel_flows = int((~live_att.isin(ref_cls)).sum())
        rows.append(dict(
            reference_fraction=f, n_ref=k, n_stream=n - k,
            attack_classes_in_ref=len(ref_cls), attack_classes_total=len(all_attacks),
            novel_attack_classes=len(set(live_att) - ref_cls),
            ref_attack_share=round(float((ref != BENIGN_LABEL).mean()), 4),
            stream_attack_flows=len(live_att),
            stream_novel_attack_flows=novel_flows,
            novel_share_of_stream_attacks=round(novel_flows / max(1, len(live_att)), 4),
            ref_has_no_attacks=len(ref_cls) == 0,
        ))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    ensure_output_dirs()
    if not a.synthetic and _find_real_csvs(RAW_DATA_DIR):
        raw = load_raw_data(add_provenance=True)
    else:
        logging.warning("SYNTHETIC data -- demo only.")
        raw = generate_synthetic_week(60_000)
    df, src = prepare_ordered(raw)
    print(f"ordering source: {src}")
    t = sweep(df, a.fractions)
    print(t.to_string(index=False))
    t.to_csv(METRICS_DIR / "phase2_reference_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t.reference_fraction, t.novel_share_of_stream_attacks, "o-", label="novel share of stream attacks")
    ax.plot(t.reference_fraction, t.attack_classes_in_ref / t.attack_classes_total, "s--", label="share of attack classes seen in reference")
    ax.set_xlabel("reference fraction"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Reference-window sensitivity")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "phase2_reference_sweep.png", dpi=140)


if __name__ == "__main__":
    main()
