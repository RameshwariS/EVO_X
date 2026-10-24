"""
EVO-X Phase 1 — Evaluation (Synopsis §8.4, Table 8.2 "Detection" metrics)
============================================================================
Accuracy is deliberately NOT used as the headline metric (Synopsis §8.4:
"a poor indicator under class imbalance and can mask ... forgetting and
false-positive changes"). Instead this module reports, overall and
per-attack-class:

    Precision, Recall, F1, False Positive Rate, False Negative Rate

plus a confusion matrix figure, matching what Experiment 1 (static
baseline) needs to report as the pre-drift reference point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from evox.config import FIGURES_DIR, METRICS_DIR

logger = logging.getLogger(__name__)


def _per_class_fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[int, Dict[str, float]]:
    """One-vs-rest False Positive Rate and False Negative Rate per class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()
    out = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        out[c] = {"FPR": fpr, "FNR": fnr}
    return out


def evaluate_model(model, X_test, y_test, label_encoder, model_name: str) -> Dict:
    y_pred = model.predict(X_test)
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    overall = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    per_class_report = classification_report(
        y_test, y_pred, labels=list(range(n_classes)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    fpr_fnr = _per_class_fpr_fnr(y_test, y_pred, n_classes)
    for i, name in enumerate(class_names):
        per_class_report[name].update(fpr_fnr[i])

    results = {"model": model_name, "overall": overall, "per_class": per_class_report}

    metrics_path = METRICS_DIR / f"{model_name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("Saved metrics to %s", metrics_path)

    _plot_confusion_matrix(y_test, y_pred, class_names, model_name)
    logger.info("[%s] Overall: %s", model_name, {k: round(v, 4) for k, v in overall.items()})
    return results


def _plot_confusion_matrix(y_test, y_pred, class_names, model_name: str) -> Path:
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.7)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalized Confusion Matrix — {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix figure to %s", path)
    return path


def compare_models(results_list) -> pd.DataFrame:
    rows = []
    for r in results_list:
        row = {"model": r["model"]}
        row.update(r["overall"])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    out_path = METRICS_DIR / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    logger.info("Saved model comparison table to %s", out_path)
    return df
