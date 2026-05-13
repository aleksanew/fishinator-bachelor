"""
evaluation.py
Unified evaluation utilities used across all three experiments.
Primary metric: Average Precision (PR-AUC).
Secondary metrics: best-threshold F1, Precision, Recall.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

THRESHOLDS = np.linspace(0.0, 1.0, 50)


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "model":     model_name,
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def best_f1_row(df_thresh: pd.DataFrame) -> pd.Series:
    return df_thresh.loc[df_thresh["f1"].idxmax()]


# ---------------------------------------------------------------------------
# Full model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    verbose: bool = True,
) -> dict:
    """
    Compute AP + best-threshold metrics for one model.
    Returns a dict suitable for building a summary table.
    """
    ap = average_precision_score(y_true, y_prob)
    df_thresh = threshold_sweep(y_true, y_prob, model_name)
    best = best_f1_row(df_thresh)

    if verbose:
        print(f"  {model_name:<15s}  AP={ap:.4f}  "
              f"F1={best['f1']:.3f}  "
              f"P={best['precision']:.3f}  "
              f"R={best['recall']:.3f}  "
              f"@t={best['threshold']:.2f}")

    return {
        "model":     model_name,
        "ap":        ap,
        "f1":        best["f1"],
        "precision": best["precision"],
        "recall":    best["recall"],
        "threshold": best["threshold"],
        "sweep":     df_thresh,
        "y_prob":    y_prob,
        "y_true":    y_true,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pr_curves(results: list, title: str, save_path: str):
    """
    results: list of dicts from evaluate_model()
    """
    plt.figure(figsize=(8, 6))

    pos_rate = results[0]["y_true"].mean()
    plt.axhline(y=pos_rate, color="gray", linestyle="--",
                linewidth=1, label=f"no-skill ({pos_rate:.3f})")

    for r in results:
        precision, recall, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        plt.plot(recall, precision, linewidth=2,
                 label=f"{r['model']} (AP={r['ap']:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_threshold_curves(results: list, title: str, save_path: str):
    plt.figure(figsize=(10, 5))

    for r in results:
        df = r["sweep"]
        plt.plot(df["threshold"], df["precision"],
                 label=f"{r['model']}-precision")
        plt.plot(df["threshold"], df["recall"],
                 linestyle="--", alpha=0.7,
                 label=f"{r['model']}-recall")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary_table(results: list, experiment_name: str):
    print(f"\n{'='*65}")
    print(f"  {experiment_name}")
    print(f"{'='*65}")
    print(f"  {'Model':<15} {'AP':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Thresh':>7}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['model']:<15} "
              f"{r['ap']:>7.4f} "
              f"{r['f1']:>7.4f} "
              f"{r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} "
              f"{r['threshold']:>7.3f}")
    print(f"{'='*65}")