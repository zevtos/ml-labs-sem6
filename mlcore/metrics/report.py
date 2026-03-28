"""High-level evaluation reporting helpers.

Combines metrics computation, figure generation, and saving
in single calls to reduce per-notebook boilerplate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mlcore.metrics.classification import accuracy, confusion_matrix, f1_score, precision, recall
from mlcore.metrics.curves import auc_score, roc_curve
from mlcore.plotting.evaluation import plot_confusion_matrix, plot_multiclass_roc, plot_roc_curve


def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> dict[str, float]:
    """Compute and print classification metrics. Returns the dict."""
    metrics = {
        "Accuracy": accuracy(y_true, y_pred),
        "Precision": precision(y_true, y_pred, average=average),
        "Recall": recall(y_true, y_pred, average=average),
        "F1-score": f1_score(y_true, y_pred, average=average),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> plt.Figure:
    """Compute, plot, save, and return confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig = plot_confusion_matrix(cm, class_names=class_names)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def save_binary_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str | Path,
) -> float:
    """Compute ROC, plot, save, and return AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = auc_score(fpr, tpr)
    fig = plot_roc_curve(fpr, tpr, auc=auc)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"AUC: {auc:.4f}")
    return auc


def save_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> dict[str, float]:
    """Compute per-class OvR ROC, plot, save, and return AUC dict."""
    classes = sorted(np.unique(y_true))
    fpr_dict: dict[str, np.ndarray] = {}
    tpr_dict: dict[str, np.ndarray] = {}
    auc_dict: dict[str, float] = {}

    for i, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        name = class_names[i] if i < len(class_names) else str(cls)
        fpr_dict[name] = fpr
        tpr_dict[name] = tpr
        auc_dict[name] = auc_score(fpr, tpr)

    fig = plot_multiclass_roc(fpr_dict, tpr_dict, auc_dict)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return auc_dict


def save_loss_curve(
    loss_history: list[float] | dict[str, list[float]],
    save_path: str | Path,
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    title: str = "Training Loss",
) -> plt.Figure:
    """Plot and save training loss curve(s).

    *loss_history*: single list or dict of {label: list} for multi-line.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if isinstance(loss_history, dict):
        for label, hist in loss_history.items():
            ax.plot(hist, label=label, alpha=0.7)
        ax.legend(fontsize="small")
    else:
        ax.plot(loss_history)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
