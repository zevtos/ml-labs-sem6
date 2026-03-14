"""Model evaluation plots (pure matplotlib)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Annotated heatmap of a confusion matrix."""
    cm = np.asarray(cm)
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float | None = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    """Plot a single ROC curve with optional AUC annotation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    label = f"AUC = {auc:.3f}" if auc is not None else "ROC"
    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    return fig


def plot_multiclass_roc(
    fpr_dict: dict[str, np.ndarray],
    tpr_dict: dict[str, np.ndarray],
    auc_dict: dict[str, float],
    title: str = "Multiclass ROC (One-vs-Rest)",
) -> plt.Figure:
    """Plot one ROC curve per class on a shared axis."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in fpr_dict:
        label = f"{cls} (AUC={auc_dict.get(cls, 0):.3f})"
        ax.plot(fpr_dict[cls], tpr_dict[cls], linewidth=1.5, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize="small")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    return fig
