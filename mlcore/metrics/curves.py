"""ROC curve and AUC computation (pure numpy)."""

from __future__ import annotations

import numpy as np


def roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve (FPR, TPR, thresholds) for binary classification.

    *y_true*: binary labels (0/1).
    *y_scores*: predicted probabilities for the positive class.
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    desc_order = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_order]
    y_true = y_true[desc_order]

    # Distinct thresholds
    distinct = np.concatenate([[True], np.diff(y_scores) != 0])
    tps = np.cumsum(y_true)[distinct]
    fps = np.cumsum(1 - y_true)[distinct]
    thresholds = y_scores[distinct]

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    tpr = np.concatenate([[0], tps / total_pos]) if total_pos > 0 else np.zeros(len(tps) + 1)
    fpr = np.concatenate([[0], fps / total_neg]) if total_neg > 0 else np.zeros(len(fps) + 1)
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])

    return fpr, tpr, thresholds


def auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve via trapezoidal rule."""
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))
