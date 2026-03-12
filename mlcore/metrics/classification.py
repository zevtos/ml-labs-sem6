"""Classification metrics (pure numpy)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix. Rows = true, columns = predicted."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[cls_to_idx[t], cls_to_idx[p]] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def _per_class_pr(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class precision and recall from confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    return prec, rec


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str | None = "macro",
) -> float | np.ndarray:
    """Precision score.

    *average*: ``"macro"``, ``"micro"``, ``"weighted"``, or ``None`` (per-class).
    """
    cm = confusion_matrix(y_true, y_pred)
    return _aggregate(cm, metric="precision", average=average)


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str | None = "macro",
) -> float | np.ndarray:
    """Recall score."""
    cm = confusion_matrix(y_true, y_pred)
    return _aggregate(cm, metric="recall", average=average)


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str | None = "macro",
) -> float | np.ndarray:
    """F1 score (harmonic mean of precision and recall)."""
    cm = confusion_matrix(y_true, y_pred)
    return _aggregate(cm, metric="f1", average=average)


def _aggregate(
    cm: np.ndarray,
    metric: str,
    average: str | None,
) -> float | np.ndarray:
    """Aggregate per-class metrics according to *average* strategy."""
    prec, rec = _per_class_pr(cm)

    if metric == "precision":
        per_class = prec
    elif metric == "recall":
        per_class = rec
    else:  # f1
        denom = prec + rec
        per_class = np.where(denom > 0, 2 * prec * rec / denom, 0.0)

    if average is None:
        return per_class

    support = cm.sum(axis=1).astype(np.float64)

    if average == "micro":
        tp = np.diag(cm).sum()
        total = cm.sum()
        if metric == "precision":
            fp = cm.sum() - np.diag(cm).sum() - (cm.sum(axis=1) - np.diag(cm)).sum()
            return float(tp / total) if total > 0 else 0.0
        # For micro average, precision == recall == accuracy
        return float(tp / total) if total > 0 else 0.0

    if average == "weighted":
        total = support.sum()
        if total == 0:
            return 0.0
        return float(np.sum(per_class * support) / total)

    # macro
    return float(np.mean(per_class))
