"""Split criteria for decision tree construction (pure numpy).

These operate on numpy arrays and are designed for hot-path use
during tree building. For pandas-based feature scoring see
mlcore.tabular.analysis.
"""

from __future__ import annotations

import numpy as np


def entropy(labels: np.ndarray) -> float:
    """Shannon entropy in bits (log2)."""
    labels = np.asarray(labels).ravel()
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def gini(labels: np.ndarray) -> float:
    """Gini impurity: 1 - sum(p_i^2)."""
    labels = np.asarray(labels).ravel()
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs ** 2))


def information_gain(
    parent_labels: np.ndarray,
    children_labels: list[np.ndarray],
    criterion: str = "entropy",
) -> float:
    """Information gain: H(parent) - weighted_sum(H(children)).

    *criterion*: ``"entropy"`` or ``"gini"``.
    """
    criterion_fn = entropy if criterion == "entropy" else gini
    parent_labels = np.asarray(parent_labels).ravel()
    n = len(parent_labels)
    if n == 0:
        return 0.0

    parent_score = criterion_fn(parent_labels)
    child_score = 0.0
    for child in children_labels:
        child = np.asarray(child).ravel()
        weight = len(child) / n
        child_score += weight * criterion_fn(child)

    return parent_score - child_score
