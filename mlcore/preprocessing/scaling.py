"""Feature scaling utilities (pure numpy)."""

from __future__ import annotations

import numpy as np


def standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardization: (x - mean) / std.

    If *mean* and *std* are provided they are reused (transform mode);
    otherwise they are computed from *X* (fit-transform mode).

    Returns (X_scaled, mean, std).
    """
    X = np.asarray(X, dtype=np.float64)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe, mean, std


def normalize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """L2-normalize along *axis* (default: columns)."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def min_max_scale(
    X: np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale features to *feature_range*.

    Returns (X_scaled, mins, maxs).
    """
    X = np.asarray(X, dtype=np.float64)
    if mins is None:
        mins = X.min(axis=0)
    if maxs is None:
        maxs = X.max(axis=0)
    rng = maxs - mins
    rng_safe = np.where(rng == 0, 1.0, rng)
    X_01 = (X - mins) / rng_safe
    lo, hi = feature_range
    return X_01 * (hi - lo) + lo, mins, maxs
