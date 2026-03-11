"""Train/test splitting utilities (pure numpy)."""

from __future__ import annotations

import numpy as np


def train_test_split(
    *arrays: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
    stratify: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Split arrays into train and test subsets.

    Returns [train_1, test_1, train_2, test_2, ...].
    """
    if not arrays:
        raise ValueError("At least one array is required")

    n = len(arrays[0])
    for arr in arrays:
        if len(arr) != n:
            raise ValueError("All arrays must have the same length")

    rng = np.random.default_rng(random_state)
    n_test = max(1, int(round(n * test_size)))

    if stratify is not None:
        train_idx, test_idx = _stratified_split(stratify, n_test, rng, shuffle)
    elif shuffle:
        indices = rng.permutation(n)
        train_idx, test_idx = indices[n_test:], indices[:n_test]
    else:
        train_idx = np.arange(0, n - n_test)
        test_idx = np.arange(n - n_test, n)

    result: list[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result


def _stratified_split(
    labels: np.ndarray,
    n_test: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Split preserving class proportions."""
    labels = np.asarray(labels).ravel()
    classes, counts = np.unique(labels, return_counts=True)
    n = len(labels)

    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for cls, count in zip(classes, counts):
        cls_indices = np.where(labels == cls)[0]
        if shuffle:
            rng.shuffle(cls_indices)
        n_test_cls = max(1, round(count * n_test / n))
        test_indices.append(cls_indices[:n_test_cls])
        train_indices.append(cls_indices[n_test_cls:])

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)

    if shuffle:
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    return train_idx, test_idx
