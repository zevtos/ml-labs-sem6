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


def group_shuffle_split(
    *arrays: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """Split arrays so that no group appears in both train and test.

    Useful when samples within the same group are correlated (e.g. multiple
    recordings from the same robot in CareerCon 2019). A random row-level
    split would leak group identity into the test set; this routine instead
    splits at the group level: all samples sharing a *group* go to the same
    side of the split.

    Returns [train_1, test_1, train_2, test_2, ...].
    """
    if not arrays:
        raise ValueError("At least one array is required")
    n = len(arrays[0])
    for arr in arrays:
        if len(arr) != n:
            raise ValueError("All arrays must have the same length")
    groups = np.asarray(groups).ravel()
    if len(groups) != n:
        raise ValueError("Groups must match array length")

    rng = np.random.default_rng(random_state)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_test_groups = max(1, int(round(len(unique_groups) * test_size)))
    test_groups = set(unique_groups[:n_test_groups].tolist())

    test_mask = np.array([g in test_groups for g in groups])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    result: list[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result


def stratified_group_shuffle_split(
    *arrays: np.ndarray,
    groups: np.ndarray,
    stratify: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """Group-aware split that also balances class proportions.

    Splits at the group level (a group goes entirely to train or test) AND
    keeps per-class proportions roughly equal in train and test. Useful when
    groups within a stratum are correlated (CareerCon 2019: each group is
    one surface, so we sample a fixed fraction of groups per surface).
    """
    if not arrays:
        raise ValueError("At least one array is required")
    n = len(arrays[0])
    groups = np.asarray(groups).ravel()
    stratify = np.asarray(stratify).ravel()
    if len(groups) != n or len(stratify) != n:
        raise ValueError("groups / stratify must match array length")

    rng = np.random.default_rng(random_state)

    # group -> its stratum label (groups are assumed to live in exactly one stratum;
    # if not, fall back to the modal stratum so behaviour stays deterministic).
    group_to_stratum: dict = {}
    for g, s in zip(groups, stratify):
        group_to_stratum.setdefault(g, []).append(s)
    group_stratum = {g: max(set(ss), key=ss.count) for g, ss in group_to_stratum.items()}

    test_groups: list = []
    train_groups: list = []
    by_stratum: dict[object, list] = {}
    for g, s in group_stratum.items():
        by_stratum.setdefault(s, []).append(g)

    for s, gs in by_stratum.items():
        gs = list(gs)
        rng.shuffle(gs)
        k = max(1, int(round(len(gs) * test_size))) if len(gs) > 1 else 0
        test_groups.extend(gs[:k])
        train_groups.extend(gs[k:])

    test_set = set(test_groups)
    test_mask = np.array([g in test_set for g in groups])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

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
