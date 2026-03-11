"""Categorical encoding utilities (pure numpy/pandas)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_encode(series: np.ndarray | pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """Map unique values to integer codes.

    Returns (encoded_array, label_to_int mapping).
    """
    if isinstance(series, pd.Series):
        values = series.to_numpy()
    else:
        values = np.asarray(series)

    categories = sorted(set(values), key=str)
    mapping: dict[str, int] = {str(cat): idx for idx, cat in enumerate(categories)}
    encoded = np.array([mapping[str(v)] for v in values], dtype=np.int64)
    return encoded, mapping


def label_decode(encoded: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    """Inverse of label_encode."""
    inverse = {v: k for k, v in mapping.items()}
    return np.array([inverse[int(code)] for code in encoded])


def one_hot_encode(series: np.ndarray | pd.Series) -> tuple[np.ndarray, list[str]]:
    """One-hot encode categorical values.

    Returns (one_hot_matrix, category_names).
    """
    encoded, mapping = label_encode(series)
    n_categories = len(mapping)
    one_hot = np.zeros((len(encoded), n_categories), dtype=np.float64)
    one_hot[np.arange(len(encoded)), encoded] = 1.0
    category_names = sorted(mapping.keys(), key=lambda k: mapping[k])
    return one_hot, category_names
