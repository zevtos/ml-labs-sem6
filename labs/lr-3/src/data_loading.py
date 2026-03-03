"""Data loading helpers for LR-3 (association rules)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_online_retail() -> tuple[pd.DataFrame, pd.DataFrame | None, Any]:
    """Fetch Online Retail dataset from UCI via ucimlrepo."""
    online_retail = fetch_ucirepo(id=352)

    x = online_retail.data.features
    y = online_retail.data.targets

    if not isinstance(x, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame in online_retail.data.features")
    if y is not None and not isinstance(y, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame or None in online_retail.data.targets")

    return x, y, online_retail


def print_dataset_info(dataset: Any) -> None:
    """Print metadata and variable information for quick inspection."""
    print(dataset.metadata)
    print(dataset.variables)


if __name__ == "__main__":
    x, y, online_retail = load_online_retail()
    print(f"Features shape: {x.shape}")
    if y is not None:
        print(f"Targets shape: {y.shape}")
    print_dataset_info(online_retail)
