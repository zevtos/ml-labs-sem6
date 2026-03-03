"""Validation helpers reused across lab modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def ensure_dataframe(value: Any, name: str) -> pd.DataFrame:
    """Validate that value is a pandas DataFrame."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(value).__name__}")
    return value


def ensure_optional_dataframe(value: Any, name: str) -> pd.DataFrame | None:
    """Validate that value is None or pandas DataFrame."""
    if value is None:
        return None
    return ensure_dataframe(value, name)


def ensure_columns_exist(df: pd.DataFrame, columns: Sequence[str], df_name: str = "dataframe") -> None:
    """Validate that all columns exist in dataframe."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {df_name}: {missing}")

