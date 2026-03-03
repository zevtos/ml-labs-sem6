"""Preprocessing helpers for generic tabular datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mlcore.base.validation import ensure_columns_exist, ensure_dataframe


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank strings with NaN while preserving existing NaN values."""
    normalized = ensure_dataframe(df, "df").copy()
    object_columns = normalized.select_dtypes(include=["object", "string"]).columns
    if len(object_columns) > 0:
        normalized[object_columns] = normalized[object_columns].replace(r"^\s*$", np.nan, regex=True)
    return normalized


def drop_rows_where_all_columns_missing(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Drop rows where all values are missing for the provided columns."""
    cleaned = normalize_missing_values(df)
    ensure_columns_exist(cleaned, columns, df_name="df")
    mask_all_missing = cleaned[list(columns)].isna().all(axis=1)
    return cleaned.loc[~mask_all_missing].copy()


def exclude_columns(df: pd.DataFrame, columns_to_exclude: Sequence[str]) -> list[str]:
    """Return column names excluding the provided list."""
    validated = ensure_dataframe(df, "df")
    ensure_columns_exist(validated, columns_to_exclude, df_name="df")
    excluded = set(columns_to_exclude)
    return [column for column in validated.columns if column not in excluded]

