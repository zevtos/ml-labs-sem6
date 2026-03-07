"""Statistics helpers for generic tabular datasets."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from mlcore.validation import ensure_columns_exist, ensure_dataframe


def _numeric_view(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    view = ensure_dataframe(df, "df")
    if columns is not None:
        ensure_columns_exist(view, columns, df_name="df")
        view = view[list(columns)]
    return view.apply(pd.to_numeric, errors="coerce")


def absolute_correlation_matrix(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Compute absolute correlation matrix for selected columns."""
    numeric = _numeric_view(df, columns=columns)
    return numeric.corr().abs()


def quartile_summary(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Return Q1 and Q3 per selected column."""
    numeric = _numeric_view(df, columns=columns)
    quartiles = numeric.quantile([0.25, 0.75]).transpose()
    quartiles.columns = ["q1", "q3"]
    return quartiles


def descriptive_statistics(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Return descriptive statistics table for selected columns."""
    view = ensure_dataframe(df, "df")
    if columns is not None:
        ensure_columns_exist(view, columns, df_name="df")
        view = view[list(columns)]
    return view.describe(include="all").transpose()

