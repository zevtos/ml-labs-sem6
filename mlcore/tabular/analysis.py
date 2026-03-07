"""Feature scoring and statistics helpers for tabular datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mlcore.validation import ensure_columns_exist, ensure_dataframe

# ---------------------------------------------------------------------------
# Feature scoring
# ---------------------------------------------------------------------------

MISSING_BIN_LABEL = "__missing__"


def entropy(values: pd.Series) -> float:
    """Compute Shannon entropy of a pandas Series (including NaN as a category)."""
    probs = values.value_counts(normalize=True, dropna=False)
    return float(-(probs * np.log2(probs)).sum())


def _as_discrete(series: pd.Series, bins: int) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        non_missing = series.dropna()
        unique_count = int(non_missing.nunique())
        if unique_count > 1:
            q = min(bins, unique_count)
            discretized = pd.qcut(series, q=q, duplicates="drop")
            return discretized.astype("object").fillna(MISSING_BIN_LABEL).astype(str)
    return series.astype("object").fillna(MISSING_BIN_LABEL).astype(str)


def _information_gain(feature: pd.Series, target: pd.Series) -> tuple[float, float]:
    parent_entropy = entropy(target)
    grouped = pd.DataFrame({"feature": feature, "target": target}).groupby("feature", observed=False)
    n = len(target)
    conditional_entropy = 0.0
    for _, part in grouped:
        weight = len(part) / n
        conditional_entropy += weight * entropy(part["target"])

    info_gain = parent_entropy - conditional_entropy
    split_info = entropy(feature)
    return info_gain, split_info


def gain_ratio_by_feature(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str] | None = None,
    bins: int = 10,
) -> pd.Series:
    """Compute gain ratio score for each feature with respect to one target."""
    validated = ensure_dataframe(df, "df")
    ensure_columns_exist(validated, [target_column], df_name="df")

    if feature_columns is None:
        feature_columns = [column for column in validated.columns if column != target_column]
    ensure_columns_exist(validated, feature_columns, df_name="df")

    target = validated[target_column].astype("object").fillna(MISSING_BIN_LABEL)
    scores: dict[str, float] = {}

    for column in feature_columns:
        feature = _as_discrete(validated[column], bins=bins)
        info_gain, split_info = _information_gain(feature, target)
        if split_info <= 0.0:
            scores[column] = 0.0
        else:
            scores[column] = float(info_gain / split_info)

    return pd.Series(scores, dtype=float).sort_values(ascending=False)


def gain_ratio_for_targets(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    feature_columns: Sequence[str] | None = None,
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute gain ratio per target and aggregated mean score."""
    validated = ensure_dataframe(df, "df")
    ensure_columns_exist(validated, target_columns, df_name="df")

    score_by_target: dict[str, pd.Series] = {}
    for target_column in target_columns:
        score_by_target[target_column] = gain_ratio_by_feature(
            validated,
            target_column=target_column,
            feature_columns=feature_columns,
            bins=bins,
        )

    score_table = pd.DataFrame(score_by_target).fillna(0.0)
    mean_score = score_table.mean(axis=1).sort_values(ascending=False)
    score_table = score_table.reindex(mean_score.index)
    return score_table, mean_score


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


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
