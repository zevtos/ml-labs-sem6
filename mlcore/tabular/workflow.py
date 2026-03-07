"""Composable workflow for tabular feature review tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from mlcore.validation import ensure_columns_exist, ensure_dataframe
from mlcore.tabular.feature_scoring import gain_ratio_for_targets
from mlcore.tabular.plotting import plot_correlation_heatmap, plot_feature_scores, save_numeric_distributions
from mlcore.tabular.preprocessing import drop_rows_where_all_columns_missing, exclude_columns
from mlcore.tabular.statistics import absolute_correlation_matrix, descriptive_statistics, quartile_summary


@dataclass(frozen=True)
class TabularFeatureReviewArtifacts:
    """Structured outputs of the tabular feature review workflow."""

    cleaned_df: pd.DataFrame
    feature_names: list[str]
    score_by_target: pd.DataFrame
    score_mean: pd.Series
    abs_corr_matrix: pd.DataFrame
    descriptive_stats: pd.DataFrame
    quartiles: pd.DataFrame


def run_tabular_feature_review(
    df: pd.DataFrame,
    target_columns: list[str],
    plots_output_dir: str | Path | None = None,
    score_bins: int = 10,
) -> TabularFeatureReviewArtifacts:
    """
    Run reusable tabular feature review steps.

    Steps:
    - normalize missing values and drop rows with all target columns missing
    - keep feature names
    - compute gain-ratio-like scores, abs correlation matrix, descriptive stats
    - optionally save plots
    """
    data = ensure_dataframe(df, "df")
    ensure_columns_exist(data, target_columns, df_name="df")

    cleaned = drop_rows_where_all_columns_missing(data, columns=target_columns)
    feature_names = exclude_columns(cleaned, columns_to_exclude=target_columns)

    score_by_target, score_mean = gain_ratio_for_targets(
        cleaned,
        target_columns=target_columns,
        feature_columns=feature_names,
        bins=score_bins,
    )

    analysis_columns = feature_names + target_columns
    abs_corr = absolute_correlation_matrix(cleaned, columns=analysis_columns)
    stats = descriptive_statistics(cleaned, columns=analysis_columns)
    quartiles = quartile_summary(cleaned, columns=analysis_columns)

    if plots_output_dir is not None:
        out = Path(plots_output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig_score = plot_feature_scores(score_mean, title="Feature importance (gain ratio)")
        fig_score.savefig(out / "feature_scores.png", dpi=150)
        plt.close(fig_score)

        fig_corr = plot_correlation_heatmap(abs_corr)
        fig_corr.savefig(out / "abs_correlation_heatmap.png", dpi=150)
        plt.close(fig_corr)

        save_numeric_distributions(cleaned[analysis_columns], output_dir=out / "distributions")

    return TabularFeatureReviewArtifacts(
        cleaned_df=cleaned,
        feature_names=feature_names,
        score_by_target=score_by_target,
        score_mean=score_mean,
        abs_corr_matrix=abs_corr,
        descriptive_stats=stats,
        quartiles=quartiles,
    )


def artifacts_to_dict(artifacts: TabularFeatureReviewArtifacts) -> dict[str, Any]:
    """Convert dataclass artifacts into serializable mapping."""
    return {
        "cleaned_df": artifacts.cleaned_df,
        "feature_names": artifacts.feature_names,
        "score_by_target": artifacts.score_by_target,
        "score_mean": artifacts.score_mean,
        "abs_corr_matrix": artifacts.abs_corr_matrix,
        "descriptive_stats": artifacts.descriptive_stats,
        "quartiles": artifacts.quartiles,
    }

