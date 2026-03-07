"""Plotting helpers for generic tabular analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mlcore.validation import ensure_dataframe
from mlcore.tabular.analysis import quartile_summary


def plot_feature_scores(scores: pd.Series, title: str = "Feature scores") -> plt.Figure:
    """Plot feature scores as a bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    scores.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#2c7fb8")
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.set_xlabel("feature")
    ax.tick_params(axis="x", labelrotation=75)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Absolute correlation matrix") -> plt.Figure:
    """Plot correlation matrix with matplotlib."""
    corr = ensure_dataframe(corr_matrix, "corr_matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr.values, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_numeric_distribution_with_quantiles(
    series: pd.Series,
    bins: int = 30,
    title: str | None = None,
) -> plt.Figure:
    """Plot numeric distribution with Q1/Q3 vertical lines."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(numeric, bins=bins, color="#7fc97f", edgecolor="black", alpha=0.85)

    if not numeric.empty:
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        ax.axvline(q1, color="#d95f02", linestyle="--", linewidth=1.5, label="Q1")
        ax.axvline(q3, color="#7570b3", linestyle="--", linewidth=1.5, label="Q3")
        ax.legend()

    ax.set_title(title or f"Distribution: {series.name}")
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def save_numeric_distributions(
    df: pd.DataFrame,
    output_dir: str | Path,
    bins: int = 30,
) -> pd.DataFrame:
    """Save one distribution plot per numeric feature and return quartiles."""
    data = ensure_dataframe(df, "df")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    quartiles = quartile_summary(data, columns=data.columns)
    for feature in quartiles.index.tolist():
        fig = plot_numeric_distribution_with_quantiles(data[feature], bins=bins)
        fig.savefig(output_path / f"{feature}.png", dpi=150)
        plt.close(fig)
    return quartiles

