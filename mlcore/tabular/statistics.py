"""Backward-compat shim — import from mlcore.tabular.analysis instead."""

from mlcore.tabular.analysis import absolute_correlation_matrix, descriptive_statistics, quartile_summary

__all__ = ["absolute_correlation_matrix", "descriptive_statistics", "quartile_summary"]

