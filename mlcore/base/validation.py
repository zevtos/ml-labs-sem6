"""Backward-compat shim — import from mlcore.validation instead."""

from mlcore.validation import ensure_columns_exist, ensure_dataframe, ensure_optional_dataframe

__all__ = ["ensure_columns_exist", "ensure_dataframe", "ensure_optional_dataframe"]

