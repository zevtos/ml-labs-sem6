"""Backward-compat shim — import from mlcore.tabular.analysis instead."""

from mlcore.tabular.analysis import (
    MISSING_BIN_LABEL,
    entropy,
    gain_ratio_by_feature,
    gain_ratio_for_targets,
)

# Keep old private name accessible for any external code
_entropy = entropy

__all__ = ["MISSING_BIN_LABEL", "gain_ratio_by_feature", "gain_ratio_for_targets"]

