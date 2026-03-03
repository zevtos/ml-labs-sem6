"""Core shared data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """Container for loaded datasets across labs."""

    features: pd.DataFrame | None
    targets: pd.DataFrame | None
    raw: Any

