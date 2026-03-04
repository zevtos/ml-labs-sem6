"""Dataset registry for reusable loader lookup."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class DatasetRegistry:
    """Small registry to map dataset keys to loader callables."""

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[..., Any]] = {}

    def register(self, key: str, loader: Callable[..., Any]) -> None:
        """Register loader by key."""
        if key in self._loaders:
            raise KeyError(f"Dataset key is already registered: {key}")
        self._loaders[key] = loader

    def get(self, key: str) -> Callable[..., Any]:
        """Get loader callable by key."""
        if key not in self._loaders:
            raise KeyError(f"Unknown dataset key: {key}")
        return self._loaders[key]

    def load(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """Load dataset by key."""
        loader = self.get(key)
        return loader(*args, **kwargs)

    def keys(self) -> list[str]:
        """Return sorted list of registered dataset keys."""
        return sorted(self._loaders.keys())

