"""Timing utilities for experiment tracking."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Generator


def timed(func: Any) -> Any:
    """Decorator that prints elapsed wall-clock time after the call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timed] {func.__name__}: {elapsed:.4f}s")
        return result

    return wrapper


@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager that prints elapsed wall-clock time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    tag = f" ({label})" if label else ""
    print(f"[timer]{tag}: {elapsed:.4f}s")
