"""Shared notebook setup utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib


def find_project_root() -> Path:
    """Walk up from CWD to find the directory containing 'mlcore'."""
    cwd = Path.cwd()
    for p in [cwd, cwd.parent, cwd.parent.parent, cwd.parent.parent.parent]:
        if (p / "mlcore").exists():
            return p
    return cwd


def lab_paths(lab_id: int | str) -> tuple[Path, Path, Path]:
    """Return (root, lab_dir, assets_dir) for a given lab number.

    Creates assets dir if missing.
    """
    root = find_project_root()
    lab_dir = root / f"labs/lr-{lab_id}"
    assets = lab_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return root, lab_dir, assets


def setup_plotting() -> None:
    """Configure matplotlib for non-interactive (notebook) use."""
    matplotlib.use("Agg")
