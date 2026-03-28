"""Shared utilities (timing, notebook setup, etc.)."""

from mlcore.utils.notebook import find_project_root, lab_paths, setup_plotting
from mlcore.utils.timing import timed, timer

__all__ = ["find_project_root", "lab_paths", "setup_plotting", "timed", "timer"]
