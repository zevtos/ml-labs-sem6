"""Helpers to import datasets for LR-1 ... LR-7."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo


@dataclass(frozen=True)
class DatasetBundle:
    """Common container for dataset loading results."""

    features: pd.DataFrame | None
    targets: pd.DataFrame | None
    raw: Any


def _to_bundle(dataset: Any) -> DatasetBundle:
    """Convert ucimlrepo dataset object to DatasetBundle."""
    x = dataset.data.features
    y = dataset.data.targets

    if x is not None and not isinstance(x, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame or None in dataset.data.features")
    if y is not None and not isinstance(y, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame or None in dataset.data.targets")

    return DatasetBundle(features=x, targets=y, raw=dataset)


def load_lr1_id_data_mass_18122012(xlsx_path: str | Path, sheet_name: str = "VU") -> DatasetBundle:
    """Load LR-1 dataset from local xlsx file (sheet VU by default)."""
    data = pd.read_excel(Path(xlsx_path), sheet_name=sheet_name)
    return DatasetBundle(features=data, targets=None, raw=data)


def load_lr2_chemical_composition_of_ceramic_samples() -> DatasetBundle:
    """Load LR-2 dataset from UCI (id=583)."""
    return _to_bundle(fetch_ucirepo(id=583))


def load_lr3_online_retail() -> DatasetBundle:
    """Load LR-3 dataset from UCI (id=352)."""
    return _to_bundle(fetch_ucirepo(id=352))


def load_lr4_moabb_p300() -> Any:
    """Load LR-4 moabb dataset class for P300 experiments."""
    try:
        from moabb.datasets import BNCI2014009
    except ImportError as exc:
        raise ImportError(
            "moabb is not installed. Install it to use LR-4 dataset import: pip install moabb"
        ) from exc

    return BNCI2014009()


def load_lr5_human_activity_recognition_using_smartphones() -> DatasetBundle:
    """Load LR-5 dataset from UCI (id=240)."""
    return _to_bundle(fetch_ucirepo(id=240))


def load_lr6_careercon(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
) -> DatasetBundle:
    """Load LR-6 Kaggle competition files from local CSV paths."""
    x_train = pd.read_csv(Path(x_train_path))
    y_train = pd.read_csv(Path(y_train_path))
    x_test = pd.read_csv(Path(x_test_path))

    raw = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
    }
    return DatasetBundle(features=x_train, targets=y_train, raw=raw)


def load_lr7_mushroom() -> DatasetBundle:
    """Load LR-7 dataset from UCI (id=73)."""
    return _to_bundle(fetch_ucirepo(id=73))
