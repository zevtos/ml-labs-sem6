"""Reusable dataset loading primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo
from moabb import datasets

from mlcore.types import DatasetBundle
from mlcore.validation import ensure_optional_dataframe


def _to_bundle(dataset: Any) -> DatasetBundle:
    """Convert ucimlrepo dataset object to DatasetBundle."""
    x = ensure_optional_dataframe(dataset.data.features, "dataset.data.features")
    y = ensure_optional_dataframe(dataset.data.targets, "dataset.data.targets")
    return DatasetBundle(features=x, targets=y, raw=dataset)


def load_uci_dataset(dataset_id: int) -> DatasetBundle:
    """Load any dataset available via ucimlrepo by dataset id."""
    return _to_bundle(fetch_ucirepo(id=dataset_id))


def load_excel_sheet(xlsx_path: str | Path, sheet_name: str) -> DatasetBundle:
    """Load one worksheet from an Excel file."""
    data = pd.read_excel(Path(xlsx_path), sheet_name=sheet_name)
    return DatasetBundle(features=data, targets=None, raw=data)


def load_csv_file(csv_path: str | Path, **read_csv_kwargs: Any) -> DatasetBundle:
    """Load CSV file into DatasetBundle."""
    data = pd.read_csv(Path(csv_path), **read_csv_kwargs)
    return DatasetBundle(features=data, targets=None, raw=data)


def load_moabb_dataset(dataset_class_name: str = "BNCI2014009"):
    """Instantiate MOABB dataset by class name."""
    if not hasattr(datasets, dataset_class_name):
        raise AttributeError(f"moabb.datasets has no class named {dataset_class_name}")

    dataset_class = getattr(datasets, dataset_class_name)
    return dataset_class()


def load_har_dataset_from_directory(base_dir: str | Path) -> DatasetBundle:
    """Load UCI HAR dataset from extracted directory layout."""
    root = Path(base_dir)

    paths = {
        "x_train": root / "train/X_train.txt",
        "x_test": root / "test/X_test.txt",
        "y_train": root / "train/y_train.txt",
        "y_test": root / "test/y_test.txt",
    }

    if not all(p.exists() for p in paths.values()):
        raise FileNotFoundError(f"HAR directory has incomplete structure: {root}")

    # feature names
    features = pd.read_csv(root / "features.txt", sep=r"\s+", header=None)[1]

    # make duplicate names unique
    col_names = pd.Index(features).to_series().groupby(level=0).cumcount()
    col_names = features + col_names.where(col_names == 0, "_" + col_names.astype(str))

    # load data
    x_train = pd.read_csv(paths["x_train"], sep=r"\s+", header=None, names=col_names)
    x_test = pd.read_csv(paths["x_test"], sep=r"\s+", header=None, names=col_names)
    y_train = pd.read_csv(paths["y_train"], sep=r"\s+", header=None, names=["activity"])
    y_test = pd.read_csv(paths["y_test"], sep=r"\s+", header=None, names=["activity"])

    X = pd.concat([x_train, x_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    return DatasetBundle(
        features=X,
        targets=y,
        raw={"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test},
    )


def load_train_target_test_csv(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
) -> DatasetBundle:
    """Load train/target/test CSV triplet into DatasetBundle."""
    x_train = pd.read_csv(Path(x_train_path))
    y_train = pd.read_csv(Path(y_train_path))
    x_test = pd.read_csv(Path(x_test_path))
    raw = {"x_train": x_train, "y_train": y_train, "x_test": x_test}
    return DatasetBundle(features=x_train, targets=y_train, raw=raw)
