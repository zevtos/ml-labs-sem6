"""Reusable dataset loading primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo

from mlcore.base.types import DatasetBundle
from mlcore.base.validation import ensure_optional_dataframe


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


def load_moabb_dataset(dataset_class_name: str = "BNCI2014009") -> Any:
    """Instantiate MOABB dataset by class name."""
    try:
        import moabb.datasets as moabb_datasets
    except ImportError as exc:
        raise ImportError(
            "moabb is not installed. Install it to use moabb dataset imports: pip install moabb"
        ) from exc
    if not hasattr(moabb_datasets, dataset_class_name):
        raise AttributeError(f"moabb.datasets has no class named {dataset_class_name}")
    dataset_class = getattr(moabb_datasets, dataset_class_name)
    return dataset_class()


def load_har_dataset_from_directory(
    base_dir: str | Path,
) -> DatasetBundle:
    """Load UCI HAR dataset from extracted directory layout."""
    root = Path(base_dir)

    x_train = root / "train/X_train.txt"
    x_test = root / "test/X_test.txt"
    y_train = root / "train/y_train.txt"
    y_test = root / "test/y_test.txt"

    if not (x_train.exists() and x_test.exists() and y_train.exists() and y_test.exists()):
        raise FileNotFoundError(f"HAR directory has incomplete structure: {root}")

    features_df = pd.read_csv(root / "features.txt", sep=r"\s+", header=None, names=["idx", "feature"])
    raw_names = features_df["feature"].tolist()

    counts: dict[str, int] = {}
    col_names: list[str] = []
    for name in raw_names:
        if name not in counts:
            counts[name] = 0
            col_names.append(name)
        else:
            counts[name] += 1
            col_names.append(f"{name}_{counts[name]}")

    xtr = pd.read_csv(x_train, sep=r"\s+", header=None, names=col_names)
    xte = pd.read_csv(x_test, sep=r"\s+", header=None, names=col_names)
    ytr = pd.read_csv(y_train, sep=r"\s+", header=None, names=["activity"])
    yte = pd.read_csv(y_test, sep=r"\s+", header=None, names=["activity"])

    x = pd.concat([xtr, xte], ignore_index=True)
    y = pd.concat([ytr, yte], ignore_index=True)
    raw = {"x_train": xtr, "x_test": xte, "y_train": ytr, "y_test": yte}
    return DatasetBundle(features=x, targets=y, raw=raw)


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
