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


def load_lr2_chemical_composition_of_ceramic_samples(
    csv_path: str | Path | None = None,
) -> DatasetBundle:
    """
    Load LR-2 dataset: Chemical Composition of Ceramic Samples.

    Prefer local CSV from labs/lr-2/data/ if available, else try UCI (id=583).
    """
    local_csv = (
        Path(csv_path)
        if csv_path is not None
        else Path(__file__).resolve().parent.parent / "labs/lr-2/data/Chemical Composion of Ceramic.csv"
    )

    if local_csv.exists():
        data = pd.read_csv(local_csv)
        return DatasetBundle(features=data, targets=None, raw=data)

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


def load_lr5_human_activity_recognition_using_smartphones(
    base_dir: str | Path | None = None,
) -> DatasetBundle:
    """
    Load LR-5 dataset: Human Activity Recognition Using Smartphones.

    Read local extraction from `labs/lr-5/data/har/UCI HAR Dataset`.
    If not found, fallback to ucimlrepo (id=240).
    """
    root = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent.parent / "labs/lr-5/data/har/UCI HAR Dataset"
    )

    x_train = root / "train/X_train.txt"
    x_test = root / "test/X_test.txt"
    y_train = root / "train/y_train.txt"
    y_test = root / "test/y_test.txt"

    if x_train.exists() and x_test.exists() and y_train.exists() and y_test.exists():
        # load feature names
        features_df = pd.read_csv(root / "features.txt", sep=r"\s+", header=None, names=["idx", "feature"])
        raw_names = features_df["feature"].tolist()
        # make feature names unique to keep pandas happy
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
        return DatasetBundle(features=x, targets=y, raw={"x_train": xtr, "x_test": xte, "y_train": ytr, "y_test": yte})

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
