"""Lab dataset catalog — convenience loaders for each lab assignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlcore.data.loaders import (
    load_csv_file,
    load_excel_sheet,
    load_har_dataset_from_directory,
    load_moabb_dataset,
    load_train_target_test_csv,
    load_uci_dataset,
)
from mlcore.types import DatasetBundle


def load_lr1_dataset(xlsx_path: str | Path, sheet_name: str = "VU") -> DatasetBundle:
    return load_excel_sheet(xlsx_path=xlsx_path, sheet_name=sheet_name)


def load_lr2_dataset(csv_path: str | Path | None = None) -> DatasetBundle:
    local_csv = (
        Path(csv_path) if csv_path is not None
        else Path(__file__).resolve().parents[2] / "labs/lr-2/data/Chemical Composion of Ceramic.csv"
    )
    if local_csv.exists():
        return load_csv_file(local_csv)
    return load_uci_dataset(583)


def load_lr3_dataset() -> DatasetBundle:
    return load_uci_dataset(352)


def load_lr4_dataset() -> Any:
    return load_moabb_dataset("BNCI2014009")


def load_lr5_dataset(base_dir: str | Path | None = None) -> DatasetBundle:
    local_root = (
        Path(base_dir) if base_dir is not None
        else Path(__file__).resolve().parents[2] / "labs/lr-5/data/har/UCI HAR Dataset"
    )
    if local_root.exists():
        return load_har_dataset_from_directory(local_root)
    return load_uci_dataset(240)


def load_lr6_dataset(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
) -> DatasetBundle:
    return load_train_target_test_csv(x_train_path, y_train_path, x_test_path)


def load_lr7_dataset() -> DatasetBundle:
    return load_uci_dataset(73)


DATASET_CATALOG: dict[str, Any] = {
    "lr1": load_lr1_dataset,
    "lr2": load_lr2_dataset,
    "lr3": load_lr3_dataset,
    "lr4": load_lr4_dataset,
    "lr5": load_lr5_dataset,
    "lr6": load_lr6_dataset,
    "lr7": load_lr7_dataset,
}
