"""Dataset import adapters for labs, built on top of reusable mlcore loaders."""

from mlcore.base.types import DatasetBundle
from mlcore.data.loaders import (
    load_csv_file,
    load_excel_sheet,
    load_har_dataset_from_directory,
    load_moabb_dataset,
    load_train_target_test_csv,
    load_uci_dataset,
)
from pathlib import Path
from typing import Any


def load_lr1_id_data_mass_18122012(xlsx_path: str | Path, sheet_name: str = "VU") -> DatasetBundle:
    return load_excel_sheet(xlsx_path=xlsx_path, sheet_name=sheet_name)


def load_lr2_chemical_composition_of_ceramic_samples(csv_path: str | Path | None = None) -> DatasetBundle:
    local_csv = (
        Path(csv_path)
        if csv_path is not None
        else Path(__file__).resolve().parent.parent / "labs/lr-2/data/Chemical Composion of Ceramic.csv"
    )
    if local_csv.exists():
        return load_csv_file(local_csv)
    return load_uci_dataset(583)


def load_lr3_online_retail() -> DatasetBundle:
    return load_uci_dataset(352)


def load_lr4_moabb_p300() -> Any:
    return load_moabb_dataset("BNCI2014009")


def load_lr5_human_activity_recognition_using_smartphones(base_dir: str | Path | None = None) -> DatasetBundle:
    local_root = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent.parent / "labs/lr-5/data/har/UCI HAR Dataset"
    )
    if local_root.exists():
        return load_har_dataset_from_directory(local_root)
    return load_uci_dataset(240)


def load_lr6_careercon(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
) -> DatasetBundle:
    return load_train_target_test_csv(x_train_path, y_train_path, x_test_path)


def load_lr7_mushroom() -> DatasetBundle:
    return load_uci_dataset(73)

__all__ = [
    "DatasetBundle",
    "load_lr1_id_data_mass_18122012",
    "load_lr2_chemical_composition_of_ceramic_samples",
    "load_lr3_online_retail",
    "load_lr4_moabb_p300",
    "load_lr5_human_activity_recognition_using_smartphones",
    "load_lr6_careercon",
    "load_lr7_mushroom",
]
