"""CLI runner for Lab 1 analysis using reusable mlcore modules."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlcore.data.loaders import load_excel_sheet
from mlcore.tabular.workflow import run_tabular_feature_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tabular feature review for LR1.")
    parser.add_argument("--xlsx-path", required=True, help="Path to ID_data_mass_18122012 xlsx file.")
    parser.add_argument("--sheet", default="VU", help="Excel sheet name (default: VU).")
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Target columns, for example: G_total KGF",
    )
    parser.add_argument(
        "--plots-dir",
        default="labs/lr-1/assets",
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="labs/lr-1/report",
        help="Directory for CSV artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_excel_sheet(args.xlsx_path, sheet_name=args.sheet)

    artifacts = run_tabular_feature_review(
        bundle.features,
        target_columns=args.targets,
        plots_output_dir=args.plots_dir,
    )

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifacts.score_by_target.to_csv(artifacts_dir / "gain_ratio_by_target.csv")
    artifacts.score_mean.to_csv(artifacts_dir / "gain_ratio_mean.csv", header=["gain_ratio"])
    artifacts.abs_corr_matrix.to_csv(artifacts_dir / "abs_corr_matrix.csv")
    artifacts.descriptive_stats.to_csv(artifacts_dir / "descriptive_stats.csv")
    artifacts.quartiles.to_csv(artifacts_dir / "quartiles.csv")

    print("LR1 analysis completed.")
    print(f"Rows after cleanup: {len(artifacts.cleaned_df)}")
    print(f"Feature count: {len(artifacts.feature_names)}")
    print(f"Artifacts saved to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
