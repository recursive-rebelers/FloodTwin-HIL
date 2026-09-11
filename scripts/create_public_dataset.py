import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

PUBLIC_COLUMNS: tuple[str, ...] = (
    # Scenario / Environment
    "scenario_id",
    "road_environment",
    "road_type",
    "weather",
    "lighting",
    "vehicle_speed",
    "true_depth",
    "water_depth",
    "true_ntu",
    "ntu",
    "severity_level",
    "severity_score",

    # Primary Sensor Observations
    "lidar",
    "ultrasonic",
    "surface_echo",
    "bottom_echo",
    "radar",
    "snr",
    "rcs",
    "clutter_probability",
    "imu_pitch",
    "imu_roll",
    "imu_acceleration",
    "water_state",
    "water_contact",

    # Sensor Reliability
    "R_lidar",
    "R_ultrasonic",
    "R_radar",
    "R_imu",
    "R_water",

    # Context / Agreement Features
    "scene_complexity",
    "measurement_spread",
    "measurement_consistency",
    "sensor_agreement",

    # Adaptive Fusion Outputs
    "W_lidar",
    "W_ultrasonic",
    "W_radar",
    "fusion_entropy",
    "fusion_confidence",
    "effective_sensor_count",
    "dominant_sensor",
    "dominance_ratio",
    "water_depth_estimate",
    "fusion_depth_linear",
)

DATASET_DESCRIPTION = (
    "Public derivative of the complete research framework dataset. "
    "Contains 30,000 scenario-level records and a curated subset of "
    "scenario context, heterogeneous sensor observations, reliability "
    "indicators, fusion diagnostics, and evaluation targets."
    "The dataset is derived from the master research dataset by "
    "column selection only. It is not a separately regenerated benchmark "
    "and should not be treated as a replacement for the master dataset.")

def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Create the public derivative of dataset_v2.csv")
    parser.add_argument("--input",
        type=Path, default=Path(base / "datasets" / "dataset_v2.csv"),
        help="Path to the authoritative master dataset")
    parser.add_argument("--output",
        type=Path, default=Path(base / "datasets" / "public_dataset.csv"),
        help="Path for the filtered public dataset")
    parser.add_argument("--metadata",
        type=Path, default=Path(base / "datasets" / "public_dataset_metadata.json"),
        help="Path for the generated metadata manifest")
    return parser.parse_args()

def validate_source(df: pd.DataFrame, input_path: Path) -> None:
    missing = [c for c in PUBLIC_COLUMNS if c not in df.columns]
    if missing: raise ValueError(
        "The input dataset is missing required columns:\n" + "\n".join(f"  - {c}" for c in missing))
    if "scenario_id" not in df.columns:
        raise ValueError("The source dataset must contain scenario_id")
    if df["scenario_id"].isna().any():
        raise ValueError("scenario_id contains missing values")
    if df["scenario_id"].duplicated().any():
        dup_count = int(df["scenario_id"].duplicated().sum())
        raise ValueError(f"scenario_id is not unique ({dup_count} duplicate rows found)")

    selected_nulls = df.loc[:, PUBLIC_COLUMNS].isna().sum()
    problematic = selected_nulls[selected_nulls > 0]
    if not problematic.empty:
        details = "\n".join(f"  - {col}: {int(n)} missing" for col, n in problematic.items())
        raise ValueError("Selected public columns contain missing values:\n" + details)

def build_public_dataset(df: pd.DataFrame) -> pd.DataFrame:
    public_df = df.loc[:, PUBLIC_COLUMNS].copy()
    public_df.reset_index(drop=True, inplace=True)
    return public_df

def build_metadata(source: pd.DataFrame, public_df: pd.DataFrame,
    input_path: Path, output_path: Path,) -> dict:
    categorical = [c for c in PUBLIC_COLUMNS
        if public_df[c].dtype == "object" or str(public_df[c].dtype) == "bool"]

    return {
        "dataset_name": "FloodTwin-HIL Public Dataset",
        "dataset_version": "v3-public",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": DATASET_DESCRIPTION,
        "source_dataset": input_path.name,
        "output_dataset": output_path.name,
        "derivation": ("Column-wise projection from the authoritative master dataset. "
            "No rows were generated, sampled, shuffled, altered, or removed."),

        "rows": int(len(public_df)),
        "source_columns": int(len(source.columns)),
        "public_columns": int(len(public_df.columns)),
        "scenario_id_unique": bool(public_df["scenario_id"].is_unique),
        "missing_values_in_public_columns": int(public_df.isna().sum().sum()),
        "public_columns_list": list(PUBLIC_COLUMNS),
        "categorical_columns": categorical,

        "notes": [
            "true_depth is retained as the supervised evaluation target.",
            "Per-sensor error columns are intentionally omitted to reduce "
            "target leakage for users building independent models.",
            "Internal *_base diagnostics, oracle error fields, and "
            "clean-benchmark availability flags are omitted.",
        ],
    }

def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input.resolve()}")

    df = pd.read_csv(args.input, low_memory=False)
    validate_source(df, args.input)
    public_df = build_public_dataset(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    public_df.to_csv(args.output, index=False, encoding="utf-8", float_format=None)

    metadata = build_metadata(df, public_df, args.input, args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("FloodTwin-HIL Public Dataset Export")
    print("=" * 60)
    print(f"Source Dataset : {args.input}")
    print(f"Source Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Public Dataset : {args.output}")
    print(f"Public Shape   : "
        f"{public_df.shape[0]:,} rows × {public_df.shape[1]} columns")
    print(f"Metadata       : {args.metadata}")
    print(f"Unique IDs     : {public_df['scenario_id'].nunique():,}")
    print(f"Missing Values : {int(public_df.isna().sum().sum()):,}")
    print("=" * 60)
    print("Export Completed Successfully!")
    print()

if __name__ == "__main__":
    main()