import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def initialize_dataset(output_dir="."):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset_path = output_dir / "dataset_v1.csv"
    metadata_path = output_dir / "metadata.json"
    columns = [
        "scenario_id",
        "road_environment",
        "road_type",
        "weather",
        "lighting",
        "vehicle_speed",
        "true_depth",
        "water_depth",
        "ntu",
        "severity",
        "lidar",
        "lidar_error",
        "lidar_noise",
        "R_lidar",
        "ultrasonic",
        "ultrasonic_error",
        "surface_echo",
        "bottom_echo",
        "R_ultrasonic",
        "radar",
        "radar_error",
        "snr",
        "rcs",
        "R_radar",
        "imu_pitch",
        "imu_roll",
        "imu_acceleration",
        "water_state",
        "water_contact"
    ]

    with open(
            dataset_path,
            "w",
            newline="",
            encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(columns)

    metadata = {
        "dataset_name":
            "FloodTwin-HIL Dataset V1",

        "created":
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "seed":
            SEED,

        "version":
            "1.0",

        "scenario_count":
            10000,

        "digital_twin_ready":
            True,

        "reliability_ready":
            True,

        "fusion_ready":
            True,

        "fault_injection_ready":
            True
    }

    with open(
            metadata_path, "w", encoding="utf-8") as file:

        json.dump(metadata, file, indent=4)

    print()
    print("=" * 60)
    print("FloodTwin-HIL Dataset Initialized")
    print("=" * 60)
    print()
    print(f"Dataset : {dataset_path}")
    print(f"Metadata : {metadata_path}")
    print()
    print(f"Columns Created : {len(columns)}")
    print()
    print("Status : READY")
    print()


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    target_path = (BASE_DIR / "datasets")

    initialize_dataset(output_dir=target_path)