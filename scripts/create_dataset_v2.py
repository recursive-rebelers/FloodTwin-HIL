import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reliability_engine.lidar_reliability import LidarReliabilityModel
from reliability_engine.ultrasonic_reliability import UltrasonicReliabilityModel
from reliability_engine.radar_reliability import RadarReliabilityModel

DATASET_V1 = (BASE_DIR / "datasets" / "dataset_v1.csv")
DATASET_V2 = ( BASE_DIR / "datasets" / "dataset_v2.csv")
RESULTS_DIR = ( BASE_DIR / "results" / "reliability_engine")
SUMMARY_FILE = (RESULTS_DIR / "reliability_summary.csv")

def generate_reliability_dataset():
    df = pd.read_csv(DATASET_V1)
    lidar_model = LidarReliabilityModel()
    ultrasonic_model = (UltrasonicReliabilityModel())
    radar_model = (RadarReliabilityModel())
    noise_sigma = (0.5 + 0.002 * df["ntu"])

    r_lidar = (lidar_model.compute_reliability(ntu=df["ntu"], 
    water_depth=df["water_depth"], 
    noise_sigma=noise_sigma))

    r_ultrasonic = (ultrasonic_model.compute_reliability(water_depth=df["water_depth"],
    surface_echo_prob=df["surface_echo"],
    bottom_echo_prob=df["bottom_echo"]))

    clutter = (df["clutter_probability"]
        if "clutter_probability" in df.columns
        else (0.05 * df["water_depth"])
        )

    r_radar = (radar_model.compute_reliability(snr=df["snr"],
    clutter=clutter,
    water_depth=df["water_depth"]))

    df["R_lidar"] = r_lidar
    df["R_ultrasonic"] = r_ultrasonic
    df["R_radar"] = r_radar

    if "R_imu" not in df.columns:
        df["R_imu"] = 0.96

    if "R_water" not in df.columns:
        df["R_water"] = 0.95

    DATASET_V2.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_V2,index=False)

    summary = {
        "samples": len(df),
        "avg_R_lidar": df["R_lidar"].mean(),
        "avg_R_ultrasonic": df["R_ultrasonic"].mean(),
        "avg_R_radar": df["R_radar"].mean(),
        "avg_R_imu": df["R_imu"].mean(),
        "avg_R_water": df["R_water"].mean(),
        "std_R_lidar": df["R_lidar"].std(),
        "std_R_ultrasonic": df["R_ultrasonic"].std(),
        "std_R_radar": df["R_radar"].std(),
        "std_R_imu": df["R_imu"].std(),
        "std_R_water": df["R_water"].std()
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print("FloodTwin-HIL Reliability Dataset")
    print()

    print(f"Samples : {len(df)}")
    print(f"Dataset Saved : {DATASET_V2}")
    print(f"Summary Saved : {SUMMARY_FILE}")

    print()
    print(summary_df)
    print()

if __name__ == "__main__":
    generate_reliability_dataset()