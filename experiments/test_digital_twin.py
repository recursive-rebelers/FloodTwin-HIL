import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from digital_twin.lidar_model import LidarModel
from digital_twin.ultrasonic_model import UltrasonicModel
from digital_twin.radar_model import RadarModel
from digital_twin.imu_model import generate_mpu6050_metrics
from digital_twin.turbidity_model import TurbidityModel
from digital_twin.water_contact_model import FSIR01Model

SCENARIO_FILE = BASE_DIR / "scenarios" / "scenario_registry.csv"
DATASET_FILE = BASE_DIR / "datasets" / "dataset_v1.csv"

RESULTS_DIR = BASE_DIR / "results" / "digital_twin"

NUM_STATS_FILE = RESULTS_DIR / "numerical_statistics.csv"
CAT_STATS_FILE = RESULTS_DIR / "categorical_statistics.csv"
NULL_FILE = RESULTS_DIR / "missing_values.csv"
CORR_FILE = RESULTS_DIR / "correlation_matrix.csv"
SUMMARY_FILE = RESULTS_DIR / "experiment_summary.csv"

def run_pipeline():
    scenarios = pd.read_csv(SCENARIO_FILE)
    dataset = []

    lidar = LidarModel()
    ultrasonic = UltrasonicModel()
    radar = RadarModel()
    turbidity = TurbidityModel()
    water_sensor = FSIR01Model()

    for _, row in scenarios.iterrows():
        true_depth = float(row["pothole_depth"])
        water_depth = float(row["water_depth"])

        if water_depth > true_depth:
            raise ValueError(
                f"Scenario {row['scenario_id']} has water_depth ({water_depth}) "
                f"> pothole_depth ({true_depth})")

        true_ntu = float(row["ntu"])
        ntu = turbidity.generate(true_ntu)

        road_type = row["road_type"]
        road_environment = row["road_environment"]

        weather = row["weather"]
        lighting = row["lighting"]
        speed = float(row["vehicle_speed"])

        lidar_data = lidar.generate(true_depth, water_depth, ntu)
        ultrasonic_data = ultrasonic.generate(true_depth, water_depth)
        radar_data = radar.generate(true_depth, water_depth, ntu)
        imu_data = generate_mpu6050_metrics(road_type, road_environment, speed, true_depth, water_depth)
        water_data = water_sensor.detect(water_depth)
        ultrasonic_error = round(float(ultrasonic_data["distance"] - true_depth), 3)

        dataset.append({
            "scenario_id": row["scenario_id"],
            "road_environment": road_environment,
            "road_type": road_type,
            "weather": weather,
            "lighting": lighting,
            "vehicle_speed": speed,
            "true_depth": true_depth,
            "water_depth": water_depth,
            "true_ntu": true_ntu,
            "ntu": ntu,
            "severity": row["severity"],
            "lidar": lidar_data,
            "lidar_error": round(float(lidar_data - true_depth), 3),
            "R_lidar": lidar.reliability(water_depth, ntu),
            "ultrasonic": ultrasonic_data["distance"],
            "ultrasonic_error": ultrasonic_error,
            "surface_echo": ultrasonic_data["surface_echo"],
            "bottom_echo": ultrasonic_data["bottom_echo"],
            "measurement_type": ultrasonic_data["measurement_type"],
            "bottom_confidence": ultrasonic_data["bottom_confidence"],
            "echo_ambiguity": ultrasonic_data["echo_ambiguity"],
            "R_ultrasonic": ultrasonic.reliability(water_depth,
                measurement_type=ultrasonic_data["measurement_type"],
                surface_echo=ultrasonic_data["surface_echo"],
                bottom_echo=ultrasonic_data["bottom_echo"],
                bottom_confidence=ultrasonic_data["bottom_confidence"],
                echo_ambiguity=ultrasonic_data["echo_ambiguity"],
                ultrasonic_error=ultrasonic_error),
            "radar": radar_data["distance"],
            "radar_error": round(float(radar_data["distance"] - true_depth), 3),
            "snr": radar_data["snr"],
            "rcs": radar_data["rcs"],
            "clutter_probability": radar_data["clutter_probability"],
            "R_radar": radar.reliability(water_depth),
            "imu_pitch": imu_data["pitch"],
            "imu_roll": imu_data["roll"],
            "imu_acceleration": imu_data["acceleration"],
            "R_imu": imu_data["reliability"],
            "water_state": water_data["state"],
            "R_water": water_data["reliability"],
            "water_contact": bool(water_data["water_present"]),
            "surface_sigma": ultrasonic_data["surface_sigma"],
            "bottom_sigma": ultrasonic_data["bottom_sigma"],
        })

    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(dataset)
    df.to_csv(DATASET_FILE, index=False)

    numeric_stats = df.describe().round(4)
    categorical_stats = df.select_dtypes(include=["object", "string"]).describe()

    nulls = df.isnull().sum()
    corr = df.corr(numeric_only=True, method="pearson")

    summary = {
        "samples": len(df),
        "avg_lidar_error": df["lidar_error"].mean(),
        "avg_ultrasonic_error": df["ultrasonic_error"].mean(),
        "avg_radar_error": df["radar_error"].mean(),
        "lidar_rmse": float((df["lidar_error"] ** 2).mean() ** 0.5),
        "ultrasonic_rmse": float((df["ultrasonic_error"] ** 2).mean() ** 0.5),
        "radar_rmse": float((df["radar_error"] ** 2).mean() ** 0.5),
        "avg_R_lidar": df["R_lidar"].mean(),
        "avg_R_ultrasonic": df["R_ultrasonic"].mean(),
        "avg_R_radar": df["R_radar"].mean(),
        "avg_R_imu": df["R_imu"].mean(),
        "avg_R_water": df["R_water"].mean(),
        "avg_water_depth": df["water_depth"].mean(),
        "avg_ntu": df["ntu"].mean(),
        "avg_snr": df["snr"].mean(),
        "avg_surface_echo": df["surface_echo"].mean(),
        "max_water_depth": df["water_depth"].max(),
        "max_ntu": df["ntu"].max(),
        "mixed_echo_ratio": (df["measurement_type"]=="Mixed").mean(),
        "surface_echo_ratio": (df["measurement_type"]=="Surface").mean(),
        "bottom_echo_ratio": (df["measurement_type"]=="Bottom").mean(),
        "avg_bottom_confidence": df["bottom_confidence"].mean(),
        "avg_echo_ambiguity": df["echo_ambiguity"].mean(),
        "avg_surface_sigma": df["surface_sigma"].mean(),
        "avg_bottom_sigma": df["bottom_sigma"].mean(),
    }

    summary_df = pd.DataFrame([summary]).round(5)
    numeric_stats.to_csv(NUM_STATS_FILE)
    categorical_stats.to_csv(CAT_STATS_FILE)
    nulls.to_frame(name="missing_values").to_csv(NULL_FILE)
    corr.to_csv(CORR_FILE)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print("=" * 60)
    print("FloodTwin-HIL Integration Pipeline")
    print("=" * 60)
    print()

    print(f"Scenarios : {len(df)}")
    print(f"Dataset Saved : {DATASET_FILE}")
    print(f"Results Saved : {RESULTS_DIR}")

    print()
    print(df.head().to_string(index=False))
    print()
    print("=" * 60)

    print("\nStatistics\n")
    print(numeric_stats)
    print(categorical_stats)

    print("\nMissing Values\n")
    print(nulls)

    print("\nCorrelation Matrix\n")
    print(corr)

    print("\nExperiment Summary\n")
    print(summary_df)

    print("\nWater State Distribution\n")
    print(df["water_state"].value_counts())

    print("\nSeverity Distribution\n")
    print(df["severity"].value_counts())

    print("\nMixed Echo Statistics\n")
    print(df["measurement_type"].value_counts(normalize=True))

    print()
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()