import pandas as pd
from pathlib import Path

from lidar_model import LidarModel
from ultrasonic_model import UltrasonicModel
from radar_model import RadarModel
from imu_model import generate_mpu6050_metrics
from turbidity_model import TurbidityModel
from water_contact_model import FSIR01Model


BASE_DIR = Path(__file__).resolve().parent.parent

SCENARIO_FILE = BASE_DIR / "scenarios" / "scenario_registry.csv"
DATASET_FILE = BASE_DIR / "datasets" / "dataset_v1.csv"

RESULTS_DIR = BASE_DIR / "results"

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

        true_depth = row["pothole_depth"]
        water_depth = row["water_depth"]

        assert water_depth <= true_depth

        true_ntu = row["ntu"]

        ntu = turbidity.generate(true_ntu)

        road_type = row["road_type"]
        road_environment = row["road_environment"]

        weather = row["weather"]
        lighting = row["lighting"]

        speed = row["vehicle_speed"]

        lidar_data = lidar.generate(
            true_depth,
            water_depth,
            ntu
        )

        ultrasonic_data = ultrasonic.generate(
            true_depth,
            water_depth
        )

        radar_data = radar.generate(
            true_depth,
            water_depth,
            ntu
        )

        imu_data = generate_mpu6050_metrics(
            road_type,
            road_environment,
            speed,
            true_depth,
            water_depth
        )

        water_data = water_sensor.detect(
            water_depth
        )

        dataset.append({

            "scenario_id":
                row["scenario_id"],

            "road_environment":
                road_environment,

            "road_type":
                road_type,

            "weather":
                weather,

            "lighting":
                lighting,

            "vehicle_speed":
                speed,

            "true_depth":
                true_depth,

            "water_depth":
                water_depth,

            "true_ntu":
                true_ntu,

            "ntu":
                ntu,

            "severity":
                row["severity"],

            "lidar":
                lidar_data,

            "lidar_error":
                round(lidar_data - true_depth, 3),

            "R_lidar":
                lidar.reliability(water_depth, ntu),

            "ultrasonic":
                ultrasonic_data["distance"],

            "ultrasonic_error":
                round(ultrasonic_data["distance"] - true_depth, 3),

            "surface_echo":
                ultrasonic_data["surface_echo"],

            "bottom_echo":
                ultrasonic_data["bottom_echo"],

            "R_ultrasonic":
                ultrasonic.reliability(water_depth),

            "radar":
                radar_data["distance"],

            "radar_error":
                round(radar_data["distance"] - true_depth, 3),

            "snr":
                radar_data["snr"],

            "rcs":
                radar_data["rcs"],

            "clutter_probability":
                radar_data["clutter_probability"],

            "R_radar":
                radar.reliability(water_depth),

            "imu_pitch":
                imu_data["pitch"],

            "imu_roll":
                imu_data["roll"],

            "imu_acceleration":
                imu_data["acceleration"],

            "R_imu":
                imu_data["reliability"],

            "water_state":
                water_data["state"],

            "R_water":
                water_data["reliability"],

            "water_contact":
                water_data["water_present"]

        })

    df = pd.DataFrame(dataset)

    DATASET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        DATASET_FILE,
        index=False
    )

    numeric_stats = df.describe()
    categorical_stats = df.describe(include=['object'])
    nulls = df.isnull().sum()
    corr = df.corr(numeric_only=True)

    summary = {

        "samples":
            len(df),

        "avg_lidar_error":
            df["lidar_error"].mean(),

        "avg_ultrasonic_error":
            df["ultrasonic_error"].mean(),

        "avg_radar_error":
            df["radar_error"].mean(),

        "avg_R_lidar":
            df["R_lidar"].mean(),

        "avg_R_ultrasonic":
            df["R_ultrasonic"].mean(),

        "avg_R_radar":
            df["R_radar"].mean(),

        "avg_R_imu":
            df["R_imu"].mean(),

        "avg_R_water":
            df["R_water"].mean(),

        "avg_water_depth":
            df["water_depth"].mean(),

        "avg_ntu":
            df["ntu"].mean(),

        "avg_snr":
            df["snr"].mean(),

        "avg_surface_echo":
            df["surface_echo"].mean(),

        "max_water_depth":
            df["water_depth"].max(),

        "max_ntu":
            df["ntu"].max()

    }

    summary_df = pd.DataFrame([summary])
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
    print(df.head())
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

    print()
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()