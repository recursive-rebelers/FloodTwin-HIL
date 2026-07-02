import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

sys.path.append(os.path.abspath("../reliability_engine"))
from reliability_engine.reliability_fusion import compute_adaptive_weights

RESULT_DIR = (BASE_DIR / "results" / "reliability_engine")
DATASET_FILE = (BASE_DIR / "datasets" / "dataset_v2.csv")
os.makedirs(RESULT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_FILE)
weights = []

for _, row in df.iterrows():
    w_lidar, w_ultra, w_radar = (
        compute_adaptive_weights(
            row["R_lidar"],
            row["R_ultrasonic"],
            row["R_radar"]
        )
    )

    weights.append({
        "scenario_id": row["scenario_id"],
        "W_lidar": w_lidar,
        "W_ultrasonic": w_ultra,
        "W_radar": w_radar
    })

weights_df = pd.DataFrame(weights)
weights_df.to_csv(f"{RESULT_DIR}/adaptive_weights.csv", index=False)

summary = {
    "avg_W_lidar": weights_df["W_lidar"].mean(),
    "avg_W_ultrasonic": weights_df["W_ultrasonic"].mean(),
    "avg_W_radar": weights_df["W_radar"].mean(),
    "max_W_lidar": weights_df["W_lidar"].max(),
    "max_W_ultrasonic": weights_df["W_ultrasonic"].max(),
    "max_W_radar": weights_df["W_radar"].max(),
    "std_W_lidar": weights_df["W_lidar"].std(),
    "std_W_ultrasonic": weights_df["W_ultrasonic"].std(),
    "std_W_radar": weights_df["W_radar"].std()
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f"{RESULT_DIR}/fusion_statistics.csv", index=False)

print()
print("=" * 60)
print("FloodTwin-HIL Reliability Validation")

print()
print(summary_df)
print()

print(weights_df.head())
print()
print("=" * 60)