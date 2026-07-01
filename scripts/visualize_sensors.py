import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("datasets/dataset_v1.csv")

df["lidar_error"] = abs(df["lidar"] - df["true_depth"])
df["ultrasonic_error"] = abs(df["ultrasonic"] - df["true_depth"])
df["radar_error"] = abs(df["radar"] - df["true_depth"])

sns.set_theme(style="whitegrid", context="talk")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#303030",
    "axes.linewidth": 1.1,
    "grid.color": "#d9d9d9",
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.labelsize": 14
})

####################################################
# FIGURE 1: LiDAR Error Under Increasing Turbidity
####################################################

fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="ntu",
    y="lidar_error",
    s=10,
    alpha=0.12,
    color="#ff4d4d",
    edgecolor=None,
    ax=ax
)

sns.regplot(
    data=df,
    x="ntu",
    y="lidar_error",
    scatter=False,
    ci=95,
    color="#8b0000",
    line_kws={"linewidth": 3.5},
    ax=ax
)

ax.set_title("LiDAR Error Under Increasing Turbidity")
ax.set_xlabel("Water Turbidity (NTU)", labelpad=12)
ax.set_ylabel("Absolute Error (cm)", labelpad=12)

sns.despine()
plt.tight_layout()

plt.savefig("figures/figure_lidar.png", dpi=600, bbox_inches="tight")
plt.close()

####################################################
# FIGURE 2: Ultrasonic Error Under Flooded Conditions
####################################################

fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="water_depth",
    y="ultrasonic_error",
    s=10,
    alpha=0.12,
    color="#4f8cff",
    edgecolor=None,
    ax=ax
)

sns.regplot(
    data=df,
    x="water_depth",
    y="ultrasonic_error",
    scatter=False,
    ci=95,
    color="#003399",
    line_kws={"linewidth": 3},
    ax=ax
)

ax.set_title("Ultrasonic Error Under Flooded Conditions")
ax.set_xlabel("Flood Depth (cm)", labelpad=12)
ax.set_ylabel("Absolute Error (cm)", labelpad=12)
ax.set_ylim(0, df["ultrasonic_error"].quantile(0.99))

sns.despine()
plt.tight_layout()

plt.savefig("figures/figure_ultrasonic.png", dpi=600, bbox_inches="tight")
plt.close()

####################################################
# FIGURE 3: Radar Stability Across Flood Depths
####################################################

fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="water_depth",
    y="radar_error",
    s=10,
    alpha=0.12,
    color="#2ca02c",
    edgecolor=None,
    ax=ax
)

sns.regplot(
    data=df,
    x="water_depth",
    y="radar_error",
    scatter=False,
    ci=95,
    color="#006400",
    line_kws={"linewidth": 3},
    ax=ax
)

ax.set_title("Radar Stability Across Flood Depths")
ax.set_xlabel("Flood Depth (cm)", labelpad=12)
ax.set_ylabel("Absolute Error (cm)", labelpad=12)
ax.set_ylim(0, df["radar_error"].quantile(0.995))

sns.despine()
plt.tight_layout()

plt.savefig("figures/figure_radar.png", dpi=600, bbox_inches="tight")
plt.close()

print("FloodTwin-HIL Day 2 Visualizations Generated!")
print("Figures Created : 3")
print()
print("✓ figure_lidar.png")
print("✓ figure_ultrasonic.png")
print("✓ figure_radar.png")