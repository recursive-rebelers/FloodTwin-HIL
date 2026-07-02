import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

os.makedirs("figures/reliability_engine", exist_ok=True)
df = pd.read_csv("datasets/dataset_v2.csv")

numeric_cols = [
    "R_lidar",
    "R_ultrasonic",
    "R_radar",
    "R_imu",
    "R_water",
    "water_depth",
    "ntu",
    "snr"
]

df[numeric_cols] = (df[numeric_cols].apply(pd.to_numeric, errors="coerce"))

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
# FIGURE 1: LiDAR Reliability Under Increasing Turbidity
####################################################

fig, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="ntu",
    y="R_lidar",
    s=8,
    alpha=0.06,
    color="#ff4d4d",
    edgecolor=None,
    ax=ax
)

smooth = lowess(df["R_lidar"], df["ntu"], frac=0.10, return_sorted=True)
ax.plot(smooth[:,0], smooth[:,1], color="#8b0000", linewidth=4)

ax.set_title("LiDAR Reliability Under Increasing Turbidity")
ax.set_xlabel("Water Turbidity (NTU)", labelpad=12)
ax.set_xlim(0,500)
ax.set_ylabel("Reliability Score", labelpad=12)
ax.set_ylim(0.25,0.95)
ax.grid(True, linestyle="--", alpha=0.3)

sns.despine()
plt.tight_layout()

plt.savefig("figures/reliability_engine/reliability_ntu.png", dpi=600, bbox_inches="tight")
plt.close()

####################################################
# FIGURE 2: Reliability Degradation Under Flooded Conditions
####################################################

fig, ax = plt.subplots(figsize=(8, 5))

ultra = (df.groupby("water_depth")["R_ultrasonic"].mean().sort_index())
ultra = ultra.rolling(5, center=True).mean()
ultra = ultra.interpolate()
ultra = ultra.dropna()

radar = (df.groupby("water_depth")["R_radar"].mean().sort_index().rolling(3, center=True).mean())
radar = radar.dropna()

ax.plot(ultra.index, ultra.values, linewidth=3.5, color="#006400", label="Ultrasonic")
ax.plot(radar.index, radar.values, linewidth=3.5, color="#003399", label="Radar")

ax.set_title("Reliability Degradation Under Flooded Conditions")
ax.set_xlabel("Flood Depth (cm)", labelpad=12)
ax.set_xlim(0,20)
ax.set_ylabel("Reliability Score", labelpad=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=12, frameon=True, loc="upper right")

sns.despine()
plt.tight_layout()

plt.savefig("figures/reliability_engine/reliability_depth.png", dpi=600, bbox_inches="tight")
plt.close()

####################################################
# FIGURE 3: Reliability Correlation Matrix
####################################################

fig, ax = plt.subplots(figsize=(10, 8))

corr = df[[
    "R_lidar",
    "R_ultrasonic",
    "R_radar",
    "R_imu",
    "R_water",
    "water_depth",
    "ntu",
    "snr"
]].corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.25,
    square=True,
    cbar_kws={"label": "Correlation"},
    ax=ax
)

ax.set_title("Reliability Correlation Matrix")
plt.tight_layout()

plt.savefig("figures/reliability_engine/reliability_heatmap.png", dpi=600, bbox_inches="tight")
plt.close()

print()

print("FloodTwin-HIL Day 3 Visualizations Generated!")
print("Figures Created : 3")

print()

print("✓ reliability_ntu.png")
print("✓ reliability_depth.png")
print("✓ reliability_heatmap.png")