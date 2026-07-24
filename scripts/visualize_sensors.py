import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures/digital_twin", exist_ok=True)
df = pd.read_csv("datasets/dataset_v1.csv")
sns.set_theme(style="whitegrid", context="talk")

df["lidar_error"] = (df["lidar"] - df["true_depth"]).abs()
df["ultrasonic_error"] = (df["ultrasonic"] - df["true_depth"]).abs()
df["radar_error"] = (df["radar"] - df["true_depth"]).abs()

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

def _pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

def _slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    m, _ = np.polyfit(x[mask], y[mask], 1)
    return float(m)

def _add_metrics_box(ax, x, y, x_name, y_name):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    r = _pearson_r(x, y)
    m = _slope(x, y)
    r2 = float(r ** 2) if np.isfinite(r) else np.nan

    y_valid = y[mask]
    mean_v = float(np.mean(y_valid))
    std_v = float(np.std(y_valid))
    median_v = float(np.median(y_valid))
    p95_v = float(np.percentile(y_valid, 95))
    n = int(mask.sum())

    box_text = (
        f"n = {n:,}\n"
        f"Pearson r = {r:.3f}\n"
        f"R² = {r2:.3f}\n"
        f"Slope = {m:.4f} {y_name}/{x_name}\n"
        f"Mean = {mean_v:.3f}\n"
        f"Std = {std_v:.3f}\n"
        f"Median = {median_v:.3f}\n"
        f"P95 = {p95_v:.3f}")

    ax.text(0.035, 0.95,
        box_text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="#4a4a4a",
            alpha=0.93),
        family="monospace")

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
    ax=ax)

sns.regplot(
    data=df,
    x="ntu",
    y="lidar_error",
    scatter=False,
    ci=95,
    robust=True,
    color="#8b0000",
    line_kws={"linewidth": 3.0},
    ax=ax)

ax.set_title("LiDAR Error Under Increasing Turbidity")
ax.set_xlabel("Water Turbidity (NTU)", labelpad=12)
ax.set_ylabel("Absolute Error (cm)", labelpad=12)

_add_metrics_box(ax, df["ntu"], df["lidar_error"], "(NTU)", "(cm)")

sns.despine()
plt.tight_layout()
plt.savefig("figures/digital_twin/figure_lidar.png", dpi=600, bbox_inches="tight")
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
    ax=ax)

sns.regplot(
    data=df,
    x="water_depth",
    y="ultrasonic_error",
    scatter=False,
    ci=95,
    robust=True,
    color="#003399",
    line_kws={"linewidth": 3.0},
    ax=ax)

ax.set_title("Ultrasonic Error Under Flooded Conditions")
ax.set_xlabel("Flood Depth (cm)", labelpad=12)
ax.set_ylabel("Absolute Error (cm)", labelpad=12)
ax.set_ylim(0, df["ultrasonic_error"].quantile(0.99))

_add_metrics_box(ax, df["water_depth"], df["ultrasonic_error"], "(cm)", "(cm)")

sns.despine()
plt.tight_layout()
plt.savefig("figures/digital_twin/figure_ultrasonic.png", dpi=600, bbox_inches="tight")
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
    alpha=0.10,
    color="#2ca02c",
    edgecolor=None,
    ax=ax)

sns.regplot(
    data=df,
    x="water_depth",
    y="radar_error",
    scatter=False,
    ci=95,
    robust=True,
    color="#006400",
    line_kws={"linewidth": 3.0},
    ax=ax)

ax.set_title("Radar Stability Across Flood Depths")
ax.set_xlabel("Flood Depth (cm)", labelpad=12)
ax.set_ylabel("Absolute Error (cm)", labelpad=12)
ax.set_ylim(0, df["radar_error"].quantile(0.995))

_add_metrics_box(ax, df["water_depth"], df["radar_error"], "(cm)", "(cm)")

sns.despine()
plt.tight_layout()
plt.savefig("figures/digital_twin/figure_radar.png", dpi=600, bbox_inches="tight")
plt.close()

print()
print("Digital Twin Visualizations Generated!")
print("Figures Created: 3")
print()
print("✓ figure_lidar.png")
print("✓ figure_ultrasonic.png")
print("✓ figure_radar.png")
print()