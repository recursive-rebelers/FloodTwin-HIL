import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    "snr" ]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

sns.set_theme(style="whitegrid", context="talk")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbfb",
    "axes.edgecolor": "#2f2f2f",
    "axes.linewidth": 1.0,
    "grid.color": "#d9d9d9",
    "grid.alpha": 0.55,
    "grid.linestyle": "--",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

def _lowess_curve(x, y, frac=0.12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10:
        return None
    order = np.argsort(x)
    return lowess(y[order], x[order], frac=frac, return_sorted=True)

def _binned_mean(df_in, xcol, ycol, bins=30):
    temp = df_in[[xcol, ycol]].dropna().copy()
    if temp.empty:
        return temp

    x = temp[xcol].astype(float)
    y = temp[ycol].astype(float)

    try:
        bins_obj = pd.cut(x, bins=bins, duplicates="drop")
        out = temp.groupby(bins_obj, observed=True).agg({xcol: "mean", ycol: "mean"}).dropna()
        out = out.reset_index(drop=True).sort_values(xcol)
        return out

    except Exception:
        return temp.sort_values(xcol)

####################################################
# FIGURE 1: LiDAR Reliability Under Increasing Turbidity
####################################################

fig, ax = plt.subplots(figsize=(8.4, 5.4))

sns.scatterplot(
    data=df,
    x="ntu",
    y="R_lidar",
    s=10,
    alpha=0.08,
    color="#c62828",
    edgecolor=None,
    ax=ax)

smooth = _lowess_curve(df["ntu"], df["R_lidar"], frac=0.12)
if smooth is not None:
    ax.plot(
        smooth[:, 0],
        smooth[:, 1],
        color="#7f0000",
        linewidth=3.5,
        label="LOWESS Trend")

binned = _binned_mean(df, "ntu", "R_lidar", bins=28)
if not binned.empty:
    ax.plot(
        binned["ntu"],
        binned["R_lidar"],
        color="#ff7043",
        linewidth=2.2,
        alpha=0.95,
        label="Binned Mean")

ax.set_title("LiDAR Reliability Under Increasing Turbidity")
ax.set_xlabel("Water Turbidity (NTU)", labelpad=10)
ax.set_ylabel("Reliability Score", labelpad=10)
ax.set_xlim(0, 500)
ax.set_ylim(0.20, 1.00)
ax.set_xticks([0, 100, 200, 300, 400, 500])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.35)

ax.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="gray",
    fontsize=11)

sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("figures/reliability_engine/reliability_ntu.png", dpi=600, bbox_inches="tight")
plt.close()

####################################################
# FIGURE 2: Reliability Degradation Under Flooded Conditions
####################################################

fig, ax = plt.subplots(figsize=(8.6, 5.4))

if "water_depth" in df.columns:
    ultra = df.groupby("water_depth")["R_ultrasonic"].mean().sort_index()
    radar = df.groupby("water_depth")["R_radar"].mean().sort_index()

    ultra = ultra.rolling(5, center=True, min_periods=1).mean().interpolate()
    radar = radar.rolling(3, center=True, min_periods=1).mean().interpolate()

    ax.plot(ultra.index, ultra.values, linewidth=3.2, color="#2e7d32", label="Ultrasonic")
    ax.plot(radar.index, radar.values, linewidth=3.2, color="#1565c0", label="Radar")

    # Optional smooth reference points
    if "R_ultrasonic" in df.columns:
        ultra_b = _binned_mean(df, "water_depth", "R_ultrasonic", bins=28)
        if not ultra_b.empty:
            ax.scatter(ultra_b["water_depth"], ultra_b["R_ultrasonic"], s=18, color="#2e7d32", alpha=0.55)

    if "R_radar" in df.columns:
        radar_b = _binned_mean(df, "water_depth", "R_radar", bins=28)
        if not radar_b.empty:
            ax.scatter(radar_b["water_depth"], radar_b["R_radar"], s=18, color="#1565c0", alpha=0.55)

ax.set_title("Reliability Degradation Under Flooded Conditions")
ax.set_xlabel("Flood Depth (cm)", labelpad=10)
ax.set_ylabel("Reliability Score", labelpad=10)
ax.set_xlim(0, 20)
ax.set_ylim(0.0, 1.05)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.legend(loc="upper right", frameon=True)

sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("figures/reliability_engine/reliability_depth.png", dpi=600, bbox_inches="tight")
plt.close()

####################################################
# FIGURE 3: Reliability Correlation Matrix
####################################################

fig, ax = plt.subplots(figsize=(10.2, 8.2))

heatmap_cols = [
    "R_lidar",
    "R_ultrasonic",
    "R_radar",
    "R_imu",
    "R_water",
    "water_depth",
    "ntu",
    "snr" ]

heatmap_cols = [c for c in heatmap_cols if c in df.columns]
corr = df[heatmap_cols].corr(numeric_only=True)

sns.heatmap(corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 10},
    linewidths=0.35,
    linecolor="#f0f0f0",
    square=True,
    cbar_kws={"label": "Pearson Correlation"},
    ax=ax)

ax.set_title("Reliability Correlation Matrix")
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("figures/reliability_engine/reliability_heatmap.png", dpi=600, bbox_inches="tight")
plt.close()

print()
print("Reliability Engine Visualizations Generated!")
print("Figures Created: 3")
print()
print("✓ reliability_ntu.png")
print("✓ reliability_depth.png")
print("✓ reliability_heatmap.png")
print()