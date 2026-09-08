import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = BASE_DIR / "datasets" / "dataset_v2.csv"
PEARSON_FILE = BASE_DIR / "results" / "reliability_engine" / "reliability_correlations_pearson.csv"
SPEARMAN_FILE = BASE_DIR / "results" / "reliability_engine" / "reliability_correlations_spearman.csv"
FIGURES_DIR = BASE_DIR / "figures" / "reliability_engine"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------
if not DATASET_FILE.exists():
    raise FileNotFoundError(f"Dataset V2 not found: {DATASET_FILE}")
df = pd.read_csv(DATASET_FILE)

# Numeric coercion for required observables/diagnostics
numeric_cols = [
    "R_lidar", "R_ultrasonic", "R_radar", "R_imu", "R_water",
    "water_depth", "water_context_depth_proxy", "ntu", "snr",
    "clutter_probability", "fusion_confidence", "scene_complexity",
    "measurement_spread", "measurement_consistency", "fusion_error_abs",
    "equal_fusion_error_abs", "lidar_error_abs", "ultrasonic_error_abs", "radar_error_abs"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------------
# Plot Configuration
# ------------------------------------------------------------------
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
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

SENSOR_COLORS = {
    "LiDAR": "#b71c1c",
    "Ultrasonic": "#2e7d32",
    "Radar": "#1565c0",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _lowess_curve(x, y, frac=0.12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]
    if len(x) < 10: return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Collapse repeated x-values for a stable LOWESS representation
    xy = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean()
    if len(xy) < 10: return None
    return lowess(xy["y"].to_numpy(), xy["x"].to_numpy(), frac=frac, return_sorted=True)

def _binned_mean(df_in, xcol, ycol, bins=24):
    temp = df_in[[xcol, ycol]].dropna().copy()
    if temp.empty: return temp
    try:
        intervals = pd.cut(temp[xcol].astype(float),
        bins=bins, duplicates="drop", include_lowest=True)
        out = (temp.groupby(intervals, observed=True).agg({xcol: "mean", ycol: "mean"})
            .dropna().reset_index(drop=True).sort_values(xcol))            
        return out
    except Exception: return temp.sort_values(xcol)

def _load_corr_matrix(path):
    if not path.exists(): raise FileNotFoundError(f"Correlation table not found: {path}")
    corr = pd.read_csv(path, index_col=0)
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)
    return corr.apply(pd.to_numeric, errors="coerce")

# ------------------------------------------------------------------
# FIGURE 1: LiDAR Reliability Under Increasing Turbidity
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 5.4))
plot_df = df[["ntu", "R_lidar"]].dropna()

sns.scatterplot(
    data=plot_df, x="ntu", y="R_lidar",
    s=10, alpha=0.08, color=SENSOR_COLORS["LiDAR"], edgecolor=None, ax=ax)

smooth = _lowess_curve(plot_df["ntu"], plot_df["R_lidar"], frac=0.12)
if smooth is not None:
    ax.plot(smooth[:, 0], smooth[:, 1], color="#7f0000", linewidth=3.5, label="LOWESS Trend")

binned = _binned_mean(plot_df, "ntu", "R_lidar", bins=28)
if not binned.empty:
    ax.plot(binned["ntu"], binned["R_lidar"],
        color="#ff7043", linewidth=2.0, alpha=0.95, label="Binned Mean")

ax.set_title("LiDAR Reliability Under Increasing Turbidity")
ax.set_xlabel("Water Turbidity (NTU)", labelpad=12)
ax.set_ylabel("Reliability Score", labelpad=12)
ax.set_xlim(0, 500)
ax.set_ylim(0.20, 1.00)
ax.set_xticks([0, 100, 200, 300, 400, 500])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.35)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=11)

sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "reliability_ntu.png", dpi=600, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# FIGURE 2: Reliability Response to Observable Water-Context Severity
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 5.6))
xcol = "water_context_depth_proxy"
required = [xcol, "R_lidar", "R_ultrasonic", "R_radar"]
missing = [c for c in required if c not in df.columns]
if missing: raise ValueError(f"Missing columns required for depth-context figure: {missing}")

for sensor, reliability_col in [
    ("LiDAR", "R_lidar"), ("Ultrasonic", "R_ultrasonic"), ("Radar", "R_radar")]:
    plot_df = df[[xcol, reliability_col]].dropna()
    binned = _binned_mean(plot_df, xcol, reliability_col, bins=24)
    if not binned.empty:
        ax.plot(binned[xcol], binned[reliability_col],
            linewidth=3.0, color=SENSOR_COLORS[sensor], label=sensor)

ax.set_title("Reliability Response to Observable Water-Context Severity")
ax.set_xlabel("Water-Context Depth Proxy (cm-equivalent)", labelpad=12)
ax.set_ylabel("Reliability Score", labelpad=12)
ax.set_xlim(0, 14)
ax.set_ylim(0.20, 1.02)
ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.35)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")

sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "reliability_depth.png", dpi=600, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# FIGURE 3: Reliability Correlation Matrix
# ------------------------------------------------------------------
pearson = _load_corr_matrix(PEARSON_FILE)
heatmap_cols = [
    "R_lidar", "R_ultrasonic", "R_radar", "R_imu", "R_water", "water_depth",
    "water_context_depth_proxy", "ntu", "snr", "scene_complexity", "fusion_confidence"]

heatmap_cols = [c for c in heatmap_cols if c in pearson.index and c in pearson.columns]
if len(heatmap_cols) < 2:
    raise ValueError("Insufficient columns available in Pearson correlation matrix")

alias = {"water_context_depth_proxy": "water_context"}
corr = pearson.loc[heatmap_cols, heatmap_cols]
corr = corr.rename(index=alias, columns=alias)
fig, ax = plt.subplots(figsize=(10.6, 9.0))

sns.heatmap(corr,
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 9},
    linewidths=0.35, linecolor="#f0f0f0", square=True,
    cbar_kws={"label": "Pearson Correlation"}, ax=ax)

ax.set_title("Reliability and Observable-Context Correlation Matrix")
ax.tick_params(axis="x", rotation=35)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "reliability_heatmap.png", dpi=600, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# FIGURE 4: Reliability–Error Association (Pearson vs Spearman)
# ------------------------------------------------------------------
pearson = _load_corr_matrix(PEARSON_FILE)
spearman = _load_corr_matrix(SPEARMAN_FILE)

pairs = [
    ("LiDAR", "R_lidar", "lidar_error_abs"),
    ("Ultrasonic", "R_ultrasonic", "ultrasonic_error_abs"),
    ("Radar", "R_radar", "radar_error_abs")]

rows = []
for sensor, reliability_col, error_col in pairs:
    if reliability_col not in pearson.index or error_col not in pearson.columns:
        raise ValueError(f"Missing Pearson pair: {reliability_col} vs {error_col}")
    if reliability_col not in spearman.index or error_col not in spearman.columns:
        raise ValueError(f"Missing Spearman pair: {reliability_col} vs {error_col}")

    rows.append({
        "Sensor": sensor,
        "Pearson": float(pearson.loc[reliability_col, error_col]),
        "Spearman": float(spearman.loc[reliability_col, error_col]),
    })

association = pd.DataFrame(rows).set_index("Sensor")
fig, ax = plt.subplots(figsize=(9.0, 5.8))
x = np.arange(len(association))
width = 0.34

ax.bar(x - width / 2, association["Pearson"].to_numpy(),
    width, label="Pearson", color="#8b0000")

ax.bar(x + width / 2, association["Spearman"].to_numpy(),
    width, label="Spearman", color="#003399")

ax.axhline(0.0, linewidth=1.0, color="#333333")

ax.set_title("Reliability–Absolute Error Association")
ax.set_xlabel("Sensor", labelpad=12)
ax.set_ylabel("Correlation Coefficient", labelpad=12)
ax.set_xticks(x)
ax.set_xticklabels(association.index)
ax.set_ylim(-1.0, 0.15)
ax.set_yticks([-1.0, -0.75, -0.50, -0.25, 0.0])
ax.grid(True, axis="y", alpha=0.35)
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="gray")

for i, sensor in enumerate(association.index):
    p = association.loc[sensor, "Pearson"]
    s = association.loc[sensor, "Spearman"]
    ax.text(i - width / 2, p - 0.05, f"{p:.2f}", ha="center", va="top", fontsize=10)
    ax.text(i + width / 2, s - 0.05, f"{s:.2f}", ha="center", va="top", fontsize=10)

sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "reliability_error_association.png", dpi=600, bbox_inches="tight")
plt.close()

print()
print("Reliability Engine Visualizations Generated!")
print("Figures Created: 4")
print()
print("✓ reliability_ntu.png")
print("✓ reliability_depth.png")
print("✓ reliability_heatmap.png")
print("✓ reliability_error_association.png")
print()