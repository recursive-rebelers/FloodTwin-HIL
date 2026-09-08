import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fusion_engine.adaptive_fusion import AdaptiveFusionEngine
from fusion_engine.bayesian_fusion import BayesianFusionCore

RESULTS_DIR = BASE_DIR / "results" / "fusion_engine"
FIGURES_DIR = BASE_DIR / "figures" / "fusion_engine"
DATASET_PATH = RESULTS_DIR / "fusion_results.csv"
SUMMARY_PATH = RESULTS_DIR / "fusion_summary.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"fusion_results.csv not found at: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH).copy()

DEMO_MEASUREMENTS = {
    "z_lidar": 12.0,
    "z_ultra": 13.0,
    "z_radar": 11.0,
    "r_lidar": 0.30,
    "r_ultra": 0.45,
    "r_radar": 0.85,
}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#2f2f2f",
    "axes.linewidth": 1.1,
    "grid.color": "#d9d9d9",
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "font.size": 11,
    "ps.fonttype": 42,
})

def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(value) if np.isfinite(value) else float(default)

def density_to_mass(posterior_density: np.ndarray, delta_x: float) -> np.ndarray:
    posterior_density = np.asarray(posterior_density, dtype=float).ravel()
    posterior_density = np.nan_to_num(posterior_density, nan=0.0, posinf=0.0, neginf=0.0)
    mass = np.clip(posterior_density, 0.0, None) * float(delta_x)
    total = float(np.sum(mass))
    if not np.isfinite(total) or total <= 1e-12:
        return np.ones_like(mass, dtype=float) / max(len(mass), 1)
    return mass / total

def credible_interval_from_mass(
    depths: np.ndarray,
    posterior_mass: np.ndarray,
    mass_level: float = 0.95,
) -> Tuple[float, float]:
    depths = np.asarray(depths, dtype=float).ravel()
    posterior_mass = np.asarray(posterior_mass, dtype=float).ravel()

    if len(depths) != len(posterior_mass):
        raise ValueError("depths and posterior_mass must have the same length")

    cdf = np.cumsum(posterior_mass)
    cdf = np.clip(cdf, 0.0, 1.0)

    lower_q = (1.0 - mass_level) / 2.0
    upper_q = 1.0 - lower_q

    lower = float(np.interp(lower_q, cdf, depths))
    upper = float(np.interp(upper_q, cdf, depths))
    return lower, upper

def save_clean(fig: plt.Figure, filename_stem: str) -> None:
    fig.tight_layout()
    png_path = FIGURES_DIR / f"{filename_stem}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

def style_axes(ax: plt.Axes) -> None:
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

def load_summary_metrics() -> Dict[str, float]:
    if not SUMMARY_PATH.exists():
        return {}

    summary = pd.read_csv(SUMMARY_PATH)
    if summary.empty:
        return {}

    row = summary.iloc[0].to_dict()
    out: Dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
    return out

summary_metrics = load_summary_metrics()

# ---------------------------------------------------------------------
# Figure 1: Bayesian posterior distribution
# ---------------------------------------------------------------------
core = BayesianFusionCore(depth_min=0.0, depth_max=30.0, resolution=300)
fusion = AdaptiveFusionEngine(core)
# Example operating conditions used to visualize the posterior distribution
scene_complexity = 0.18
sensor_agreement = 0.91
measurement_spread = 0.55
sigmas = (
    safe_float(summary_metrics.get("fixed_sigma_lidar", 0.85)),
    safe_float(summary_metrics.get("fixed_sigma_ultrasonic", 0.75)),
    safe_float(summary_metrics.get("fixed_sigma_radar", 0.60)))

demo = fusion.fuse(
    z_lidar=DEMO_MEASUREMENTS["z_lidar"],
    z_ultra=DEMO_MEASUREMENTS["z_ultra"],
    z_radar=DEMO_MEASUREMENTS["z_radar"],
    r_lidar=DEMO_MEASUREMENTS["r_lidar"],
    r_ultra=DEMO_MEASUREMENTS["r_ultra"],
    r_radar=DEMO_MEASUREMENTS["r_radar"],
    sigmas=sigmas,
    scene_complexity=scene_complexity,
    sensor_agreement=sensor_agreement,
    measurement_spread=measurement_spread)

posterior_density = np.asarray(demo["posterior"], dtype=float)
posterior_mass = density_to_mass(posterior_density, core.delta_x)
ci_lower, ci_upper = credible_interval_from_mass(core.depths, posterior_mass, 0.95)

map_depth = safe_float(demo["depth_map"])
expected_depth = safe_float(demo["depth_mean"])
variance = safe_float(demo["variance"])
entropy = safe_float(demo["entropy"])
weights = demo["weights"]
adaptive_sigmas = demo["sigmas"]
diagnostics = demo.get("diagnostics", {})
posterior_peak = safe_float(diagnostics.get("posterior_peak", np.max(posterior_mass)))
confidence = safe_float(demo.get("confidence", diagnostics.get("confidence", float("nan"))))
effective_sensor_count = safe_float(diagnostics.get("effective_sensor_count", float("nan")))
dominance_ratio = safe_float(diagnostics.get("dominance_ratio", float("nan")))
weight_entropy = safe_float(diagnostics.get("weight_entropy", float("nan")))
ci_width = float(max(ci_upper - ci_lower, 0.0))
context = diagnostics.get("context_difficulty", np.nan)
gate = diagnostics.get("adaptation_gate", np.nan)

fig, ax = plt.subplots(figsize=(8.4, 5.2))

ax.plot(
    core.depths,
    posterior_density,
    color="#003399",
    linewidth=3.0,
    label="Posterior density")

ax.fill_between(core.depths, posterior_density, color="#003399", alpha=0.16)

ax.axvspan(ci_lower, ci_upper,
    color="#7b2cbf",
    alpha=0.09,
    label="95% credible interval")

ax.axvline(map_depth,
    color="#2e8b57",
    linestyle=":",
    linewidth=2.4,
    label=f"MAP = {map_depth:.2f} cm")

ax.axvline(expected_depth,
    color="#8b0000",
    linestyle="--",
    linewidth=2.2,
    label=f"Expected = {expected_depth:.2f} cm")

ax.set_title("Adaptive Reliability-Aware Bayesian Fusion")
ax.set_xlabel("Estimated Depth (cm)", labelpad=12)
ax.set_ylabel("Probability Density", labelpad=12)
ax.set_xlim(core.depth_min, core.depth_max)
ax.set_ylim(bottom=0)

posterior_text = (
    f"Confidence              {confidence:.3f}\n"
    f"Posterior Peak         {posterior_peak:.3f}\n"
    f"CI Width                   {ci_width:.3f}\n"
    f"Entropy                    {entropy:.3f}\n"
    f"Effective Sensors    {effective_sensor_count:.3f}\n"
    f"Context Difficulty     {context:.3f}")

ax.text(0.77, 0.72,
    posterior_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9.2,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#d0d0d0",
        alpha=0.96))

ax.legend(loc="upper right", frameon=True, framealpha=0.96)
style_axes(ax)
save_clean(fig, "fusion_posterior")

# ---------------------------------------------------------------------
# Figure 2: RMSE comparison across turbidity bins
# ---------------------------------------------------------------------
required_rmse_cols = {"ntu_observed", "fixed_abs_error", "adaptive_abs_error"}
missing_rmse_cols = required_rmse_cols - set(df.columns)
if missing_rmse_cols:
    raise ValueError(f"Missing required columns for Figure 2: {sorted(missing_rmse_cols)}")

df["ntu_bin"] = pd.cut(df["ntu_observed"], bins=8, include_lowest=True)
bin_intervals = df["ntu_bin"].cat.categories
bin_labels = []

for interval in bin_intervals:
    left = int(round(interval.left))
    right = int(round(interval.right))
    left = max(left, 0)
    bin_labels.append(f"{left}-{right}")

rmse_fixed = (
    df.groupby("ntu_bin", observed=True)["fixed_abs_error"]
    .apply(lambda x: float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=float))))))
    .astype(float))

rmse_adaptive = (
    df.groupby("ntu_bin", observed=True)["adaptive_abs_error"]
    .apply(lambda x: float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=float))))))
    .astype(float))

paired_mae_df = (df[["fixed_abs_error", "adaptive_abs_error"]]
    .apply(pd.to_numeric, errors="coerce").dropna())

overall_fixed = summary_metrics["rmse_fixed_fusion_map"]
overall_adaptive = summary_metrics["rmse_adaptive_fusion_map"]
gain_percent = ((overall_fixed - overall_adaptive) / max(overall_fixed, 1e-12)) * 100.0

mae_fixed = float(paired_mae_df["fixed_abs_error"].mean())
mae_adaptive = float(paired_mae_df["adaptive_abs_error"].mean())
mae_gain_percent = ((mae_fixed - mae_adaptive) / max(mae_fixed, 1e-12)) * 100.0
adaptive_win_rate = summary_metrics["adaptive_win_rate_map"]

x = np.arange(len(rmse_fixed))
fig, ax = plt.subplots(figsize=(8.4, 5.2))

ax.plot(x, rmse_fixed.values,
    color="#cc0000",
    linewidth=2.8,
    marker="o",
    markersize=6,
    label="Fixed fusion")

ax.plot(x, rmse_adaptive.values,
    color="#003399",
    linewidth=3.2,
    marker="o",
    markersize=6,
    label="Adaptive fusion")

ax.fill_between(x, rmse_fixed.values,
    rmse_adaptive.values,
    where=(rmse_fixed.values >= rmse_adaptive.values),
    interpolate=True,
    color="#1b9e77",
    alpha=0.12)

ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=0, ha="right", fontsize=15)
ax.set_title("Fixed vs Adaptive Bayesian Fusion Across Turbidity Levels")
ax.set_xlabel("Water Turbidity (NTU) Bins", labelpad=12)
ax.set_ylabel("RMSE (cm)", labelpad=12)

rmse_text = (
    f"Fixed RMSE            {overall_fixed:.3f} cm\n"
    f"Adaptive RMSE       {overall_adaptive:.3f} cm\n"
    f"RMSE Gain             {gain_percent:.2f}%\n\n"
    f"Fixed MAE              {mae_fixed:.3f} cm\n"
    f"Adaptive MAE         {mae_adaptive:.3f} cm\n"
    f"MAE Gain               {mae_gain_percent:.2f}%\n\n"
    f"Adaptive Win Rate  {adaptive_win_rate:.3f}\n"
    f"Mean Win Rate       {summary_metrics.get('adaptive_win_rate_mean', float('nan')):.3f}")

ax.text(0.025, 0.965,
    rmse_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9.7,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#d0d0d0",
        alpha=0.96))

ax.legend(loc="lower right", frameon=True, framealpha=0.96)
style_axes(ax)
save_clean(fig, "rmse_comparison")

# ---------------------------------------------------------------------
# Figure 3: Hazard probability assessment
# ---------------------------------------------------------------------
hazard_column = "hazard_probability" if "hazard_probability" in df.columns else "hazard_prob"
if hazard_column not in df.columns:
    raise ValueError("Missing hazard probability column for Figure 3")

corr_hazard_depth = float(
    df[["true_depth", hazard_column]].corr().iloc[0, 1]) if "true_depth" in df.columns else float("nan")
hazard_rate = float(
    (df[hazard_column] >= 0.80).mean() * 100.0)
safe_rate = float(
    (df["hazard_status"] == "SAFE").mean() * 100.0) if "hazard_status" in df.columns else float("nan")
caution_rate = float(
    (df["hazard_status"] == "CAUTION").mean() * 100.0) if "hazard_status" in df.columns else float("nan")
mean_risk = float(
    df["risk_score"].mean()) if "risk_score" in df.columns else float("nan")

fig, ax = plt.subplots(figsize=(8.4, 5.2))

scatter = ax.scatter(
    df["true_depth"],
    df[hazard_column],
    c=df[hazard_column],
    cmap="viridis",
    alpha=0.72,
    s=26,
    edgecolor="none",
    vmin=0.0,
    vmax=1.0)

x = df["true_depth"].to_numpy(dtype=float)
y = df[hazard_column].to_numpy(dtype=float)

params, _ = curve_fit(sigmoid, x, y,
    p0=[1.0, 10.0, 1.0],
    bounds=([0.9, 5.0, 0.1], [1.1, 20.0, 10.0]),
    maxfev=10000)
x_fit = np.linspace(x.min(), x.max(), 300)
y_fit = sigmoid(x_fit, *params)

ax.axvline(
    x=10.0,
    linestyle="--",
    linewidth=2.0,
    color="#8b0000",
    label="Hazard Threshold")

ax.axhline(
    y=0.80,
    linestyle=":",
    linewidth=2.0,
    color="#303030",
    label="Decision Boundary")

ax.plot(x_fit, y_fit,
    color="navy",
    linewidth=3.2,
    label="Logistic Fit")

cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Hazard probability", rotation=270, labelpad=16)

ax.set_title("Posterior Hazard Probability Assessment")
ax.set_xlabel("True Pothole Depth (cm)", labelpad=12)
ax.set_ylabel("Hazard Probability", labelpad=12)
ax.set_ylim(0.0, 1.02)

hazard_text = (
    f"Safe Rate                  {safe_rate:.1f}%\n"
    f"Caution Rate               {caution_rate:.1f}%\n"
    f"Hazard Rate              {hazard_rate:.1f}%\n"
    f"Corr(hazard, depth)    {corr_hazard_depth:.3f}\n"
    f"Mean Risk Score        {mean_risk:.2f}")

ax.text(0.98, 0.06,
    hazard_text,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9.5,
    bbox=dict(
        boxstyle="round, pad=0.4",
        facecolor="white",
        edgecolor="#d0d0d0",
        alpha=0.96))

ax.legend(loc="upper left", frameon=True, framealpha=0.96)
style_axes(ax)
save_clean(fig, "hazard_probability")

print("=" * 60)
print("Fusion Engine Evaluation Figures Generated:")
print()
print("✓ fusion_posterior.png")
print("✓ rmse_comparison.png")
print("✓ hazard_probability.png")
print()
print("=" * 60)