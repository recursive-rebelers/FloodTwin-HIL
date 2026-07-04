import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fusion_engine.adaptive_fusion import AdaptiveFusionEngine
from fusion_engine.bayesian_fusion import BayesianFusionCore

FIGURES_DIR = BASE_DIR / "figures" / "fusion_engine"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = BASE_DIR / "results" / "fusion_engine" / "fusion_results.csv"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"fusion_results.csv not found at: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH).copy()

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#2f2f2f",
        "axes.linewidth": 1.1,
        "grid.color": "#d9d9d9",
        "grid.alpha": 0.6,
        "grid.linestyle": "--",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 17,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "font.size": 11,
    }
)

def density_to_mass(posterior_density: np.ndarray, delta_x: float) -> np.ndarray:
    posterior_density = np.asarray(posterior_density, dtype=float).ravel()
    mass = np.clip(posterior_density, 0.0, None) * float(delta_x)
    total = float(np.sum(mass))

    if not np.isfinite(total) or total <= 1e-12:
        return np.ones_like(mass, dtype=float) / len(mass)
    return mass / total

def credible_interval_from_mass(
    depths: np.ndarray, posterior_mass: np.ndarray, mass_level: float = 0.95
) -> tuple[float, float]:
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

def save_clean(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=600, bbox_inches="tight")
    plt.close(fig)

def style_axes(ax: plt.Axes) -> None:
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

# ----------------------------------------------------
# FIGURE 1: Bayesian posterior distribution
# ----------------------------------------------------
core = BayesianFusionCore(depth_min=0.0, depth_max=30.0, resolution=300)
fusion = AdaptiveFusionEngine(
    core,
    gamma=1.20,
    weight_blend=0.60,
    sigma_scale=1.0,
    sigma_offset=0.08,
    sigma_min=0.45,
    sigma_max=2.85,
)

demo = fusion.fuse(
    z_lidar=12.0,
    z_ultra=13.0,
    z_radar=11.0,
    r_lidar=0.30,
    r_ultra=0.45,
    r_radar=0.85,
)

posterior_density = np.asarray(demo["posterior"], dtype=float)
posterior_mass = density_to_mass(posterior_density, core.delta_x)
ci_lower, ci_upper = credible_interval_from_mass(core.depths, posterior_mass, 0.95)

map_depth = float(demo["depth_map"])
expected_depth = float(demo["depth_mean"])
variance = float(demo["variance"])
entropy = float(demo["entropy"])
weights = demo["weights"]
posterior_peak = float(np.max(posterior_mass))

fig, ax = plt.subplots(figsize=(8.4, 5.2))

ax.plot(
    core.depths,
    posterior_density,
    color="#003399",
    linewidth=3.0,
    label="Posterior density",
)

ax.fill_between(
    core.depths,
    posterior_density,
    color="#003399",
    alpha=0.16,
)

ax.axvspan(
    ci_lower,
    ci_upper,
    color="#7b2cbf",
    alpha=0.09,
    label="95% credible interval",
)

ax.axvline(
    map_depth,
    color="#2e8b57",
    linestyle=":",
    linewidth=2.4,
    label=f"MAP = {map_depth:.2f} cm",
)

ax.axvline(
    expected_depth,
    color="#8b0000",
    linestyle="--",
    linewidth=2.2,
    label=f"Expected = {expected_depth:.2f} cm",
)

ax.set_title("Bayesian Posterior Distribution")
ax.set_xlabel("Estimated depth (cm)", labelpad=12)
ax.set_ylabel("Probability density", labelpad=12)
ax.set_xlim(core.depth_min, core.depth_max)
ax.set_ylim(bottom=0)

posterior_text = (
    f"Weights  L:{weights['lidar']:.2f}  "
    f"U:{weights['ultrasonic']:.2f}  "
    f"R:{weights['radar']:.2f}\n"
    f"Peak     {posterior_peak:.3f}\n"
    f"Variance {variance:.3f}\n"
    f"Entropy  {entropy:.3f}"
)
ax.text(
    0.02,
    0.98,
    posterior_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#d0d0d0",
        alpha=0.96,
    ),
)

ax.legend(loc="upper right", frameon=True, framealpha=0.96)
style_axes(ax)
save_clean(fig, "fusion_posterior.png")

# ----------------------------------------------------
# FIGURE 2: RMSE comparison across turbidity bins
# ----------------------------------------------------
df["ntu_bin"] = pd.cut(df["ntu"], bins=8, include_lowest=True)

rmse_fixed = (
    df.groupby("ntu_bin", observed=True)["fixed_mae"]
    .apply(lambda x: np.sqrt(np.mean(np.square(x))))
    .astype(float)
)
rmse_adaptive = (
    df.groupby("ntu_bin", observed=True)["adaptive_mae"]
    .apply(lambda x: np.sqrt(np.mean(np.square(x))))
    .astype(float)
)

overall_fixed = float(np.sqrt(np.mean(np.square(df["fixed_mae"]))))
overall_adaptive = float(np.sqrt(np.mean(np.square(df["adaptive_mae"]))))
gain_percent = ((overall_fixed - overall_adaptive) / overall_fixed) * 100.0

x = np.arange(len(rmse_fixed))
bin_labels = [f"Bin {i + 1}" for i in x]

fig, ax = plt.subplots(figsize=(8.4, 5.2))

ax.plot(
    x,
    rmse_fixed.values,
    color="#cc0000",
    linewidth=2.8,
    marker="o",
    markersize=6,
    label="Fixed fusion",
)

ax.plot(
    x,
    rmse_adaptive.values,
    color="#003399",
    linewidth=2.8,
    marker="o",
    markersize=6,
    label="Adaptive fusion",
)

ax.fill_between(
    x,
    rmse_fixed.values,
    rmse_adaptive.values,
    where=(rmse_fixed.values >= rmse_adaptive.values),
    interpolate=True,
    color="#1b9e77",
    alpha=0.10,
)

ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.set_title("Fixed vs Adaptive Fusion RMSE by NTU Bin")
ax.set_xlabel("Water turbidity (NTU) bins", labelpad=12)
ax.set_ylabel("RMSE (cm)", labelpad=12)

rmse_text = (
    f"Overall fixed RMSE     {overall_fixed:.3f} cm\n"
    f"Overall adaptive RMSE   {overall_adaptive:.3f} cm\n"
    f"Relative gain           {gain_percent:.1f}%"
)
ax.text(
    0.98,
    0.06,
    rmse_text,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#d0d0d0",
        alpha=0.96,
    ),
)

ax.legend(loc="upper right", frameon=True, framealpha=0.96)
style_axes(ax)
save_clean(fig, "rmse_comparison.png")

# ----------------------------------------------------
# FIGURE 3: Confidence under flood severity
# ----------------------------------------------------
corr_conf_mae = float(df[["adaptive_confidence", "adaptive_mae"]].corr().iloc[0, 1])

fig, ax = plt.subplots(figsize=(8.4, 5.2))

sns.scatterplot(
    data=df,
    x="water_depth",
    y="adaptive_confidence",
    s=18,
    alpha=0.28,
    color="#6a1b9a",
    edgecolor=None,
    ax=ax,
)

sns.regplot(
    data=df,
    x="water_depth",
    y="adaptive_confidence",
    scatter=False,
    lowess=True,
    ci=95,
    color="#2f0f59",
    line_kws={"linewidth": 3},
    ax=ax,
)

mean_conf = float(df["adaptive_confidence"].mean())
std_conf = float(df["adaptive_confidence"].std())

ax.axhline(
    mean_conf,
    color="#666666",
    linestyle="--",
    linewidth=1.6,
    alpha=0.8,
    label=f"Mean confidence = {mean_conf:.3f}",
)

ax.set_title("Bayesian Confidence Under Increasing Flood Severity")
ax.set_xlabel("Flood depth (cm)", labelpad=12)
ax.set_ylabel("Confidence score", labelpad=12)
ax.set_ylim(0, 1.05)

confidence_text = (
    f"Mean confidence  {mean_conf:.3f}\n"
    f"Std. confidence   {std_conf:.3f}\n"
    f"Corr(conf, MAE)   {corr_conf_mae:.3f}"
)
ax.text(
    0.02,
    0.98,
    confidence_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#d0d0d0",
        alpha=0.96,
    ),
)

ax.legend(loc="lower right", frameon=True, framealpha=0.96)
style_axes(ax)
save_clean(fig, "confidence_analysis.png")

# ----------------------------------------------------
# FIGURE 4: Hazard probability assessment
# ----------------------------------------------------
hazard_column = "hazard_probability" if "hazard_probability" in df.columns else "hazard_prob"
corr_hazard_depth = float(df[["true_depth", hazard_column]].corr().iloc[0, 1])
hazard_rate = float((df[hazard_column] >= 0.80).mean() * 100.0)

fig, ax = plt.subplots(figsize=(8.4, 5.2))

scatter = ax.scatter(
    df["true_depth"],
    df[hazard_column],
    c=df[hazard_column],
    cmap="viridis",
    alpha=0.72,
    s=22,
    edgecolor="none",
    vmin=0.0,
    vmax=1.0,
)

sns.regplot(
    data=df,
    x="true_depth",
    y=hazard_column,
    scatter=False,
    lowess=True,
    color="#111111",
    line_kws={"linewidth": 2.4, "alpha": 0.90},
    ax=ax,
)

ax.axvline(
    x=10.0,
    linestyle="--",
    linewidth=2.0,
    color="#8b0000",
    label="Hazard threshold",
)

ax.axhline(
    y=0.80,
    linestyle=":",
    linewidth=2.0,
    color="#303030",
    label="Decision boundary",
)

cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Hazard probability", rotation=270, labelpad=16)

ax.set_title("Posterior Hazard Probability Assessment")
ax.set_xlabel("True pothole depth (cm)", labelpad=12)
ax.set_ylabel("Hazard probability", labelpad=12)
ax.set_ylim(0.0, 1.02)

hazard_text = (
    f"Detected hazards  {hazard_rate:.1f}%\n"
    f"Corr(hazard, depth) {corr_hazard_depth:.3f}"
)
ax.text(
    0.98,
    0.06,
    hazard_text,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#d0d0d0",
        alpha=0.96,
    ),
)

ax.legend(loc="upper left", frameon=True, framealpha=0.96)
style_axes(ax)
save_clean(fig, "hazard_probability.png")

print()
print("=" * 60)
print("Fusion Engine Evaluation Figures Generated")
print()
print("✓ fusion_posterior.png")
print("✓ rmse_comparison.png")
print("✓ confidence_analysis.png")
print("✓ hazard_probability.png")
print()
print("=" * 60)