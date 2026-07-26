import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "results" / "fault_injection" / "robustness_results.csv"
FIGURES_DIR = BASE_DIR / "figures" / "fault_injection"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ORDER = [
    "Baseline (No Faults)",
    "LiDAR Low Dropout (10%)",
    "LiDAR High Dropout (50%)",
    "Ultrasonic High Dropout (50%)",
    "Radar High Dropout (50%)",
    "Radar Severe Noise",
    "Ultrasonic High Delay",
    "Radar High Sync Error",
    "Compound Scenario A",
    "Compound Scenario C" ]

LABEL_MAP = {
    "Baseline (No Faults)": "Baseline",
    "LiDAR Low Dropout (10%)": "LiDAR\nDrop 10%",
    "LiDAR High Dropout (50%)": "LiDAR\nDrop 50%",
    "Ultrasonic High Dropout (50%)": "Ultra\nDrop 50%",
    "Radar High Dropout (50%)": "Radar\nDrop 50%",
    "Radar Severe Noise": "Radar\nNoise",
    "Ultrasonic High Delay": "Ultra\nDelay",
    "Radar High Sync Error": "Radar\nSync",
    "Compound Scenario A": "Comp A",
    "Compound Scenario C": "Comp C" }

REQUIRED_COLUMNS = [
    "Experiment",
    "Fixed_RMSE_cm",
    "Adaptive_RMSE_cm",
    "Fixed_Degradation_%",
    "Adaptive_Degradation_%",
    "Fixed_Failure_Rate_%",
    "Adaptive_Failure_Rate_%",
    "Mean_Fixed_Confidence",
    "Mean_Adaptive_Confidence" ]

def _validate_columns(df, required_columns):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError("robustness_results.csv is missing required columns: " + ", ".join(missing))

def _apply_publication_style():
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

def _prepare_ordered_frame(df, order):
    ordered = df.copy()
    order_map = {name: idx for idx, name in enumerate(order)}
    ordered["__order__"] = ordered["Experiment"].map(order_map).fillna(len(order))
    ordered["__label__"] = ordered["Experiment"].map(LABEL_MAP).fillna(ordered["Experiment"])
    ordered = ordered.sort_values("__order__").drop(columns="__order__")
    return ordered

def _save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=600, bbox_inches="tight")
    plt.close(fig)

def generate_robustness_figures(results_path=RESULTS_FILE):
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found. Run the fault robustness pipeline first")

    _apply_publication_style()
    df = pd.read_csv(results_path)
    _validate_columns(df, REQUIRED_COLUMNS)

    baseline_name = "Baseline (No Faults)"
    ordered_df = _prepare_ordered_frame(df, EXPERIMENT_ORDER)
    dropout_df = ordered_df[ordered_df["Experiment"].str.contains("Dropout", na=False)].copy()
    compound_df = ordered_df[ordered_df["Experiment"].isin([baseline_name,
        "Compound Scenario A", "Compound Scenario C"])].copy()

    # ---------------------------------------------------------
    # Figure 1: RMSE Under Fault Severity
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    x_vals = np.arange(len(ordered_df))
    width = 0.36

    ax.bar(
        x_vals - width / 2,
        ordered_df["Fixed_RMSE_cm"],
        width=width,
        label="Fixed Fusion",
        color="#cc4c4c",
        edgecolor="#8b0000")

    ax.bar(
        x_vals + width / 2,
        ordered_df["Adaptive_RMSE_cm"],
        width=width,
        label="Adaptive Fusion",
        color="#4d94ff",
        edgecolor="#003399")

    ax.set_title("Estimation Error Under Fault Severity")
    ax.set_ylabel("RMSE (cm)", labelpad=12)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(ordered_df["__label__"], rotation=0, ha="right", fontsize=15)
    ax.set_ylim(0, max(
        ordered_df["Fixed_RMSE_cm"].max(),
        ordered_df["Adaptive_RMSE_cm"].max()) * 1.18)
    ax.legend(fontsize=12, frameon=True, ncol=2)

    sns.despine()
    _save_figure(fig, "fault_rmse.png")

    # ---------------------------------------------------------
    # Figure 2: Performance Degradation Relative to Baseline
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        ordered_df["__label__"],
        ordered_df["Fixed_Degradation_%"],
        marker="o",
        color="#8b0000",
        linewidth=3,
        markersize=8,
        label="Fixed Fusion")

    ax.plot(
        ordered_df["__label__"],
        ordered_df["Adaptive_Degradation_%"],
        marker="s",
        color="#003399",
        linewidth=3,
        markersize=8,
        label="Adaptive Fusion")

    ax.axhline(0, color="#303030", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_title("Performance Degradation Relative to Fault-Free Baseline")
    ax.set_ylabel("Degradation (%)", labelpad=12)
    ax.set_xlabel("Fault Condition", labelpad=12)
    ax.tick_params(axis="x", rotation=0, labelsize=14)
    ax.legend(fontsize=12, frameon=True, ncol=2)

    sns.despine()
    _save_figure(fig, "performance_degradation.png")

    # ---------------------------------------------------------
    # Figure 3: Adaptive vs Fixed Under Dropout
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    x_vals = np.arange(len(dropout_df))
    width = 0.36

    ax.bar(
        x_vals - width / 2,
        dropout_df["Fixed_RMSE_cm"],
        width=width,
        label="Fixed Fusion",
        color="#ff7b7b",
        edgecolor="#8b0000")

    ax.bar(
        x_vals + width / 2,
        dropout_df["Adaptive_RMSE_cm"],
        width=width,
        label="Adaptive Fusion",
        color="#6ca8ff",
        edgecolor="#003399")

    ax.set_title("Adaptive vs Fixed Fusion Under Dropout Faults")
    ax.set_ylabel("RMSE (cm)", labelpad=12)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(dropout_df["__label__"], rotation=0, fontsize=15)
    ax.set_ylim(0, max(
        dropout_df["Fixed_RMSE_cm"].max(),
        dropout_df["Adaptive_RMSE_cm"].max()) * 1.20)
    ax.legend(fontsize=12, frameon=True, ncol=2)

    sns.despine()
    _save_figure(fig, "dropout_comparison.png")

    # ---------------------------------------------------------
    # Figure 4: Compound Fault Analysis
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_vals = np.arange(len(compound_df))
    bar_h = 0.38

    ax.barh(
        y_vals - bar_h / 2,
        compound_df["Fixed_RMSE_cm"],
        height=bar_h,
        label="Fixed Fusion",
        color="#ff6666",
        edgecolor="#8b0000")

    ax.barh(
        y_vals + bar_h / 2,
        compound_df["Adaptive_RMSE_cm"],
        height=bar_h,
        label="Adaptive Fusion",
        color="#4d94ff",
        edgecolor="#003399")

    ax.set_title("Worst-Case Compound Fault Stress Test")
    ax.set_xlabel("RMSE (cm)", labelpad=12)
    ax.set_yticks(y_vals)
    ax.set_yticklabels(compound_df["__label__"])
    ax.set_xlim(0, max(
            compound_df["Fixed_RMSE_cm"].max(),
            compound_df["Adaptive_RMSE_cm"].max()) * 1.20)
    ax.legend(fontsize=12, frameon=True, loc="lower right")

    sns.despine()
    _save_figure(fig, "compound_faults.png")

print("\n✓ Fault Injection Evaluation Figures Generated:")
print()
print(" figures/fault_injection/fault_rmse.png")
print(" figures/fault_injection/performance_degradation.png")
print(" figures/fault_injection/dropout_comparison.png")
print(" figures/fault_injection/compound_faults.png\n")

if __name__ == "__main__":
    generate_robustness_figures()