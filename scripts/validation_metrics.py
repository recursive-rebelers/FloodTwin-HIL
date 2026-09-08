import ast
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuration & Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RES_DIR = BASE_DIR / "results" / "benchmarking"
FIG_DIR = BASE_DIR / "figures" / "benchmarking"
TABLE_DIR = BASE_DIR / "paper" / "tables" / "benchmarking"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Publication Figure Style
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk", rc={
    "font.size": 10,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": "#d8d8d8",
    "grid.linewidth": 0.65,
    "grid.alpha": 0.7,
    "grid.linestyle": "-",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

COLORS = {
    "adaptive": "#075985",
    "fixed": "#9f1239",
    "single": "#166534",
    "neutral": "#374151",
    "light": "#e5e7eb",
    "medium": "#9ca3af",
    "accent": "#b45309",
}

DEPTH_RANGE_CM: Tuple[float, float] = (0.0, 30.0)
NTU_RANGE: Tuple[float, float] = (0.0, 500.0)
WATER_PROXY_RANGE_CM: Tuple[float, float] = (0.0, 20.0)
NTU_BIN_EDGES = np.arange(0.0, 500.0 + 50.0, 50.0)
DEPTH_BIN_EDGES = np.arange(0.0, 30.0 + 5.0, 5.0)

VALIDATION_COMPONENTS = (
    "dataset_coverage",
    "environment_coverage",
    "fault_coverage",
    "benchmark_breadth",
    "predictive_accuracy",
    "uncertainty_calibration",
    "statistical_evidence",
    "hazard_validation",
)

VALIDATION_LABELS = {
    "dataset_coverage": "Dataset Coverage",
    "environment_coverage": "Environment Coverage",
    "fault_coverage": "Fault Coverage",
    "benchmark_breadth": "Benchmark Breadth",
    "predictive_accuracy": "Predictive Accuracy",
    "uncertainty_calibration": "Uncertainty Calibration",
    "statistical_evidence": "Statistical Evidence",
    "hazard_validation": "Hazard Validation",
}

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------
def _load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "benchmark": RES_DIR / "benchmark_results.csv",
        "ablation": RES_DIR / "ablation_results.csv",
        "statistics": RES_DIR / "statistical_summary.csv",
        "validation": RES_DIR / "validation_scores.csv",
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing: raise FileNotFoundError(
        "Required benchmark artifacts are missing:\n" + "\n".join(missing))

    return (
        pd.read_csv(required["benchmark"]),
        pd.read_csv(required["ablation"]),
        pd.read_csv(required["statistics"]),
        pd.read_csv(required["validation"]))

# ------------------------------------------------------------------
# General Helpers
# ------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing: raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")

def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _clean_numeric(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]

def _ci95_mean(values: pd.Series) -> Tuple[float, float]:
    arr = _clean_numeric(values)
    if arr.size < 2: return np.nan, np.nan
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    margin = 1.96 * sem
    return mean - margin, mean + margin

def _safe_mean(values: Any) -> float:
    arr = _clean_numeric(values)
    return float(arr.mean()) if arr.size else np.nan

def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if value is None: return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t", "available", "active"}: return True
    if text in {"false", "0", "no", "n", "f", "unavailable", "inactive", "nan", "none", ""}:
        return False
    try: return bool(float(value))
    except (TypeError, ValueError): return False

def _parse_bool_series(series: pd.Series) -> pd.Series:
    return series.map(_safe_bool).astype(bool)

def _pvalue_text(value: object) -> str:
    try: p = float(value)
    except (TypeError, ValueError): return "NA"
    if not np.isfinite(p): return "NA"
    if p <= 0.0: return "<1e-300"
    if p < 1e-4: return f"{p:.2e}"
    return f"{p:.4f}"

def _comparison_label(value: object) -> str:
    mapping = {
        "adaptive_vs_fixed": "Adaptive vs Fixed", "adaptive_vs_single": "Adaptive vs LiDAR-only"}
    text = str(value).strip().lower()
    return mapping.get(text, str(value))

def _validation_band(score: float) -> str:
    if not np.isfinite(score): return "Unavailable"
    if score >= 0.85: return "Excellent"
    if score >= 0.70: return "Strong"
    if score >= 0.50: return "Moderate"
    return "Weak"

def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns: return col
    return None

def _ensure_observable_ntu(bm_df: pd.DataFrame) -> pd.Series:
    col = _first_existing_column(bm_df, (
        "ntu_observed", "observed_ntu", "ntu_sensor", "ntu_proxy", "ntu", "ntu_measurement"))
    if col is None: raise ValueError("Benchmark results do not contain an observable NTU field")
    return _numeric(bm_df[col])

def _ensure_water_proxy(bm_df: pd.DataFrame) -> pd.Series:
    col = _first_existing_column(bm_df, (
        "water_context_depth_proxy", "observable_water_depth", "water_depth_proxy", "water_proxy"))
    if col is None: raise ValueError(
        "Benchmark results do not contain the observable water-context proxy field")
    return _numeric(bm_df[col])

def _parse_pair(value: Any) -> Optional[Tuple[float, float]]:
    if value is None: return None
    try:
        if pd.isna(value): return None
    except (TypeError, ValueError): pass

    candidate = value
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text: return None
        try: candidate = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
            if len(nums) != 2: return None
            candidate = (float(nums[0]), float(nums[1]))

    if isinstance(candidate, (tuple, list, np.ndarray, pd.Series)) and len(candidate) == 2:
        try: a, b = float(candidate[0]), float(candidate[1])
        except (TypeError, ValueError): return None
        if np.isfinite(a) and np.isfinite(b): return a, b
    return None

def _format_pair(value: Any) -> str:
    pair = _parse_pair(value)
    return "NA" if pair is None else f"[{pair[0]:.4f}, {pair[1]:.4f}]"

def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def _annotate_bars(ax, bars, fmt="{:.3f}", dy=0.012):
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
            value + dy * span, fmt.format(value),
            ha="center", va="bottom",
            fontsize=8.5, color="#333333")

# ------------------------------------------------------------------
# SYNTHESIS TABLES
# ------------------------------------------------------------------
def _table_1_statistics(stat_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(stat_df, (
        "comparison", "improvement_percent", "hedges_g", "p_value",
        "holm_adjusted_p_value"), "statistical_summary.csv")

    mean_difference_col = _first_existing_column(
        stat_df, ("mean_error_difference", "mean_difference"))
    if mean_difference_col is None:
        raise ValueError("statistical_summary.csv must contain mean_error_difference")

    bootstrap_col = _first_existing_column(stat_df, (
        "bootstrap_ci95_mean_difference", "bootstrap_ci_95", "confidence_interval_95"))
    significance_col = _first_existing_column(stat_df, (
        "holm_significant", "adjusted_and_directionally_supported", "is_significant"))

    rows = []
    for _, row in stat_df.iterrows():
        if significance_col:
            significant = _safe_bool(row.get(significance_col))
        else:
            try: significant = float(row["holm_adjusted_p_value"]) < 0.05
            except (TypeError, ValueError): significant = False

        n_value = pd.to_numeric(
            pd.Series([row.get("n_samples", np.nan)]), errors="coerce").iloc[0]
            
        rows.append({
            "Comparison": _comparison_label(row["comparison"]),
            "n": n_value,
            "Mean Error Reduction (cm)": row[mean_difference_col],
            "Improvement (%)": row["improvement_percent"],
            "Hedges' g": row["hedges_g"],
            "Holm-Adjusted p": _pvalue_text(row["holm_adjusted_p_value"]),
            "Bootstrap 95% CI (cm)": (_format_pair(row[bootstrap_col]) if bootstrap_col else "NA"),
            "Holm Significant": significant,
        })

    out = pd.DataFrame(rows)
    if out.empty: return out

    out["n"] = pd.to_numeric(out["n"], errors="coerce").round().astype("Int64")
    for col in ("Mean Error Reduction (cm)", "Improvement (%)", "Hedges' g"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    order = {"Adaptive vs Fixed": 0, "Adaptive vs LiDAR-only": 1}
    out["_order"] = out["Comparison"].map(order).fillna(99)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    out["Mean Error Reduction (cm)"] = out["Mean Error Reduction (cm)"].round(4)
    out["Improvement (%)"] = out["Improvement (%)"].round(2)
    out["Hedges' g"] = out["Hedges' g"].round(4)
    return out

def _table_2_domain_robustness(bm_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(bm_df, ("fixed_error", "adaptive_error"), "benchmark_results.csv")
    ntu = _ensure_observable_ntu(bm_df)

    temp = pd.DataFrame({
        "ntu": ntu,
        "fixed_error": _numeric(bm_df["fixed_error"]),
        "adaptive_error": _numeric(bm_df["adaptive_error"]),
    }).replace([np.inf, -np.inf], np.nan)
    temp = temp.dropna(subset=["ntu"])

    labels = [f"{int(a)}–{int(b)}" for a, b in zip(NTU_BIN_EDGES[:-1], NTU_BIN_EDGES[1:])]
    temp["NTU Bin"] = pd.cut(temp["ntu"].clip(*NTU_RANGE),
        bins=NTU_BIN_EDGES, include_lowest=True, right=False, labels=labels)

    rows = []
    for label, group in temp.groupby("NTU Bin", observed=False):
        if group.empty: continue
        fixed = group["fixed_error"].dropna()
        adaptive = group["adaptive_error"].dropna()
        paired = group[["fixed_error", "adaptive_error"]].dropna()
        
        fixed_mae = float(np.mean(np.abs(fixed))) if len(fixed) else np.nan
        adaptive_mae = float(np.mean(np.abs(adaptive))) if len(adaptive) else np.nan
        
        gain = ((fixed_mae - adaptive_mae) / fixed_mae * 100.0
            if np.isfinite(fixed_mae) and fixed_mae > 1e-12 and np.isfinite(adaptive_mae)
            else np.nan)
            
        win_rate = (float((paired["adaptive_error"].abs() < paired[
            "fixed_error"].abs()).mean() * 100.0) if len(paired) else np.nan)
            
        rows.append({
            "Observable NTU Range": str(label),
            "Samples": int(len(group)),
            "Fixed Fusion MAE (cm)": fixed_mae,
            "Adaptive Fusion MAE (cm)": adaptive_mae,
            "Adaptive Improvement (%)": gain,
            "Adaptive Win Rate (%)": win_rate,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        for col in out.columns[2:]:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out

def _table_3_ablation(ab_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ab_df, ("experiment", "error"), "ablation_results.csv")
    temp = ab_df.copy()
    temp["error"] = _numeric(temp["error"])

    grouped = (temp.groupby("experiment", sort=False)["error"].agg(
        Samples="count",
        Mean_Absolute_Error_cm="mean",
        Maximum_Error_cm="max",
        Error_Std_Dev="std").reset_index())

    label_map = {
        "A_All_Sensors": "A — All Sensors (Proposed)",
        "B_No_LiDAR": "B — No LiDAR",
        "C_No_Radar": "C — No Radar",
        "D_No_Ultra": "D — No Ultrasonic",
        "E_No_Reliability": "E — No Reliability Weighting",
        "F_No_Context": "F — No Context Adaptation",
        "G_No_Adaptive": "G — No Adaptive Fusion (Fixed)",
    }

    grouped["Architecture Variant"] = grouped[
        "experiment"].map(label_map).fillna(grouped["experiment"])
    proposed = grouped.loc[grouped["experiment"] == "A_All_Sensors", "Mean_Absolute_Error_cm"]
    proposed_mae = float(proposed.iloc[0]) if not proposed.empty else np.nan

    grouped["Degradation vs Proposed (%)"] = ((grouped[
        "Mean_Absolute_Error_cm"] - proposed_mae) / proposed_mae * 100.0
        if np.isfinite(proposed_mae) and proposed_mae > 1e-12 else np.nan)

    grouped = grouped[[
        "Architecture Variant", "Samples",
        "Mean_Absolute_Error_cm", "Maximum_Error_cm",
        "Error_Std_Dev", "Degradation vs Proposed (%)",
    ]].rename(columns={
        "Mean_Absolute_Error_cm": "Mean Absolute Error (cm)",
        "Maximum_Error_cm": "Maximum Error (cm)",
        "Error_Std_Dev": "Error Std. Dev. (cm)",
    })

    for col in (
        "Mean Absolute Error (cm)", "Maximum Error (cm)",
        "Error Std. Dev. (cm)", "Degradation vs Proposed (%)"):
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(3)
    return grouped.reset_index(drop=True)

def _table_4_validation(val_df: pd.DataFrame) -> pd.DataFrame:
    if val_df.empty: raise ValueError("validation_scores.csv is empty")
    row = val_df.iloc[0]
    rows = []
    for component in VALIDATION_COMPONENTS:
        value = pd.to_numeric(pd.Series([row.get(component, np.nan)]), errors="coerce").iloc[0]
        rows.append({
            "Evaluation Dimension": VALIDATION_LABELS[component],
            "Score (0–1)": float(value) if np.isfinite(value) else np.nan,
            "Assessment": _validation_band(float(value)) if np.isfinite(value) else "Unavailable",
        })

    overall = pd.to_numeric(pd.Series([
        row.get("validation_evidence_score", np.nan)]), errors="coerce").iloc[0]
    overall_label = row.get("validation_evidence_label",
        _validation_band(float(overall)) if np.isfinite(overall) else "Unavailable")

    rows.append({
        "Evaluation Dimension": "OVERALL VALIDATION EVIDENCE",
        "Score (0–1)": float(overall) if np.isfinite(overall) else np.nan,
        "Assessment": str(overall_label),
    })

    out = pd.DataFrame(rows)
    out["Score (0–1)"] = pd.to_numeric(out["Score (0–1)"], errors="coerce").round(3)
    return out

def generate_tables(
    bm_df: pd.DataFrame, ab_df: pd.DataFrame, stat_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    _table_1_statistics(stat_df).to_csv(TABLE_DIR / "statistical_significance.csv", index=False)
    _table_2_domain_robustness(bm_df).to_csv(TABLE_DIR / "domain_robustness.csv", index=False)
    _table_3_ablation(ab_df).to_csv(TABLE_DIR / "ablation_study.csv", index=False)
    _table_4_validation(val_df).to_csv(TABLE_DIR / "validation_audit.csv", index=False)

# ------------------------------------------------------------------
# VALIDATION FIGURES
# ------------------------------------------------------------------
def _plot_turbidity_performance(bm_df: pd.DataFrame) -> None:
    _require_columns(bm_df, ("fixed_error", "adaptive_error"), "benchmark_results.csv")
    ntu = _ensure_observable_ntu(bm_df)
    temp = pd.DataFrame({
        "ntu": ntu,
        "Fixed Bayesian Fusion": np.abs(_numeric(bm_df["fixed_error"])),
        "Adaptive Reliability-Aware Fusion": np.abs(_numeric(bm_df["adaptive_error"])),
    }).replace([np.inf, -np.inf], np.nan)

    labels = [f"{int(a)}–{int(b)}" for a, b in zip(NTU_BIN_EDGES[:-1], NTU_BIN_EDGES[1:])]
    temp["NTU Bin"] = pd.cut(temp["ntu"].clip(*NTU_RANGE),
        bins=NTU_BIN_EDGES, include_lowest=True, right=False, labels=labels)

    stats = []
    for label, group in temp.groupby("NTU Bin", observed=False):
        if group.empty: continue
        for method in ("Fixed Bayesian Fusion", "Adaptive Reliability-Aware Fusion"):
            values = group[method].dropna()
            if not len(values): continue
            low, high = _ci95_mean(values) if len(values) >= 2 else (np.nan, np.nan)
            stats.append({
                "center": (float(labels.index(str(label))) + 0.5) * 50.0,
                "method": method,
                "mean": float(values.mean()),
                "low": low, "high": high,
            })

    stats_df = pd.DataFrame(stats)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    for method, key in (
        ("Fixed Bayesian Fusion", "fixed"),
        ("Adaptive Reliability-Aware Fusion", "adaptive")):
        subset = stats_df.loc[stats_df["method"] == method].sort_values("center")
        if subset.empty: continue
        x = subset["center"].to_numpy(float)
        y = subset["mean"].to_numpy(float)
        
        ax.plot(x, y, color=COLORS[key], linewidth=2.2, marker="o", markersize=4.5,
        markeredgewidth=0.8, markeredgecolor="white", label=method, zorder=3)
        
        low = subset["low"].to_numpy(float)
        high = subset["high"].to_numpy(float)
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            ax.fill_between(x, low, high, color=COLORS[key], alpha=0.10, linewidth=0, zorder=1)

    ax.set_title("Depth Estimation Error Across Observable Turbidity", pad=10)
    ax.set_xlabel("Observable Water Turbidity (NTU)", labelpad=12)
    ax.set_ylabel("Mean Absolute Error (cm)", labelpad=12)
    ax.set_xlim(0, 500)
    ax.xaxis.set_major_locator(MaxNLocator(11))
    ax.grid(axis="y", linewidth=0.6, alpha=0.65)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", framealpha=1.0, borderpad=0.7)
    _save_figure(fig, FIG_DIR / "turbidity_performance.png")

def _plot_overall_performance(bm_df: pd.DataFrame) -> None:
    _require_columns(bm_df, (
        "single_error", "fixed_error", "adaptive_error"), "benchmark_results.csv")
    methods = ["LiDAR-only", "Fixed Bayesian Fusion", "Adaptive Reliability-Aware Fusion"]

    rmse = [
        np.sqrt(np.mean(np.square(_clean_numeric(bm_df["single_error"])))),
        np.sqrt(np.mean(np.square(_clean_numeric(bm_df["fixed_error"])))),
        np.sqrt(np.mean(np.square(_clean_numeric(bm_df["adaptive_error"]))))]
    mae = [
        np.mean(np.abs(_clean_numeric(bm_df["single_error"]))),
        np.mean(np.abs(_clean_numeric(bm_df["fixed_error"]))),
        np.mean(np.abs(_clean_numeric(bm_df["adaptive_error"])))]

    x = np.arange(len(methods))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)

    bars_rmse = ax.bar(x - width / 2, rmse, width, label="RMSE",
        color="#64748b", edgecolor="white", linewidth=0.8)        
    bars_mae = ax.bar(x + width / 2, mae, width, label="MAE",
        color="#0f766e", edgecolor="white", linewidth=0.8)

    ax.set_title("Overall Depth Estimation Accuracy", pad=10)
    ax.set_ylabel("Error (cm)", labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["LiDAR-only", "Fixed Fusion", "Adaptive Fusion"], rotation=0)
    ax.set_ylim(0, max(max(rmse), max(mae)) * 1.23)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", linewidth=0.6, alpha=0.65)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", framealpha=1.0, ncol=2)

    for bars in (bars_rmse, bars_mae): _annotate_bars(ax, bars, fmt="{:.3f}", dy=0.018)
    _save_figure(fig, FIG_DIR / "overall_accuracy_comparison.png")

def _plot_ablation_sensitivity(ab_df: pd.DataFrame) -> None:
    _require_columns(ab_df, ("experiment", "error"), "ablation_results.csv")
    label_map = {
        "A_All_Sensors": "All\nSensors",
        "B_No_LiDAR": "No\nLiDAR",
        "C_No_Radar": "No\nRadar",
        "D_No_Ultra": "No\nUltrasonic",
        "E_No_Reliability": "No Reliability\nWeighting",
        "F_No_Context": "No Context\nAdaptation",
        "G_No_Adaptive": "Fixed\nFusion",
    }

    order = list(label_map)
    grouped = (ab_df.assign(error=_numeric(ab_df["error"]))
        .groupby("experiment", sort=False)["error"].mean().to_dict())

    if "A_All_Sensors" not in grouped:
        raise ValueError("ablation_results.csv does not contain A_All_Sensors")

    baseline = float(grouped["A_All_Sensors"])
    degradation = [((float(grouped[name]) - baseline) / baseline * 100.0)
        if name in grouped and baseline > 1e-12 else np.nan for name in order]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    bars = ax.bar(x, degradation, width=0.68, color=[COLORS["adaptive"]
        if name == "A_All_Sensors" else "#94a3b8" for name in order],
        edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.set_title("Architecture Sensitivity Relative to the Proposed Fusion", pad=10)
    ax.set_ylabel("Error Degradation vs Proposed (%)", labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[name] for name in order])
    ax.yaxis.set_major_locator(MaxNLocator(7))
    ax.grid(axis="y", linewidth=0.6, alpha=0.65)
    ax.grid(axis="x", visible=False)

    finite = np.asarray(degradation, dtype=float)
    ymax = float(np.nanmax(finite)) if np.isfinite(finite).any() else 1.0
    ymin = float(np.nanmin(finite)) if np.isfinite(finite).any() else 0.0
    pad = max((ymax - ymin) * 0.18, 5.0)
    ax.set_ylim(min(-pad * 0.35, ymin - pad * 0.15), ymax + pad)

    for bar, value in zip(bars, degradation):
        if not np.isfinite(value): continue
        va = "bottom" if value >= 0 else "top"
        offset = 4 if value >= 0 else -5
        ax.annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset), textcoords="offset points",
            ha="center", va=va, fontsize=8.5, color="#333333")

    _save_figure(fig, FIG_DIR / "ablation_sensitivity.png")

def _plot_uncertainty_calibration(bm_df: pd.DataFrame) -> None:
    _require_columns(bm_df, ("true_depth", "adaptive_ci_contains_truth",
        "fixed_ci_contains_truth"), "benchmark_results.csv")

    temp = pd.DataFrame({
        "true_depth": _numeric(bm_df["true_depth"]),
        "Adaptive": _parse_bool_series(bm_df["adaptive_ci_contains_truth"]),
        "Fixed": _parse_bool_series(bm_df["fixed_ci_contains_truth"]),
    }).replace([np.inf, -np.inf], np.nan)
    temp = temp.dropna(subset=["true_depth"])

    labels = [f"{int(a)}–{int(b)}" for a, b in zip(DEPTH_BIN_EDGES[:-1], DEPTH_BIN_EDGES[1:])]
    temp["Depth Bin"] = pd.cut(temp["true_depth"].clip(*DEPTH_RANGE_CM),
        bins=DEPTH_BIN_EDGES, include_lowest=True, right=False, labels=labels)

    rows = []
    for label, group in temp.groupby("Depth Bin", observed=False):
        if group.empty: continue
        idx = labels.index(str(label))
        center = (DEPTH_BIN_EDGES[idx] + DEPTH_BIN_EDGES[idx + 1]) / 2.0
        rows.append({
            "center": center,
            "samples": len(group),
            "Fixed": group["Fixed"].mean() * 100.0,
            "Adaptive": group["Adaptive"].mean() * 100.0,
        })

    coverage = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    if not coverage.empty:
        ax.plot(coverage["center"], coverage["Fixed"],
            color=COLORS["fixed"], linewidth=2.1, marker="o",
            markersize=4.5, markeredgecolor="white", markeredgewidth=0.8,
            label="Fixed 95% interval")
            
        ax.plot(coverage["center"], coverage["Adaptive"],
            color=COLORS["adaptive"], linewidth=2.1, marker="o",
            markersize=4.5, markeredgecolor="white", markeredgewidth=0.8,
            label="Adaptive 95% interval")
            
    ax.axhline(95.0, color="#444444", linewidth=1.1, linestyle="--", label="Nominal 95% coverage")
    ax.set_title("Empirical Coverage of the 95% Posterior Interval", pad=10)
    ax.set_xlabel("True Pothole Depth (cm)", labelpad=12)
    ax.set_ylabel("Observed Coverage (%)", labelpad=12)
    ax.set_xlim(*DEPTH_RANGE_CM)
    ax.set_ylim(80.0, 101.0)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", linewidth=0.6, alpha=0.65)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", framealpha=1.0)
    _save_figure(fig, FIG_DIR / "uncertainty_calibration.png")

def generate_figures(bm_df: pd.DataFrame, ab_df: pd.DataFrame) -> None:
    for legacy_name in ("error_distribution_KDE.png", "operational_domain_coverage.png"):
        legacy_path = FIG_DIR / legacy_name
        if legacy_path.exists(): legacy_path.unlink()
        
    _plot_turbidity_performance(bm_df)
    _plot_overall_performance(bm_df)
    _plot_ablation_sensitivity(ab_df)
    _plot_uncertainty_calibration(bm_df)

# =====================================================================
# Public Entry Point
# =====================================================================
def generate_assets() -> None:
    bm_df, ab_df, stat_df, val_df = _load_data()
    generate_tables(bm_df, ab_df, stat_df, val_df)
    generate_figures(bm_df, ab_df)

    print("\n" + "=" * 60)
    print("FloodTwin-HIL Benchmarking Publication Assets")
    print("=" * 60)
    print("Table Directory   :", TABLE_DIR)
    print("Figure Directory  :", FIG_DIR)
    print("\nGenerated Figures :")
    print("-" * 60)
    print("✓ turbidity_performance.png")
    print("✓ overall_accuracy_comparison.png")
    print("✓ ablation_sensitivity.png")
    print("✓ uncertainty_calibration.png")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    generate_assets()