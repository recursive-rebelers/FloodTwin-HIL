import ast
import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
from matplotlib.lines import Line2D
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, r2_score)

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Small Utilities
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        if not np.isfinite(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

def _sanitize_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna("—")

def _sanitize_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return _sanitize_json(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _sanitize_json(obj.tolist())
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj

def _write_json(payload: Dict[str, Any], path: Path) -> None:
    _ensure_dir(path.parent)
    payload = _sanitize_json(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default, allow_nan=False)

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df = _sanitize_table(df)
    df.to_csv(path, index=False, encoding="utf-8")

def _fmt(x: Any, digits: int = 4) -> Any:
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        return round(float(x), digits)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x

def _format_table_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows available_\n"
    tmp = df.copy()
    if len(tmp) > max_rows:
        tmp = tmp.head(max_rows)
    try:
        return tmp.to_markdown(index=False) + "\n"
    except Exception:
        return "```\n" + tmp.to_string(index=False) + "\n```\n"

def _clean_md_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return "—" if not np.isfinite(value) else f"{float(value):.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        text = json.dumps(_sanitize_json(value), ensure_ascii=False, default=_json_default, allow_nan=False)
        return text if len(text) <= 120 else text[:117] + "..."
    return str(value)

def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available_\n"
    work = df.copy()
    for col in work.columns:
        work[col] = work[col].map(_clean_md_value)
    try:
        return work.to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(work.columns.astype(str)) + " |"
        sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
        rows = []
        for row in work.itertuples(index=False):
            rows.append("| " + " | ".join(_clean_md_value(v) for v in row) + " |")
        return "\n".join([header, sep, *rows])

def _section(title: str, df: pd.DataFrame) -> str:
    return f"## {title}\n\n{_df_to_markdown_table(df)}\n\n"

def _parse_maybe_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s[0] not in "[{":
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(s)
        except Exception:
            continue
    return value

def _parse_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            sample = out[col].dropna().astype(str).head(20)
            if len(sample) and sample.map(lambda s: s.strip().startswith(("{", "["))).any():
                out[col] = out[col].map(_parse_maybe_json)
    return out

def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(arr).sum() == 0:
        return float("nan")
    return float(np.nanmean(arr))

def _safe_std(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(arr).sum() == 0:
        return float("nan")
    return float(np.nanstd(arr))

def _safe_median(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(arr).sum() == 0:
        return float("nan")
    return float(np.nanmedian(arr))

def _safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, q))

def _safe_auc_from_ranks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    r_pos = ranks[: len(pos)].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(x).rank(method="average").to_numpy()
    y = pd.Series(y).rank(method="average").to_numpy()
    return _pearsonr(x, y)

def _interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    y_true = y_true[mask]
    lower = lower[mask]
    upper = upper[mask]
    if len(y_true) == 0: return float("nan")
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def _winkler_interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.05) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    y_true = y_true[mask]
    lower = lower[mask]
    upper = upper[mask]
    if len(y_true) == 0: return float("nan")
    width = upper - lower
    below = (y_true < lower).astype(float)
    above = (y_true > upper).astype(float)
    score = width + (2.0 / alpha) * (lower - y_true) * below + (2.0 / alpha) * (y_true - upper) * above
    return float(np.mean(score))

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    mask = np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if len(y_true) == 0: return {"ece": float("nan"), "mce": float("nan"), "bins": []}
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=False) + 1
    rows = []
    ece = 0.0
    mce = 0.0

    for b in range(1, n_bins + 1):
        idx = bin_ids == b
        if not np.any(idx): continue
        conf = float(np.mean(y_prob[idx]))
        acc = float(np.mean(y_true[idx]))
        gap = abs(acc - conf)
        weight = float(np.mean(idx))
        ece += weight * gap
        mce = max(mce, gap)
        rows.append({
            "bin": f"{bins[b - 1]:.1f}-{bins[b]:.1f}",
            "count": int(np.sum(idx)),
            "mean_confidence": conf,
            "empirical_accuracy": acc,
            "gap": gap
        })

    return {"ece": float(ece), "mce": float(mce), "bins": rows}

def _binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / max(len(y_true), 1)

    f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(
        precision) and np.isfinite(recall) and (precision + recall) else float("nan")
    bal_acc = np.nanmean([recall, specificity])
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else float("nan")

    out = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        mask = np.isfinite(y_prob)

        y_prob_valid = y_prob[mask]
        y_true_valid = y_true[mask]

        if (len(y_prob_valid) > 0 and len(np.unique(y_true_valid)) > 1):
            try:
                if SKLEARN_AVAILABLE:
                    out["roc_auc"] = float(roc_auc_score(y_true_valid, y_prob_valid))
                    out["pr_auc"] = float(average_precision_score(y_true_valid, y_prob_valid))
                    out["brier_score"] = float(brier_score_loss(y_true_valid, np.clip(y_prob_valid, 0.0, 1.0)))
                else:
                    out["roc_auc"] = _safe_auc_from_ranks(y_true_valid, y_prob_valid)
                    out["pr_auc"] = float("nan")
                    out["brier_score"] = float(np.mean((np.clip(y_prob_valid, 0.0, 1.0) - y_true_valid) ** 2))
            except Exception:
                out["roc_auc"] = float("nan")
                out["pr_auc"] = float("nan")
                out["brier_score"] = float("nan")
        else:
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")
            out["brier_score"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["brier_score"] = float("nan")
    return out

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "median_ae": float("nan"),
            "max_ae": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
            "mape_percent": float("nan"),
        }

    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    median_ae = float(np.median(abs_err))
    max_ae = float(np.max(abs_err))
    bias = float(np.mean(y_pred - y_true))
    mape = float(np.mean(abs_err / np.clip(np.abs(y_true), 1e-12, None)) * 100.0)

    if len(np.unique(y_true)) > 1:
        try:
            if SKLEARN_AVAILABLE:
                r2 = float(r2_score(y_true, y_pred))
            else:
                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        except Exception:
            r2 = float("nan")
    else:
        r2 = float("nan")

    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "max_ae": max_ae,
        "bias": bias,
        "r2": r2,
        "mape_percent": mape,
    }

# ---------------------------------------------------------------------------
# Input Loading / Normalization
# ---------------------------------------------------------------------------

ALIASES = {
    "scenario_id": ["scenario_id", "id"],
    "road_type": ["road_type", "surface_type", "pavement_type"],
    "road_environment": ["road_environment", "environment", "traffic_zone"],
    "weather": ["weather"],
    "lighting": ["lighting"],
    "vehicle_speed": ["vehicle_speed", "speed_kmh", "speed"],
    "pothole_depth": ["pothole_depth", "true_depth", "depth_cm"],
    "true_depth": ["true_depth", "pothole_depth", "depth_cm"],
    "water_depth": ["water_depth", "flood_depth", "water_cm"],
    "true_ntu": ["true_ntu", "ntu", "turbidity_ntu"],
    "ntu": ["ntu", "true_ntu", "turbidity_ntu"],
    "lidar": ["lidar", "lidar_distance", "lidar_reading"],
    "radar": ["radar", "radar_distance", "radar_reading"],
    "ultrasonic": ["ultrasonic", "ultrasonic_distance", "ultrasonic_reading"],
    "imu_pitch": ["imu_pitch", "pitch"],
    "imu_roll": ["imu_roll", "roll"],
    "imu_acceleration": ["imu_acceleration", "acceleration"],
    "turbidity": ["turbidity"],
    "water_contact": ["water_contact", "water_present"],
    "hazard_probability": ["hazard_probability", "pred_hazard_probability",
    "posterior_hazard_probability", "risk_probability"],
    "status": ["status"],
    "posterior_confidence": ["posterior_confidence", "fusion_confidence", "uncertainty_confidence"],
    "confidence": ["confidence", "posterior_confidence", "fusion_confidence", "uncertainty_confidence"],
    "expected_depth": ["expected_depth"],
    "map_depth": ["map_depth"],
    "ci_lower": ["ci_lower"],
    "ci_upper": ["ci_upper"],
    "fault_profile": ["fault_profile", "profile_name"],
    "sensor_ms": ["sensor_ms"],
    "fault_ms": ["fault_ms"],
    "reliability_ms": ["reliability_ms"],
    "fusion_ms": ["fusion_ms"],
    "hazard_ms": ["hazard_ms"],
    "total_ms": ["total_ms"],
    "r_lidar": ["r_lidar"],
    "r_radar": ["r_radar"],
    "r_ultrasonic": ["r_ultrasonic"],
    "r_imu": ["r_imu"],
    "r_turbidity": ["r_turbidity"],
    "r_water": ["r_water", "water_reliability", "r_water_contact"],
    "scenario_label": ["scenario_label"],
    "experiment_profile": ["experiment_profile"],
    "source_mode": ["source_mode"],
}

def _pick_first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_input(path: Path, deduplicate: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    df = _parse_object_columns(df)

    if deduplicate and "scenario_id" in df.columns:
        before = len(df)
        sort_cols = [c for c in ["frame_idx", "timestamp_s", "resource_timestamp_s"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        df = df.drop_duplicates(subset=["scenario_id"], keep="last").reset_index(drop=True)
        after = len(df)
        if after != before:
            warnings.warn(f"Deduplicated scenario rows by scenario_id: {before} -> {after}")
    return df

def normalize_experiment_frame(df: pd.DataFrame, default_threshold_cm: float,
    hazard_pred_threshold: float) -> pd.DataFrame:
    out = df.copy()

    if "scenario_id" in out.columns:
        out["scenario_id"] = out["scenario_id"].astype(str)
    if "threshold_cm" not in out.columns:
        out["threshold_cm"] = default_threshold_cm

    out["threshold_cm"] = pd.to_numeric(out["threshold_cm"], errors="coerce").fillna(default_threshold_cm)
    gt_col = _pick_first_present(out, ["hazard_ground_truth", "ground_truth_hazard", "gt_hazard", "true_hazard",
    "hazard_label_true"])

    if gt_col is not None:
        out["true_hazard"] = pd.to_numeric(out[gt_col], errors="coerce").fillna(0).astype(int)
    else:
        if "true_depth" not in out.columns and "pothole_depth" in out.columns:
            out["true_depth"] = out["pothole_depth"]
        if "true_depth" not in out.columns:
            out["true_depth"] = np.nan

        out["true_depth"] = pd.to_numeric(out["true_depth"], errors="coerce")
        out["true_hazard"] = (out["true_depth"] >= out["threshold_cm"]).astype(int)

    prob_col = _pick_first_present(out, [
        "hazard_probability",
        "pred_hazard_probability",
        "posterior_hazard_probability",
        "risk_probability",
    ])

    if prob_col is not None:
        out["hazard_probability"] = pd.to_numeric(out[prob_col], errors="coerce")
    else:
        out["hazard_probability"] = np.nan
    
    invalid_prob = (out["hazard_probability"].notna() & (
        (out["hazard_probability"] < 0.0) | (out["hazard_probability"] > 1.0)))

    if invalid_prob.any():
        warnings.warn(f"{int(invalid_prob.sum())} hazard probabilities were outside [0, 1] and have been clipped")

    out["hazard_probability"] = (out["hazard_probability"].clip(lower=0.0, upper=1.0))

    if "status" in out.columns:
        out["pred_hazard"] = out["status"].astype(str).str.upper().eq("HAZARD").astype(int)
    else:
        out["pred_hazard"] = (out["hazard_probability"] >= hazard_pred_threshold).astype(int)

    conf_col = _pick_first_present(out, ["posterior_confidence", "fusion_confidence", "uncertainty_confidence", "confidence"])

    if conf_col is not None:
        out["confidence"] = pd.to_numeric(out[conf_col], errors="coerce")
    else:
        out["confidence"] = np.nan
        
    out["confidence"] = out["confidence"].clip(lower=0.0, upper=1.0)
    if "expected_depth" not in out.columns and "map_depth" not in out.columns and "posterior_summary" in out.columns:

        def _extract_expected_depth(v: Any) -> float:
            if isinstance(v, dict):
                return _safe_float(v.get("expected_depth", np.nan), np.nan)
            return np.nan
        out["expected_depth"] = out["posterior_summary"].map(_extract_expected_depth)

    if "expected_depth" in out.columns:
        out["pred_depth"] = pd.to_numeric(out["expected_depth"], errors="coerce")
    elif "map_depth" in out.columns:
        out["pred_depth"] = pd.to_numeric(out["map_depth"], errors="coerce")
    else:
        out["pred_depth"] = np.nan

    if "ci_lower" not in out.columns:
        out["ci_lower"] = np.nan
    if "ci_upper" not in out.columns:
        out["ci_upper"] = np.nan
    if "uncertainty" in out.columns:

        def _get_unc(v: Any, key: str) -> float:
            if isinstance(v, dict):
                return _safe_float(v.get(key, np.nan), np.nan)
            return np.nan

        missing_lower = out["ci_lower"].isna()
        missing_upper = out["ci_upper"].isna()

        if missing_lower.any():
            out.loc[missing_lower, "ci_lower"] = out.loc[missing_lower, "uncertainty"].map(lambda v: _get_unc(v, "ci_lower"))
        if missing_upper.any():
            out.loc[missing_upper, "ci_upper"] = out.loc[missing_upper, "uncertainty"].map(lambda v: _get_unc(v, "ci_upper"))

    out["ci_lower"] = pd.to_numeric(out["ci_lower"], errors="coerce")
    out["ci_upper"] = pd.to_numeric(out["ci_upper"], errors="coerce")
    out["ci_width_95"] = out["ci_upper"] - out["ci_lower"]

    out["true_depth"] = pd.to_numeric(out["true_depth"], errors="coerce")
    out["pred_depth"] = pd.to_numeric(out["pred_depth"], errors="coerce")
    out["abs_error"] = (out["pred_depth"] - out["true_depth"]).abs()
    out["signed_error"] = out["pred_depth"] - out["true_depth"]
    out["relative_error_pct"] = out["abs_error"] / np.clip(out["true_depth"].abs(), 1e-12, None) * 100.0

    if "fault_profile" in out.columns:
        out["fault_group"] = out["fault_profile"].astype(str)
    elif "profile_name" in out.columns:
        out["fault_group"] = out["profile_name"].astype(str)
    else:
        out["fault_group"] = "unknown"

    for col in ["sensor_ms", "fault_ms", "reliability_ms", "fusion_ms", "hazard_ms", "total_ms",
    "process_rss_mb", "process_cpu_percent", "system_cpu_percent"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

# ---------------------------------------------------------------------------
# Experiment Discovery and Metric Extraction
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMetrics:
    experiment: str
    source_mode: str
    profile_name: str
    input_rows: int
    used_rows: int
    unique_scenarios: int
    detection: Dict[str, Any]
    regression: Dict[str, Any]
    uncertainty: Dict[str, Any]
    reliability: Dict[str, Any]
    runtime: Dict[str, Any]
    robustness: Dict[str, Any]
    notes: Dict[str, Any]

def discover_experiments(results_root: Path, scenario_filename: str = "virtual_edge_scenarios.csv") -> List[Path]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    candidates: List[Path] = []
    for p in sorted(results_root.iterdir()):
        if p.is_dir() and (p / scenario_filename).exists():
            candidates.append(p)
    return candidates

def evaluate_experiment(
    experiment_dir: Path,
    threshold_cm: float = 10.0,
    hazard_pred_threshold: float = 0.5,
    deduplicate: bool = True) -> Tuple[ExperimentMetrics, pd.DataFrame]:

    input_csv = experiment_dir / "virtual_edge_scenarios.csv"
    raw_df = pd.read_csv(input_csv)
    input_rows = len(raw_df)
    df = load_input(input_csv, deduplicate=deduplicate)
    df = normalize_experiment_frame(df, default_threshold_cm=threshold_cm,
    hazard_pred_threshold=hazard_pred_threshold)
    used_rows = len(df)

    y_true = df["true_hazard"].to_numpy(dtype=int)
    y_pred = df["pred_hazard"].to_numpy(dtype=int)
    y_prob = pd.to_numeric(df["hazard_probability"], errors="coerce").to_numpy(dtype=float)
    true_depth = pd.to_numeric(df["true_depth"], errors="coerce").to_numpy(dtype=float)
    pred_depth = pd.to_numeric(df["pred_depth"], errors="coerce").to_numpy(dtype=float)
    conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(dtype=float)
    abs_err = pd.to_numeric(df["abs_error"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(df["ci_lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(df["ci_upper"], errors="coerce").to_numpy(dtype=float)

    detection = _binary_classification_metrics(y_true, y_pred, y_prob)
    regression = _regression_metrics(true_depth, pred_depth)

    calib = _ece(y_true, y_prob, n_bins=10)
    valid_ci_mask = (
        np.isfinite(true_depth)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & (upper >= lower))
    valid_ci_count = int(np.sum(valid_ci_mask))

    uncertainty = {
        "ece": calib["ece"],
        "mce": calib["mce"],
        "confidence_error_pearson": _pearsonr(conf, abs_err),
        "confidence_error_spearman": _spearmanr(conf, abs_err),

        "interval_coverage_95": _interval_coverage(
            true_depth[valid_ci_mask],
            lower[valid_ci_mask],
            upper[valid_ci_mask]),
        "interval_width_95_mean": (float(np.mean(
            upper[valid_ci_mask] - lower[valid_ci_mask]))
            if valid_ci_count > 0 else float("nan")),
        "winkler_score_95": _winkler_interval_score(
            true_depth[valid_ci_mask],
            lower[valid_ci_mask],
            upper[valid_ci_mask], alpha=0.05),

        "valid_ci_count": valid_ci_count,
        "brier_score": detection.get("brier_score", float("nan")),
        "calibration_counts_total": int(sum(row["count"] for row in calib["bins"])) if calib["bins"] else 0,
        "calibration_n_bins": 10,
        "calibration_bins": calib["bins"],
    }

    reliability: Dict[str, Any] = {}
    for sensor_col, out_key in [
        ("r_lidar", "r_lidar"),
        ("r_ultrasonic", "r_ultrasonic"),
        ("r_radar", "r_radar"),
        ("r_imu", "r_imu"),
        ("r_turbidity", "r_turbidity"),
        ("r_water", "r_water")]:

        if sensor_col in df.columns:
            r = pd.to_numeric(df[sensor_col], errors="coerce").to_numpy(dtype=float)
            reliability[f"{out_key}_mean"] = _safe_mean(df[sensor_col])
            reliability[f"{out_key}_std"] = _safe_std(df[sensor_col])
            reliability[f"{out_key}_pearson"] = _pearsonr(r, abs_err)
            reliability[f"{out_key}_spearman"] = _spearmanr(r, abs_err)
            reliability[f"{out_key}_points"] = int(np.isfinite(r).sum())

    runtime = {
        "avg_digital_twin_ms": _safe_mean(df["sensor_ms"]) if "sensor_ms" in df.columns else float("nan"),
        "avg_fault_ms": _safe_mean(df["fault_ms"]) if "fault_ms" in df.columns else float("nan"),
        "avg_reliability_ms": _safe_mean(df["reliability_ms"]) if "reliability_ms" in df.columns else float("nan"),
        "avg_fusion_ms": _safe_mean(df["fusion_ms"]) if "fusion_ms" in df.columns else float("nan"),
        "avg_hazard_ms": _safe_mean(df["hazard_ms"]) if "hazard_ms" in df.columns else float("nan"),
        "avg_total_ms": _safe_mean(df["total_ms"]) if "total_ms" in df.columns else float("nan"),
        "p95_total_ms": _safe_quantile(df["total_ms"], 0.95) if "total_ms" in df.columns else float("nan"),
        "throughput_sps": (1000.0 / _safe_mean(df["total_ms"])) if "total_ms" in df.columns and np.isfinite(
            _safe_mean(df["total_ms"])) and _safe_mean(df["total_ms"]) > 0 else float("nan"),
        "peak_process_rss_mb": float(np.nanmax(pd.to_numeric(df["process_rss_mb"],
        errors="coerce"))) if "process_rss_mb" in df.columns and pd.to_numeric(df["process_rss_mb"],
        errors="coerce").notna().any() else float("nan"),
        "mean_process_cpu_percent": _safe_mean(
            df["process_cpu_percent"]) if "process_cpu_percent" in df.columns else float("nan"),
        "mean_system_cpu_percent": _safe_mean(
            df["system_cpu_percent"]) if "system_cpu_percent" in df.columns else float("nan")}

    rows = []
    for group, g in df.groupby("fault_group"):
        yy_true = pd.to_numeric(g["true_hazard"], errors="coerce").to_numpy(dtype=int)
        yy_pred = pd.to_numeric(g["pred_hazard"], errors="coerce").to_numpy(dtype=int)
        yy_prob = pd.to_numeric(g["hazard_probability"],
        errors="coerce").to_numpy(dtype=float) if "hazard_probability" in g.columns else None

        det = _binary_classification_metrics(yy_true, yy_pred, yy_prob)
        reg = _regression_metrics(
            pd.to_numeric(g["true_depth"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(g["pred_depth"], errors="coerce").to_numpy(dtype=float))

        rows.append({
            "fault_group": str(group),
            "n": int(len(g)),
            "accuracy": det["accuracy"],
            "f1": det["f1"],
            "roc_auc": det["roc_auc"],
            "mae": reg["mae"],
            "rmse": reg["rmse"],
            "mean_confidence": _safe_mean(g["confidence"]) if "confidence" in g.columns else float("nan"),
            "mean_latency_ms": _safe_mean(g["total_ms"]) if "total_ms" in g.columns else float("nan"),
        })

    robustness = pd.DataFrame(rows).sort_values(by="fault_group").reset_index(drop=True)
    if robustness.empty:
        robustness = pd.DataFrame([{
            "fault_group": "N/A",
            "n": 0,
            "accuracy": None,
            "f1": None,
            "roc_auc": None,
            "mae": None,
            "rmse": None,
            "mean_confidence": None,
            "mean_latency_ms": None,
        }])

    experiment_name = experiment_dir.name
    source_mode = str(df["source_mode"].iloc[0]) if "source_mode" in df.columns and len(df) else "unknown"
    profile_name = str(df["profile_name"].iloc[0]) if "profile_name" in df.columns and len(df) else "unknown"
    unique_scenarios = int(df["scenario_id"].nunique()) if "scenario_id" in df.columns else used_rows

    notes = {
        "input_csv": str(input_csv),
        "deduplicated": bool(deduplicate),
        "duplicated_rows_removed": int(input_rows - used_rows),
        "has_json_columns": bool(any(df[c].dtype == object for c in df.columns)),
    }

    metrics = ExperimentMetrics(
        experiment=experiment_name,
        source_mode=source_mode,
        profile_name=profile_name,
        input_rows=int(input_rows),
        used_rows=int(used_rows),
        unique_scenarios=unique_scenarios,
        detection=detection,
        regression=regression,
        uncertainty=uncertainty,
        reliability=reliability,
        runtime=runtime,
        robustness={
            "rows": robustness,
            "groups": int(len(robustness)),
            "top_group_by_mae": str(robustness.sort_values(by="mae",
            ascending=True).iloc[0]["fault_group"]) if "mae" in robustness.columns and len(robustness) else None,
            "top_group_by_f1": str(robustness.sort_values(by="f1",
            ascending=False).iloc[0]["fault_group"]) if "f1" in robustness.columns and len(robustness) else None,
        }, notes=notes)
    return metrics, df

# ---------------------------------------------------------------------------
# Comparative Analysis
# ---------------------------------------------------------------------------

def _relative_change(value: float, baseline: float, better_when_lower: bool = False) -> float:
    if (not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-12):
        return float("nan")
    raw_change = ((value - baseline) / abs(baseline) * 100.0)
    if better_when_lower:
        return raw_change
    return -raw_change

def _rank_series(series: pd.Series, ascending: bool = False) -> pd.Series:
    return series.rank(method="min", ascending=ascending)

def _normalize_minmax(series: pd.Series, reverse: bool = False) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi - lo <= 1e-12:
        vals = pd.Series([0.5] * len(series), index=series.index)
    else:
        vals = (x - lo) / (hi - lo)
    if reverse:
        vals = 1.0 - vals
    return vals

def compare_experiments(metrics_list: List[ExperimentMetrics],
    baseline_experiment: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows = []
    for m in metrics_list:
        rows.append({
            "experiment": m.experiment,
            "source_mode": m.source_mode,
            "profile_name": m.profile_name,
            "input_rows": m.input_rows,
            "used_rows": m.used_rows,
            "unique_scenarios": m.unique_scenarios,
            "accuracy": m.detection.get("accuracy", np.nan),
            "precision": m.detection.get("precision", np.nan),
            "recall": m.detection.get("recall", np.nan),
            "specificity": m.detection.get("specificity", np.nan),
            "f1": m.detection.get("f1", np.nan),
            "balanced_accuracy": m.detection.get("balanced_accuracy", np.nan),
            "mcc": m.detection.get("mcc", np.nan),
            "roc_auc": m.detection.get("roc_auc", np.nan),
            "pr_auc": m.detection.get("pr_auc", np.nan),
            "brier_score": m.detection.get("brier_score", np.nan),
            "mae": m.regression.get("mae", np.nan),
            "rmse": m.regression.get("rmse", np.nan),
            "median_ae": m.regression.get("median_ae", np.nan),
            "max_ae": m.regression.get("max_ae", np.nan),
            "bias": m.regression.get("bias", np.nan),
            "r2": m.regression.get("r2", np.nan),
            "mape_percent": m.regression.get("mape_percent", np.nan),
            "ece": m.uncertainty.get("ece", np.nan),
            "mce": m.uncertainty.get("mce", np.nan),
            "interval_coverage_95": m.uncertainty.get("interval_coverage_95", np.nan),
            "interval_width_95_mean": m.uncertainty.get("interval_width_95_mean", np.nan),
            "winkler_score_95": m.uncertainty.get("winkler_score_95", np.nan),
            "confidence_error_pearson": m.uncertainty.get("confidence_error_pearson", np.nan),
            "confidence_error_spearman": m.uncertainty.get("confidence_error_spearman", np.nan),
            "avg_total_ms": m.runtime.get("avg_total_ms", np.nan),
            "p95_total_ms": m.runtime.get("p95_total_ms", np.nan),
            "throughput_sps": m.runtime.get("throughput_sps", np.nan),
            "peak_process_rss_mb": m.runtime.get("peak_process_rss_mb", np.nan),
            "mean_process_cpu_percent": m.runtime.get("mean_process_cpu_percent", np.nan),
            "mean_system_cpu_percent": m.runtime.get("mean_system_cpu_percent", np.nan),
            "r_lidar_mean": m.reliability.get("r_lidar_mean", np.nan),
            "r_lidar_pearson": m.reliability.get("r_lidar_pearson", np.nan),
            "r_ultrasonic_mean": m.reliability.get("r_ultrasonic_mean", np.nan),
            "r_ultrasonic_pearson": m.reliability.get("r_ultrasonic_pearson", np.nan),
            "r_radar_mean": m.reliability.get("r_radar_mean", np.nan),
            "r_radar_pearson": m.reliability.get("r_radar_pearson", np.nan),
        })

    summary = pd.DataFrame(rows)
    baseline_name = baseline_experiment

    if baseline_name is None:
        preferred_baselines = ["baseline_dataset", "baseline_registry"]
        baseline_name = next((name for name in preferred_baselines if name in summary["experiment"].values), None)
        if baseline_name is None:
            raise ValueError("No baseline experiment was explicitly provided and no known baseline experiment was found")

    baseline_row = summary[summary["experiment"] == baseline_name]
    if baseline_row.empty:
        available = (summary["experiment"].astype(str).tolist())
        raise ValueError(f"Baseline experiment '{baseline_name}' was not found. Available experiments: {available}")
    b = baseline_row.iloc[0].to_dict()

    rel = summary.copy()
    for col in ["f1", "balanced_accuracy", "roc_auc", "pr_auc", "mcc", "precision", "recall", "specificity", "throughput_sps"]:
        if col in rel.columns:
            rel[f"delta_{col}_pct"] = rel[col].apply(lambda v: _relative_change(v, b[col], better_when_lower=False))

    for col in ["mae", "rmse", "ece", "mce", "avg_total_ms", "p95_total_ms",
    "peak_process_rss_mb", "interval_width_95_mean", "winkler_score_95"]:
        if col in rel.columns:
            rel[f"delta_{col}_pct"] = rel[col].apply(lambda v: _relative_change(v, b[col], better_when_lower=True))

    rel["rank_f1"] = _rank_series(rel["f1"], ascending=False)
    rel["rank_mae"] = _rank_series(rel["mae"], ascending=True)
    rel["rank_ece"] = _rank_series(rel["ece"], ascending=True)
    rel["rank_total_ms"] = _rank_series(rel["avg_total_ms"], ascending=True)
    rel["rank_robustness"] = _rank_series(rel["mcc"].fillna(-np.inf), ascending=False)
    rel["rank_overall"] = rel[["rank_f1", "rank_mae", "rank_ece", "rank_total_ms", "rank_robustness"]].mean(axis=1)

    analysis = {
        "baseline_experiment": baseline_name,
        "top_f1_experiment": str(summary.sort_values(by="f1", ascending=False).iloc[0]["experiment"]),
        "top_mae_experiment": str(summary.sort_values(by="mae", ascending=True).iloc[0]["experiment"]),
        "top_ece_experiment": str(summary.sort_values(by="ece", ascending=True).iloc[0]["experiment"]),
        "fastest_experiment": str(summary.sort_values(by="avg_total_ms", ascending=True).iloc[0]["experiment"]),
        "best_ranking_experiment": str(rel.sort_values(by="rank_overall", ascending=True).iloc[0]["experiment"])}
    return summary, rel, analysis

def _validate_comparison_frame(comparison: pd.DataFrame) -> None:
    if comparison is None or comparison.empty:
        raise ValueError("Comparison DataFrame is empty")

    required = ["experiment", "f1", "mae", "rmse", "ece"]
    missing = [col for col in required if col not in comparison.columns]
    if missing:
        raise ValueError(f"Missing required comparison columns: {missing}")

    if comparison["experiment"].duplicated().any():
        duplicates = (comparison.loc[comparison["experiment"].duplicated(), "experiment"].astype(str).tolist())
        raise ValueError("Duplicate experiment names found: " f"{duplicates}")

    numeric_columns = ["f1", "mae", "rmse", "ece"]
    for col in numeric_columns:
        values = pd.to_numeric(comparison[col], errors="coerce")
        if not np.isfinite(values).any():
            warnings.warn(f"No finite values available for {col}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12.5,
    "axes.labelsize": 10.5,
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

def _save_fig(path: Path, dpi: int = 600) -> None:
    _ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def _metric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)

def _normalize_metric(
    series: pd.Series,
    higher_is_better: bool = True,
    clip_quantiles: Tuple[float, float] = (0.05, 0.95)) -> pd.Series:

    s = pd.to_numeric(series, errors="coerce").astype(float)
    valid = s[np.isfinite(s)]
    if valid.empty:
        return pd.Series([np.nan] * len(s), index=s.index)

    if len(valid) >= 3:
        lo = float(np.nanquantile(valid, clip_quantiles[0]))
        hi = float(np.nanquantile(valid, clip_quantiles[1]))
    else:
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))

    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        vals = pd.Series([0.5] * len(s), index=s.index, dtype=float)
    else:
        clipped = s.clip(lower=lo, upper=hi)
        vals = (clipped - lo) / (hi - lo)
    if not higher_is_better:
        vals = 1.0 - vals

    vals = vals.fillna(float(np.nanmedian(vals.to_numpy(dtype=float))) if np.isfinite(
        np.nanmedian(vals.to_numpy(dtype=float))) else 0.5)
    return vals.clip(0.0, 1.0)

def _annotate_points(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str,
    label_col: str = "experiment", fontsize: int = 8, plot_type: str = "detection") -> None:
    valid_rows = []

    detection_label_offsets = {
        "baseline_dataset": (-25, 0),
        "baseline_registry": (-25, 10),
        "complex_environment": (-30, 5),
        "compound_fault": (-35, -20),
        "flood_severity": (-25, 25),
        "high_speed": (25, -15),
        "muddy_water": (-30, 20),
        "sensor_failure": (30, -20),
        "stress_test": (-30, -15),
        "uncertainty_analysis": (-30, -20)
    }

    regression_label_offsets = {
        "baseline_dataset": (18, 0),
        "baseline_registry": (0, 43),
        "complex_environment": (30, -5),
        "compound_fault": (-20, 15),
        "flood_severity": (-1, 22),
        "high_speed": (-15, -12),
        "muddy_water": (-10, 30),
        "sensor_failure": (-10, 30),
        "stress_test": (-15, -18),
        "uncertainty_analysis": (35, -1),
    }

    runtime_label_offsets = {
        "baseline_dataset": (18, -25),
        "baseline_registry": (-18, -18),
        "complex_environment": (-20, -40),
        "compound_fault": (-20, -15),
        "flood_severity": (10, 22),
        "high_speed": (-15, -10),
        "muddy_water": (-10, -10),
        "sensor_failure": (-10, -15),
        "stress_test": (-15, 20),
        "uncertainty_analysis": (15, 30),
    }

    if plot_type == "detection":
        label_offsets = detection_label_offsets
    elif plot_type == "regression":
        label_offsets = regression_label_offsets
    elif plot_type == "runtime":
        label_offsets = runtime_label_offsets
    else:
        raise ValueError(f"Unknown plot_type: {plot_type!r}")

    for _, row in df.iterrows():
        x = _safe_float(row.get(x_col, np.nan))
        y = _safe_float(row.get(y_col, np.nan))
        if not (np.isfinite(x) and np.isfinite(y)): continue

        label = str(row.get(label_col, ""))
        dx, dy = label_offsets.get(label, (18, 18))
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"

        ax.annotate(label, xy=(x, y), xytext=(dx, dy),
            textcoords="offset points", ha=ha, va=va, fontsize=fontsize, color="0.15", annotation_clip=False,
            bbox=dict(boxstyle="round, pad=0.18", fc="white", ec="0.82", alpha=0.90),
            arrowprops=dict(arrowstyle="-", color="0.50", lw=0.7, shrinkA=0, shrinkB=4), zorder=5)

def fig_performance_landscape(comparison: pd.DataFrame, path: Path) -> None:
    metric_specs = [
        ("f1", True, "F1"),
        ("mcc", True, "MCC"),
        ("mae", False, "MAE"),
        ("rmse", False, "RMSE"),
        ("ece", False, "ECE"),
        ("avg_total_ms", False, "Latency"),
        ("throughput_sps", True, "Through\n-put"),
        ("r_lidar_pearson", False, "LiDAR \nR–Error \ncorr."),
        ("r_ultrasonic_pearson", False, "Ultra. \nR–Error \ncorr."),
        ("r_radar_pearson", False, "Radar \nR–Error \ncorr."),]

    if comparison is None or comparison.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        _save_fig(path)
        return

    work = comparison.copy()
    if "rank_overall" in work.columns:
        work = work.sort_values("rank_overall", ascending=True)
    elif "experiment" in work.columns:
        work = work.sort_values("experiment", ascending=True)

    available = [m for m in metric_specs if m[0] in work.columns]
    if not available:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        _save_fig(path)
        return

    labels = [m[2] for m in available]
    experiments = work["experiment"].astype(str).tolist()
    data = pd.DataFrame(index=experiments)

    for col, higher_is_better, _label in available:
        normalized = _normalize_metric(work[col], higher_is_better=higher_is_better)
        data[col] = normalized.to_numpy(dtype=float)

    mat = data.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(max(9, 0.90 * len(work)), max(4.8, 0.42 * len(available) + 3.2)))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index.tolist())
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="right", fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(data.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                txt_color = "white" if v < 0.45 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=txt_color)

    ax.set_title("Cross-Experiment Performance Landscape")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Experiment")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Normalized Score", labelpad=12)

    ax.text(0.5, -0.20,
        ("All metrics are normalized so higher score is better; error metrics are inverted, and more-negative "
        "reliability–error correlation is preferred."),
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9,
        color="0.35")
    _save_fig(path)

def fig_detection_tradeoff(comparison: pd.DataFrame, path: Path) -> None:
    if comparison is None or comparison.empty:
        fig, ax = plt.subplots(figsize=(7.2, 5.5))
        ax.set_axis_off()
        _save_fig(path)
        return

    work = comparison.copy()
    work = work.sort_values(["roc_auc", "f1"], ascending=[False, False])

    x = _metric_series(work, "roc_auc")
    y = _metric_series(work, "f1")
    brier = _metric_series(work, "brier_score")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sc = ax.scatter(x, y,
        s=230, c=brier,
        cmap="viridis_r",
        alpha=0.92,
        edgecolors="0.2",
        linewidths=0.8,
        zorder=3)

    _annotate_points(ax, work, "roc_auc", "f1", fontsize=8, plot_type="detection")

    ax.axvline(0.5, linestyle="--", linewidth=1, color="0.6", zorder=1)
    ax.axvline(np.nanmedian(x), linestyle=":", linewidth=1, color="0.45", zorder=1)
    ax.axhline(np.nanmedian(y), linestyle=":", linewidth=1, color="0.45", zorder=1)

    ax.set_xlabel("ROC-AUC", labelpad=12)
    ax.set_ylabel("F1", labelpad=10)
    ax.set_title("Detection Quality Trade-off")

    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]

    if len(finite_x):
        x_min = max(0.0, float(finite_x.min()) - 0.05)
        x_max = min(1.0, float(finite_x.max()) + 0.03)
        ax.set_xlim(x_min, x_max)

    if len(finite_y):
        y_min = max(0.0, float(finite_y.min()) - 0.05)
        y_max = min(1.0, float(finite_y.max()) + 0.03)
        ax.set_ylim(y_min, y_max)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Brier Score", labelpad=12)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Experiment",
            markerfacecolor="0.7", markeredgecolor="0.2", markersize=8),
        Line2D([0], [0], color="0.6", lw=1, linestyle="--", label="ROC-AUC = 0.5")]
    ax.legend(handles=legend_items, loc="upper left", frameon=True)
    ax.grid(True, alpha=0.25)
    _save_fig(path)

def fig_regression_comparison(comparison: pd.DataFrame, path: Path) -> None:
    if comparison is None or comparison.empty:
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        ax.set_axis_off()
        _save_fig(path)
        return

    work = comparison.copy()
    work = work.sort_values(["mae", "rmse"], ascending=[True, True])

    x = _metric_series(work, "mae")
    y = _metric_series(work, "rmse")
    bias = _metric_series(work, "bias")
    r2 = _metric_series(work, "r2")
    
    r2_norm = _normalize_metric(r2, higher_is_better=True)
    sizes = (220.0 + 420.0 * r2_norm.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sc = ax.scatter(x, y,
        s=sizes, c=bias,
        cmap="coolwarm",
        alpha=0.90,
        edgecolors="0.2",
        linewidths=0.8,
        zorder=3)

    _annotate_points(ax, work, "mae", "rmse", fontsize=8, plot_type="regression")
    finite_x = x[np.isfinite(x)]

    if len(finite_x):
        line_max = max(float(np.nanmax(x)), float(np.nanmax(y)))
        ax.plot([0.0, line_max], [0.0, line_max], linestyle="--", color="0.55", linewidth=1.0, label="RMSE = MAE")
    ax.set_xlabel("MAE (cm)", labelpad=12)
    ax.set_ylabel("RMSE (cm)", labelpad=12)
    ax.set_title("Depth Estimation Comparison")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Bias (cm)", labelpad=12)

    size_legend = [
        Line2D([0], [0], marker="o", color="w", label="Higher R²", markerfacecolor="0.55",
            markeredgecolor="0.2", markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Lower R²", markerfacecolor="0.85",
            markeredgecolor="0.2", markersize=7)]
    reference_handle = Line2D([0], [0], color="0.55", lw=1.0, linestyle="--", label="RMSE = MAE")
    ax.legend(handles=size_legend + [reference_handle], loc="upper left", frameon=True)
    ax.grid(True, alpha=0.25)
    _save_fig(path)

def fig_uncertainty_quality(comparison: pd.DataFrame, path: Path) -> None:
    if comparison is None or comparison.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.set_axis_off()
        _save_fig(path)
        return

    work = comparison.copy()
    if "rank_overall" in work.columns:
        work = work.sort_values("rank_overall", ascending=True)
    elif "experiment" in work.columns:
        work = work.sort_values("experiment", ascending=True)

    experiments = work["experiment"].astype(str).tolist()
    x = np.arange(len(experiments))
    width = 0.24

    ece = _metric_series(work, "ece")
    mce = _metric_series(work, "mce")
    brier = _metric_series(work, "brier_score")
    coverage = _metric_series(work, "interval_coverage_95")
    fig, ax1 = plt.subplots(figsize=(max(9.5, 0.95 * len(experiments)), 5.6))

    bars1 = ax1.bar(x - width, ece, width=width, label="ECE", color="#4C78A8")
    bars2 = ax1.bar(x, mce, width=width, label="MCE", color="#F58518")
    bars3 = ax1.bar(x + width, brier, width=width, label="Brier", color="#E45756")

    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments, rotation=30, ha="right")
    ax1.set_ylabel("Calibration Error", labelpad=12)
    ax1.set_title("Uncertainty Quality Across Experiments")
    ax1.set_ylim(bottom=0.0)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    line = ax2.plot(x, coverage, marker="o", linewidth=2.2, color="#2CA02C", label="95% coverage")[0]
    target = ax2.axhline(0.95, linestyle="--", color="0.35", linewidth=1.2, label="Target coverage")
    ax2.set_ylabel("Coverage", labelpad=12)
    ax2.set_ylim(0.0, 1.15)

    ax1.legend([bars1, bars2, bars3, line, target], ["ECE", "MCE", "Brier", "Coverage", "Target 0.95"],
    ncol=5, fontsize=8, loc="upper left")
    _save_fig(path)

def fig_runtime_efficiency(comparison: pd.DataFrame, path: Path) -> None:
    if comparison is None or comparison.empty:
        fig, ax = plt.subplots(figsize=(7.3, 5.3))
        ax.set_axis_off()
        _save_fig(path)
        return

    work = comparison.copy()
    work = work.sort_values(["throughput_sps", "avg_total_ms"], ascending=[False, True])

    x = _metric_series(work, "avg_total_ms")
    y = _metric_series(work, "throughput_sps")
    f1 = _metric_series(work, "f1")
    rss = _metric_series(work, "peak_process_rss_mb")

    rss_vals = rss.to_numpy(dtype=float)
    if np.isfinite(rss_vals).any() and np.nanmax(rss_vals) > 0:
        sizes = 120 + 780 * (np.nan_to_num(rss_vals, nan=0.0) / np.nanmax(rss_vals))
    else:
        sizes = np.full(len(work), 260.0, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sc = ax.scatter(x, y,
        s=sizes, c=f1,
        cmap="plasma",
        alpha=0.90,
        edgecolors="0.2",
        linewidths=0.8,
        zorder=3)

    _annotate_points(ax, work, "avg_total_ms", "throughput_sps", fontsize=8, plot_type="runtime")

    ax.set_xlabel("Average total latency (ms)", labelpad=12)
    ax.set_ylabel("Throughput (scenarios/s)", labelpad=12)
    ax.set_title("Runtime Efficiency Frontier")
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("F1", labelpad=12)

    ax.text(0.98, 0.98,
        "Marker size ∝ peak RSS",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.30", fc="white", ec="0.85", alpha=0.9))
    _save_fig(path)

def fig_baseline_degradation(relative: pd.DataFrame, path: Path) -> None:
    if relative is None or relative.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_axis_off()
        _save_fig(path)
        return

    work = relative.copy()
    if "rank_overall" in work.columns:
        work = work.sort_values("rank_overall", ascending=True)
    elif "experiment" in work.columns:
        work = work.sort_values("experiment", ascending=True)

    exps = work["experiment"].astype(str).tolist()
    x = np.arange(len(exps))
    width = 0.18

    metrics = [
        ("delta_f1_pct", "ΔF1%"),
        ("delta_mae_pct", "ΔMAE%"),
        ("delta_ece_pct", "ΔECE%"),
        ("delta_avg_total_ms_pct", "ΔLatency%")]

    fig, ax = plt.subplots(figsize=(max(10.5, 0.95 * len(work)), 5.8))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    plotted_any = False
    for i, ((col, label), color) in enumerate(zip(metrics, palette)):
        if col not in work.columns: continue
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (i - 1.5) * width, vals,
            width=width, label=label, color=color, edgecolor="white", linewidth=0.4)
        plotted_any = True

    if not plotted_any:
        ax.set_axis_off()
        _save_fig(path)
        return

    ax.axhline(0.0, linestyle="--", linewidth=1.1, color="0.35")
    ax.set_yscale("symlog", linthresh=10.0, linscale=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=25, ha="right")
    ax.set_ylabel("Degradation relative to baseline (%)", labelpad=12)
    ax.text(0.01, 0.87, "+ve = Worse | -ve = Better",
    transform=ax.transAxes, ha="left", va="top",
    fontsize=9, bbox=dict(boxstyle="round, pad=0.30", fc="white", ec="0.85", alpha=0.9))
    baseline_name = work["baseline_experiment"].iloc[0] if "baseline_experiment" in work.columns else "baseline"
    ax.set_title(f"Baseline-relative Degradation ({baseline_name})")
    ax.legend(ncol=2, fontsize=9, frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    _save_fig(path)

def fig_reliability_validation(comparison: pd.DataFrame, path: Path) -> None:
    if comparison is None or comparison.empty:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.set_axis_off()
        _save_fig(path)
        return

    work = comparison.copy()
    if "rank_overall" in work.columns:
        work = work.sort_values("rank_overall", ascending=True)
    elif "experiment" in work.columns:
        work = work.sort_values("experiment", ascending=True)

    exps = work["experiment"].astype(str).tolist()
    x = np.arange(len(exps))
    width = 0.24

    cols = [
        ("r_lidar_pearson", "LiDAR", "#4C78A8"),
        ("r_ultrasonic_pearson", "Ultrasonic", "#F58518"),
        ("r_radar_pearson", "Radar", "#54A24B")]

    fig, ax = plt.subplots(figsize=(max(9.0, 0.90 * len(work)), 5.4))
    handles = []
    for i, (col, label, color) in enumerate(cols):
        if col not in work.columns: continue
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=color, edgecolor="white", linewidth=0.4)
        handles.append(bars)

    ax.axhline(0.0, linestyle="--", linewidth=1.1, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=30, ha="right")
    ax.set_ylabel("Pearson correlation with absolute error", labelpad=12)
    ax.text(0.985, 0.90, "More Negative = Better Reliability–Error Alignment",
    transform=ax.transAxes, ha="right", va="top",
    fontsize=9, bbox=dict(boxstyle="round, pad=0.35", fc="white", ec="0.85", alpha=0.9))
    ax.set_title("Reliability Validation")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9, frameon=True)
    _save_fig(path)

# ---------------------------------------------------------------------------
# Report Builders
# ---------------------------------------------------------------------------

def build_publication_tables(summary: pd.DataFrame, relative: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    tab_root = output_dir / "paper" / "tables" / "virtual_edge"
    _ensure_dir(tab_root)

    table1 = summary[["experiment", "source_mode", "profile_name", "input_rows", "used_rows",
        "unique_scenarios"]].copy()

    table2 = summary[["experiment", "accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc",
    "roc_auc", "pr_auc", "brier_score"]].copy()

    table3 = summary[["experiment", "used_rows", "mae", "rmse", "median_ae", "max_ae", "bias",
    "r2", "mape_percent"]].copy()

    table3.rename(columns={"used_rows": "n"}, inplace=True)

    table4 = summary[["experiment", "ece", "mce", "interval_coverage_95", "interval_width_95_mean",
    "winkler_score_95", "confidence_error_pearson", "confidence_error_spearman", "r_lidar_pearson",
    "r_ultrasonic_pearson", "r_radar_pearson"]].copy()

    table5 = relative[["experiment", "avg_total_ms", "p95_total_ms", "throughput_sps",
    "peak_process_rss_mb", "mean_process_cpu_percent", "mean_system_cpu_percent", "rank_f1",
    "rank_mae", "rank_ece", "rank_total_ms", "rank_robustness", "rank_overall"]].copy()

    paths = {}
    for name, df in [
        ("experiment_configuration.csv", table1),
        ("detection_comparison.csv", table2),
        ("regression_comparison.csv", table3),
        ("uncertainty_reliability.csv", table4),
        ("runtime_robustness.csv", table5)]:
        for col in df.columns:
            if col != "experiment":
                df[col] = df[col].map(_fmt)

        path = tab_root / name
        _write_csv(df, path)
        paths[name] = path
    return paths

def build_master_report(
    summary: pd.DataFrame,
    relative: pd.DataFrame,
    analysis: Dict[str, Any],
    output_dir: Path,
    experiment_metrics: List[ExperimentMetrics]) -> Dict[str, Path]:

    report_root = output_dir / "paper" / "reports" / "virtual_edge"
    _ensure_dir(report_root)

    payload = {
        "analysis": analysis,
        "experiments": [asdict(m) for m in experiment_metrics],
        "summary_table": summary.replace({np.nan: None}).to_dict(orient="records"),
        "relative_table": relative.replace({np.nan: None}).to_dict(orient="records"),
    }

    json_path = report_root / "evaluation_report.json"
    md_path = report_root / "evaluation_report.md"
    _write_json(payload, json_path)

    overview = summary[[
        "experiment", "source_mode", "profile_name",
        "input_rows", "used_rows", "unique_scenarios"]].copy()

    winners = pd.DataFrame([
        ["Baseline", analysis.get("baseline_experiment", "")],
        ["Best F1", analysis.get("top_f1_experiment", "")],
        ["Lowest MAE", analysis.get("top_mae_experiment", "")],
        ["Best ECE", analysis.get("top_ece_experiment", "")],
        ["Fastest", analysis.get("fastest_experiment", "")],
        ["Best overall", analysis.get("best_ranking_experiment", "")]], 
        columns=["metric", "experiment"])

    detection = summary[[
        "experiment", "accuracy", "precision", "recall", "specificity",
        "f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "brier_score"]].copy()

    regression = summary[[
        "experiment", "used_rows", "mae", "rmse", "median_ae", "max_ae",
        "bias", "r2", "mape_percent"]].copy()

    regression.rename(columns={"used_rows": "n"}, inplace=True)

    uncertainty = summary[[
        "experiment", "ece", "mce", "interval_coverage_95",
        "interval_width_95_mean", "winkler_score_95",
        "confidence_error_pearson", "confidence_error_spearman",
        "r_lidar_pearson", "r_ultrasonic_pearson", "r_radar_pearson"]].copy()

    runtime = relative[[
        "experiment", "avg_total_ms", "p95_total_ms", "throughput_sps", "peak_process_rss_mb",
        "mean_process_cpu_percent", "mean_system_cpu_percent","rank_f1",
        "rank_mae", "rank_ece", "rank_total_ms", "rank_robustness", "rank_overall"]].copy()

    md = []
    md.append("# FloodTwin-HIL Master Evaluation Report\n\n")
    md.append(_section("Analysis Overview", winners))
    md.append(_section("Experiment Coverage", overview))
    md.append(_section("Detection Comparison", detection))
    md.append(_section("Regression Comparison", regression))
    md.append(_section("Uncertainty and Reliability", uncertainty))
    md.append(_section("Runtime and Ranking", runtime))

    with md_path.open("w", encoding="utf-8") as f:
        f.write("".join(md))
    return {"master_json": json_path, "master_md": md_path}

# ---------------------------------------------------------------------------
# CLI / Main Orchestration
# ---------------------------------------------------------------------------

@dataclass
class BundleResult:
    results_root: str
    output_dir: str
    baseline_experiment: str
    experiments_found: int
    experiments_evaluated: int
    figures: List[str]
    tables: List[str]
    master_json: str
    master_md: str

def run(
    results_root: Path,
    output_dir: Path,
    baseline_experiment: Optional[str] = None,
    threshold_cm: float = 10.0,
    hazard_pred_threshold: float = 0.5,
    deduplicate: bool = True,
    experiments: Optional[Sequence[str]] = None) -> BundleResult:

    exp_dirs = discover_experiments(results_root)
    if experiments:
        wanted = {str(x).strip() for x in experiments}
        exp_dirs = [p for p in exp_dirs if p.name in wanted]

    if not exp_dirs:
        raise RuntimeError(f"No experiment folders with virtual_edge_scenarios.csv found under {results_root}")

    metrics_list: List[ExperimentMetrics] = []
    for exp_dir in exp_dirs:
        metrics, _ = evaluate_experiment(
            exp_dir,
            threshold_cm=threshold_cm,
            hazard_pred_threshold=hazard_pred_threshold,
            deduplicate=deduplicate)
        metrics_list.append(metrics)

    summary, relative, analysis = compare_experiments(metrics_list, baseline_experiment=baseline_experiment)
    _validate_comparison_frame(summary)
    table_paths = build_publication_tables(summary, relative, output_dir)

    fig_root = output_dir / "figures" / "virtual_edge"
    _ensure_dir(fig_root)

    fig_paths = {
        "performance_landscape.png": fig_root / "performance_landscape.png",
        "detection_tradeoff.png": fig_root / "detection_tradeoff.png",
        "regression_comparison.png": fig_root / "regression_comparison.png",
        "uncertainty_quality.png": fig_root / "uncertainty_quality.png",
        "baseline_degradation.png": fig_root / "baseline_degradation.png",
        "runtime_efficiency.png": fig_root / "runtime_efficiency.png",
        "reliability_validation.png": fig_root / "reliability_validation.png",
    }

    fig_performance_landscape(relative, fig_paths["performance_landscape.png"])
    fig_detection_tradeoff(summary, fig_paths["detection_tradeoff.png"])
    fig_regression_comparison(summary, fig_paths["regression_comparison.png"])
    fig_uncertainty_quality(relative, fig_paths["uncertainty_quality.png"])
    fig_runtime_efficiency(relative, fig_paths["runtime_efficiency.png"])
    fig_reliability_validation(relative, fig_paths["reliability_validation.png"])
    fig_baseline_degradation(relative.assign(baseline_experiment=analysis["baseline_experiment"]),
    fig_paths["baseline_degradation.png"])

    master_paths = build_master_report(summary, relative.assign(baseline_experiment=analysis["baseline_experiment"]),
    analysis, output_dir, metrics_list)

    figures = [
        str(fig_paths["performance_landscape.png"]),
        str(fig_paths["detection_tradeoff.png"]),
        str(fig_paths["regression_comparison.png"]),
        str(fig_paths["uncertainty_quality.png"]),
        str(fig_paths["baseline_degradation.png"]),
        str(fig_paths["runtime_efficiency.png"]),
        str(fig_paths["reliability_validation.png"])]

    tables = [str(v) for v in table_paths.values()]

    return BundleResult(
        results_root=str(results_root),
        output_dir=str(output_dir),
        baseline_experiment=str(analysis["baseline_experiment"]),
        experiments_found=len(discover_experiments(results_root)),
        experiments_evaluated=len(exp_dirs),
        figures=figures,
        tables=tables,
        master_json=str(master_paths["master_json"]),
        master_md=str(master_paths["master_md"]))

def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="FloodTwin-HIL Virtual Edge Asset Generator")

    p.add_argument("--results-root",
        type=Path, default=Path("results/virtual_edge"),
        help="Root directory containing experiment folders (each with virtual_edge_scenarios.csv)")

    p.add_argument("--output-dir",
        type=Path, default=Path("/app/FloodTwin-HIL"),
        help="Where to write figures, tables, and master reports")

    p.add_argument("--baseline-experiment",
        type=str, default=None,
        help="Optional baseline experiment folder name (defaults to baseline_dataset, then baseline_registry)")

    p.add_argument("--threshold-cm",
        type=float, default=10.0,
        help="Ground-truth hazard threshold used to derive hazard labels when absent")

    p.add_argument("--hazard-pred-threshold",
        type=float, default=0.5,
        help="Probability threshold used to derive predictions when no hard hazard label is present")

    p.add_argument("--no-deduplicate",
        action="store_true",
        help="Disable deduplication by scenario_id")

    p.add_argument("--experiments",
        type=str, nargs="*", default=None,
        help="Optional subset of experiment folder names to evaluate")
    return p

def main() -> None:
    args = build_argparser().parse_args()
    bundle = run(
        results_root=args.results_root,
        output_dir=args.output_dir,
        baseline_experiment=args.baseline_experiment,
        threshold_cm=args.threshold_cm,
        hazard_pred_threshold=args.hazard_pred_threshold,
        deduplicate=not args.no_deduplicate,
        experiments=args.experiments)

    print("=" * 60)
    print(f"{'FloodTwin-HIL Virtual Edge Asset Generator':^60}")
    print("=" * 60)

    print(f"Results Root           : {bundle.results_root}")
    print(f"Output Dir             : {bundle.output_dir}")
    print(f"Baseline Experiment    : {bundle.baseline_experiment}")
    print(f"Experiments Found      : {bundle.experiments_found}")
    print(f"Experiments Evaluated  : {bundle.experiments_evaluated}")

    print("\nPrimary Figures:")
    for fig in bundle.figures: print(f" - {fig}")
    print("\nTables:")
    for tab in bundle.tables: print(f" - {tab}")
    print("\nMaster Reports:")
    print(f" - {bundle.master_json}")
    print(f" - {bundle.master_md}")
    print("=" * 60)

if __name__ == "__main__":
    main()