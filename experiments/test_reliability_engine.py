from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = BASE_DIR / "datasets" / "dataset_v2.csv"
RESULTS_DIR = BASE_DIR / "results" / "reliability_engine"

VALIDATION_FILE = RESULTS_DIR / "reliability_validation.csv"
SUMMARY_FILE = RESULTS_DIR / "validation_summary.csv"

REQUIRED_COLUMNS = [
    "scenario_id",
    "water_depth",
    "ntu",
    "severity_level",
    "scene_complexity",
    "measurement_spread",
    "sensor_agreement",
    "lidar_error",
    "ultrasonic_error",
    "radar_error",
    "R_lidar_base",
    "R_ultrasonic_base",
    "R_radar_base",
    "R_lidar",
    "R_ultrasonic",
    "R_radar",
    "W_lidar_base",
    "W_ultrasonic_base",
    "W_radar_base",
    "W_lidar",
    "W_ultrasonic",
    "W_radar",
    "fusion_entropy_base",
    "fusion_entropy",
    "effective_sensor_count_base",
    "effective_sensor_count",
    "dominance_ratio_base",
    "dominance_ratio",
    "fusion_confidence_base",
    "fusion_confidence",
    "dominant_sensor_base",
    "dominant_sensor",
    "reliability_spread_base",
    "reliability_spread",
    "reliability_entropy_base",
    "reliability_entropy" ]

def _safe_numeric(arr) -> np.ndarray:
    return np.asarray(arr, dtype=float)

def _safe_mean(values) -> float:
    arr = _safe_numeric(values)
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")

def _safe_std(values) -> float:
    arr = _safe_numeric(values)
    finite = np.isfinite(arr)
    return float(arr[finite].std(ddof=1)) if finite.sum() >= 2 else float("nan")

def _safe_min(values) -> float:
    arr = _safe_numeric(values)
    finite = np.isfinite(arr)
    return float(arr[finite].min()) if finite.any() else float("nan")

def _safe_max(values) -> float:
    arr = _safe_numeric(values)
    finite = np.isfinite(arr)
    return float(arr[finite].max()) if finite.any() else float("nan")

def _safe_quantile(values, q: float) -> float:
    arr = _safe_numeric(values)
    finite = np.isfinite(arr)
    return float(np.quantile(arr[finite], q)) if finite.any() else float("nan")

def _safe_corr(x, y) -> float:
    x = _safe_numeric(x)
    y = _safe_numeric(y)
    mask = np.isfinite(x) & np.isfinite(y)    
    if mask.sum() < 2: return float("nan")

    sx = x[mask]
    sy = y[mask]

    if np.std(sx) <= 1e-12 or np.std(sy) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(sx, sy)[0, 1])

def _entropy_from_weights(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    total = weights.sum()
    if total <= 1e-12:
        return float(np.log(3.0))

    weights = weights / total
    weights = np.clip(weights, 1e-12, 1.0)
    return float(-np.sum(weights * np.log(weights)))

def _effective_sensor_count(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    total = weights.sum()
    if total <= 1e-12:
        return 3.0

    weights = weights / total
    denom = float(np.sum(weights ** 2))
    return float(1.0 / denom) if denom > 1e-12 else 3.0

def _dominant_sensor(weights: np.ndarray) -> str:
    sensor_names = ["lidar", "ultrasonic", "radar"]
    idx = int(np.argmax(weights))
    return sensor_names[idx]

def _dominance_ratio(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if weights.sum() <= 1e-12:
        return 1.0

    weights = weights / weights.sum()
    dominant_idx = int(np.argmax(weights))
    dominant_val = float(np.max(weights))
    other_vals = np.delete(weights, dominant_idx)
    return float(dominant_val / (np.mean(other_vals) + 1e-12))

def _dominant_match_rate(a: pd.Series, b: pd.Series) -> float:
    if len(a) == 0:
        return float("nan")
    return float((a.astype(str).values == b.astype(str).values).mean())

def validate_reliability_engine(dataset_path: Path = DATASET_FILE):
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found")

    df = pd.read_csv(dataset_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    
    if missing:
        raise ValueError(f"dataset_v2 is missing required columns: {missing}")

    selected = df.copy()
    selected["lidar_abs_error"] = selected["lidar_error"].abs()
    selected["ultrasonic_abs_error"] = selected["ultrasonic_error"].abs()
    selected["radar_abs_error"] = selected["radar_error"].abs()

    selected["reliability_mean_base"] = selected[["R_lidar_base", "R_ultrasonic_base", "R_radar_base"]].mean(axis=1)
    selected["reliability_mean_final"] = selected[["R_lidar", "R_ultrasonic", "R_radar"]].mean(axis=1)

    selected["reliability_std_base"] = selected[["R_lidar_base", "R_ultrasonic_base", "R_radar_base"]].std(axis=1)
    selected["reliability_std_final"] = selected[["R_lidar", "R_ultrasonic", "R_radar"]].std(axis=1)

    selected["reliability_spread_base"] = (
        selected[["R_lidar_base", "R_ultrasonic_base", "R_radar_base"]].max(axis=1)
        - selected[["R_lidar_base", "R_ultrasonic_base", "R_radar_base"]].min(axis=1)).clip(0.0, 1.0)

    selected["reliability_spread_final"] = (
        selected[["R_lidar", "R_ultrasonic", "R_radar"]].max(axis=1)
        - selected[["R_lidar", "R_ultrasonic", "R_radar"]].min(axis=1)).clip(0.0, 1.0)

    selected["weight_spread_base"] = (
        selected[["W_lidar_base", "W_ultrasonic_base", "W_radar_base"]].max(axis=1)
        - selected[["W_lidar_base", "W_ultrasonic_base", "W_radar_base"]].min(axis=1)).clip(0.0, 1.0)

    selected["weight_spread_final"] = (
        selected[["W_lidar", "W_ultrasonic", "W_radar"]].max(axis=1)
        - selected[["W_lidar", "W_ultrasonic", "W_radar"]].min(axis=1)).clip(0.0, 1.0)

    selected["reliability_entropy_recomputed_base"] = selected[[
        "R_lidar_base", "R_ultrasonic_base", "R_radar_base"
    ]].apply(lambda row: _entropy_from_weights(row.to_numpy(dtype=float)), axis=1)

    selected["reliability_entropy_recomputed_final"] = selected[[
        "R_lidar", "R_ultrasonic", "R_radar"
    ]].apply(lambda row: _entropy_from_weights(row.to_numpy(dtype=float)), axis=1)

    selected["effective_sensor_count_recomputed_base"] = selected[[
        "W_lidar_base", "W_ultrasonic_base", "W_radar_base"
    ]].apply(lambda row: _effective_sensor_count(row.to_numpy(dtype=float)), axis=1)

    selected["effective_sensor_count_recomputed_final"] = selected[[
        "W_lidar", "W_ultrasonic", "W_radar"
    ]].apply(lambda row: _effective_sensor_count(row.to_numpy(dtype=float)), axis=1)

    selected["dominant_sensor_from_weights_base"] = selected[[
        "W_lidar_base", "W_ultrasonic_base", "W_radar_base"
    ]].apply(lambda row: _dominant_sensor(row.to_numpy(dtype=float)), axis=1)

    selected["dominant_sensor_from_weights_final"] = selected[[
        "W_lidar", "W_ultrasonic", "W_radar"
    ]].apply(lambda row: _dominant_sensor(row.to_numpy(dtype=float)), axis=1)

    selected["dominance_ratio_recomputed_base"] = selected[[
        "W_lidar_base", "W_ultrasonic_base", "W_radar_base"
    ]].apply(lambda row: _dominance_ratio(row.to_numpy(dtype=float)), axis=1)

    selected["dominance_ratio_recomputed_final"] = selected[[
        "W_lidar", "W_ultrasonic", "W_radar"
    ]].apply(lambda row: _dominance_ratio(row.to_numpy(dtype=float)), axis=1)

    selected["dominant_sensor_match_base_vs_final"] = (
        selected["dominant_sensor_base"].astype(str).values
        == selected["dominant_sensor"].astype(str).values)
    selected["dominant_sensor_match_base_vs_weights"] = (
        selected["dominant_sensor_base"].astype(str).values
        == selected["dominant_sensor_from_weights_base"].astype(str).values)
    selected["dominant_sensor_match_final_vs_weights"] = (
        selected["dominant_sensor"].astype(str).values
        == selected["dominant_sensor_from_weights_final"].astype(str).values)

    selected["dominant_reliability_sensor"] = selected[[
        "R_lidar", "R_ultrasonic", "R_radar"
    ]].apply(lambda row: _dominant_sensor(row.to_numpy(dtype=float)), axis=1)

    selected["dominant_reliability_match_final"] = (
        selected["dominant_sensor"].astype(str).values
        == selected["dominant_reliability_sensor"].astype(str).values)

    selected.to_csv(VALIDATION_FILE, index=False)

    summary_rows = [
        {"metric": "samples", "value": len(selected)},
        {"metric": "avg_R_lidar_base", "value": _safe_mean(selected["R_lidar_base"])},
        {"metric": "avg_R_ultrasonic_base", "value": _safe_mean(selected["R_ultrasonic_base"])},
        {"metric": "avg_R_radar_base", "value": _safe_mean(selected["R_radar_base"])},
        {"metric": "avg_R_lidar", "value": _safe_mean(selected["R_lidar"])},
        {"metric": "avg_R_ultrasonic", "value": _safe_mean(selected["R_ultrasonic"])},
        {"metric": "avg_R_radar", "value": _safe_mean(selected["R_radar"])},
        {"metric": "std_R_lidar_base", "value": _safe_std(selected["R_lidar_base"])},
        {"metric": "std_R_ultrasonic_base", "value": _safe_std(selected["R_ultrasonic_base"])},
        {"metric": "std_R_radar_base", "value": _safe_std(selected["R_radar_base"])},
        {"metric": "std_R_lidar", "value": _safe_std(selected["R_lidar"])},
        {"metric": "std_R_ultrasonic", "value": _safe_std(selected["R_ultrasonic"])},
        {"metric": "std_R_radar", "value": _safe_std(selected["R_radar"])},
        {"metric": "median_R_lidar", "value": float(selected["R_lidar"].median())},
        {"metric": "median_R_ultrasonic", "value": float(selected["R_ultrasonic"].median())},
        {"metric": "median_R_radar", "value": float(selected["R_radar"].median())},
        {"metric": "min_R_lidar", "value": _safe_min(selected["R_lidar"])},
        {"metric": "min_R_ultrasonic", "value": _safe_min(selected["R_ultrasonic"])},
        {"metric": "min_R_radar", "value": _safe_min(selected["R_radar"])},
        {"metric": "max_R_lidar", "value": _safe_max(selected["R_lidar"])},
        {"metric": "max_R_ultrasonic", "value": _safe_max(selected["R_ultrasonic"])},
        {"metric": "max_R_radar", "value": _safe_max(selected["R_radar"])},
        {"metric": "p05_R_lidar", "value": _safe_quantile(selected["R_lidar"], 0.05)},
        {"metric": "p05_R_ultrasonic", "value": _safe_quantile(selected["R_ultrasonic"], 0.05)},
        {"metric": "p05_R_radar", "value": _safe_quantile(selected["R_radar"], 0.05)},
        {"metric": "p95_R_lidar", "value": _safe_quantile(selected["R_lidar"], 0.95)},
        {"metric": "p95_R_ultrasonic", "value": _safe_quantile(selected["R_ultrasonic"], 0.95)},
        {"metric": "p95_R_radar", "value": _safe_quantile(selected["R_radar"], 0.95)},
        {"metric": "avg_R_lidar_delta", "value": _safe_mean(selected["R_lidar"]) 
        - _safe_mean(selected["R_lidar_base"])},
        {"metric": "avg_R_ultrasonic_delta", "value": _safe_mean(selected["R_ultrasonic"]) 
        - _safe_mean(selected["R_ultrasonic_base"])},
        {"metric": "avg_R_radar_delta", "value": _safe_mean(selected["R_radar"]) 
        - _safe_mean(selected["R_radar_base"])},
        {"metric": "avg_W_lidar_base", "value": _safe_mean(selected["W_lidar_base"])},
        {"metric": "avg_W_ultrasonic_base", "value": _safe_mean(selected["W_ultrasonic_base"])},
        {"metric": "avg_W_radar_base", "value": _safe_mean(selected["W_radar_base"])},
        {"metric": "avg_W_lidar", "value": _safe_mean(selected["W_lidar"])},
        {"metric": "avg_W_ultrasonic", "value": _safe_mean(selected["W_ultrasonic"])},
        {"metric": "avg_W_radar", "value": _safe_mean(selected["W_radar"])},
        {"metric": "std_W_lidar_base", "value": _safe_std(selected["W_lidar_base"])},
        {"metric": "std_W_ultrasonic_base", "value": _safe_std(selected["W_ultrasonic_base"])},
        {"metric": "std_W_radar_base", "value": _safe_std(selected["W_radar_base"])},
        {"metric": "std_W_lidar", "value": _safe_std(selected["W_lidar"])},
        {"metric": "std_W_ultrasonic", "value": _safe_std(selected["W_ultrasonic"])},
        {"metric": "std_W_radar", "value": _safe_std(selected["W_radar"])},
        {"metric": "avg_fusion_entropy_base", "value": _safe_mean(selected["fusion_entropy_base"])},
        {"metric": "avg_fusion_entropy", "value": _safe_mean(selected["fusion_entropy"])},
        {"metric": "avg_effective_sensor_count_base", "value": _safe_mean(selected["effective_sensor_count_base"])},
        {"metric": "avg_effective_sensor_count", "value": _safe_mean(selected["effective_sensor_count"])},
        {"metric": "avg_dominance_ratio_base", "value": _safe_mean(selected["dominance_ratio_base"])},
        {"metric": "avg_dominance_ratio", "value": _safe_mean(selected["dominance_ratio"])},
        {"metric": "avg_fusion_confidence_base", "value": _safe_mean(selected["fusion_confidence_base"])},
        {"metric": "avg_fusion_confidence", "value": _safe_mean(selected["fusion_confidence"])},
        {"metric": "dominant_lidar_count_base", "value": int((selected["dominant_sensor_base"] == "lidar").sum())},
        {"metric": "dominant_ultrasonic_count_base", "value": int((selected["dominant_sensor_base"] 
        == "ultrasonic").sum())},
        {"metric": "dominant_radar_count_base", "value": int((selected["dominant_sensor_base"] == "radar").sum())},
        {"metric": "dominant_lidar_count_final", "value": int((selected["dominant_sensor"] == "lidar").sum())},
        {"metric": "dominant_ultrasonic_count_final", "value": int((selected["dominant_sensor"] == "ultrasonic").sum())},
        {"metric": "dominant_radar_count_final", "value": int((selected["dominant_sensor"] == "radar").sum())},
        {"metric": "dominant_sensor_match_rate_base_vs_final", "value":
         _dominant_match_rate(selected["dominant_sensor_base"], selected["dominant_sensor"])},
        {"metric": "dominant_sensor_match_rate_base_vs_weights", "value":
         _dominant_match_rate(selected["dominant_sensor_base"], selected["dominant_sensor_from_weights_base"])},
        {"metric": "dominant_reliability_match_rate_final", "value": 
        _dominant_match_rate(selected["dominant_sensor"], selected["dominant_reliability_sensor"])},
        {"metric": "avg_scene_complexity", "value": _safe_mean(selected["scene_complexity"])},
        {"metric": "avg_severity_level", "value": _safe_mean(selected["severity_level"])},
        {"metric": "avg_measurement_spread", "value": _safe_mean(selected["measurement_spread"])},
        {"metric": "avg_sensor_agreement", "value": _safe_mean(selected["sensor_agreement"])},
        {"metric": "avg_reliability_mean_base", "value": _safe_mean(selected["reliability_mean_base"])},
        {"metric": "avg_reliability_mean_final", "value": _safe_mean(selected["reliability_mean_final"])},
        {"metric": "avg_reliability_std_base", "value": _safe_mean(selected["reliability_std_base"])},
        {"metric": "avg_reliability_std_final", "value": _safe_mean(selected["reliability_std_final"])},
        {"metric": "avg_reliability_spread_base", "value": _safe_mean(selected["reliability_spread_base"])},
        {"metric": "avg_reliability_spread_final", "value": _safe_mean(selected["reliability_spread_final"])},
        {"metric": "avg_reliability_entropy_base", "value": _safe_mean(selected["reliability_entropy_base"])},
        {"metric": "avg_reliability_entropy_final", "value": _safe_mean(selected["reliability_entropy"])},
        {"metric": "avg_reliability_effective_sensor_count_base", "value": 
        _safe_mean(selected["effective_sensor_count_base"])},
        {"metric": "avg_reliability_effective_sensor_count_final", "value": _safe_mean(selected["effective_sensor_count"])},
        {"metric": "avg_lidar_error", "value": _safe_mean(selected["lidar_error"])},
        {"metric": "avg_ultrasonic_error", "value": _safe_mean(selected["ultrasonic_error"])},
        {"metric": "avg_radar_error", "value": _safe_mean(selected["radar_error"])},
        {"metric": "avg_ultrasonic_abs_error", "value": _safe_mean(selected["ultrasonic_abs_error"])},
        {"metric": "median_ultrasonic_abs_error", "value": float(selected["ultrasonic_abs_error"].median())},
        {"metric": "max_ultrasonic_abs_error", "value": _safe_max(selected["ultrasonic_abs_error"])},
        {"metric": "corr_reliability_lidar_abs_error_base", "value": _safe_corr(selected["R_lidar_base"], 
        selected["lidar_abs_error"])},
        {"metric": "corr_reliability_ultrasonic_abs_error_base", "value": _safe_corr(selected["R_ultrasonic_base"], 
        selected["ultrasonic_abs_error"])},
        {"metric": "corr_reliability_radar_abs_error_base", "value": _safe_corr(selected["R_radar_base"], 
        selected["radar_abs_error"])},
        {"metric": "corr_reliability_lidar_abs_error_final", "value": _safe_corr(selected["R_lidar"], 
        
        selected["lidar_abs_error"])},
        {"metric": "corr_reliability_ultrasonic_abs_error_final", "value": _safe_corr(selected["R_ultrasonic"], 
        selected["ultrasonic_abs_error"])},
        {"metric": "corr_reliability_radar_abs_error_final", "value": _safe_corr(selected["R_radar"], 
        selected["radar_abs_error"])},
        {"metric": "corr_scene_complexity_confidence_base", "value": _safe_corr(selected["scene_complexity"], 
        selected["fusion_confidence_base"])},
        {"metric": "corr_scene_complexity_confidence_final", "value": _safe_corr(selected["scene_complexity"], 
        selected["fusion_confidence"])},
        {"metric": "corr_scene_complexity_entropy_base", "value": _safe_corr(selected["scene_complexity"], 
        selected["fusion_entropy_base"])},
        {"metric": "corr_scene_complexity_entropy_final", "value": _safe_corr(selected["scene_complexity"], 
        selected["fusion_entropy"])},
        {"metric": "corr_reliability_spread_confidence_base", "value": _safe_corr(selected["reliability_spread_base"], 
        selected["fusion_confidence_base"])},
        {"metric": "corr_reliability_spread_confidence_final", "value": _safe_corr(selected["reliability_spread_final"], 
        selected["fusion_confidence"])},
        {"metric": "corr_reliability_spread_entropy_base", "value": _safe_corr(selected["reliability_spread_base"], 
        selected["fusion_entropy_base"])},
        {"metric": "corr_reliability_spread_entropy_final", "value": _safe_corr(selected["reliability_spread_final"], 
        selected["fusion_entropy"])},
        {"metric": "corr_weight_lidar_reliability_base", "value": _safe_corr(selected["W_lidar_base"], 
        selected["R_lidar_base"])},
        {"metric": "corr_weight_ultrasonic_reliability_base", "value": _safe_corr(selected["W_ultrasonic_base"], 
        selected["R_ultrasonic_base"])},
        {"metric": "corr_weight_radar_reliability_base", "value": _safe_corr(selected["W_radar_base"], 
        selected["R_radar_base"])},
        {"metric": "corr_weight_lidar_reliability_final", "value": _safe_corr(selected["W_lidar"], selected["R_lidar"])},
        {"metric": "corr_weight_ultrasonic_reliability_final", "value": _safe_corr(selected["W_ultrasonic"], 
        selected["R_ultrasonic"])},
        {"metric": "corr_weight_radar_reliability_final", "value": _safe_corr(selected["W_radar"], selected["R_radar"])},
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print("=" * 60)
    print("FloodTwin-HIL Reliability Validation")
    print("=" * 60)
    print()
    print(f"Samples : {len(selected)}")
    print(f"Validation Dataset Saved : {VALIDATION_FILE}")
    print(f"Summary Saved : {SUMMARY_FILE}")
    print()
    print(summary_df.head(20).to_string(index=False))
    print()
    print("Dominance Counts (Base)")
    print(selected["dominant_sensor_base"].value_counts())
    print()
    print("Dominance Counts (Final)")
    print(selected["dominant_sensor"].value_counts())
    print()
    print("=" * 60)

    return selected, summary_df

if __name__ == "__main__":
    validate_reliability_engine()