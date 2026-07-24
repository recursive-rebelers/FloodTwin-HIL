import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fusion_engine.bayesian_fusion import BayesianFusionCore
from fusion_engine.fixed_fusion import FixedFusionEngine
from fusion_engine.adaptive_fusion import AdaptiveFusionEngine
from fusion_engine.uncertainty import UncertaintyQuantification
from fusion_engine.hazard_assessment import HazardAssessment

DATASET_FILE_IN = BASE_DIR / "datasets" / "dataset_v2.csv"
RESULTS_DIR = BASE_DIR / "results" / "fusion_engine"
DATASET_FILE_OUT = RESULTS_DIR / "fusion_results.csv"
SUMMARY_FILE = RESULTS_DIR / "fusion_summary.csv"

DEPTH_MIN_CM = 0.0
DEPTH_MAX_CM = 30.0
DEPTH_RESOLUTION = 300

FIXED_SIGMA_BOUNDS = {
    "lidar": (0.85, 3.50),
    "ultrasonic": (0.75, 3.20),
    "radar": (0.60, 2.80),
}

ADAPTIVE_SIGMA_BOUNDS = {
    "lidar": (0.60, 3.40),
    "ultrasonic": (0.60, 3.00),
    "radar": (0.50, 2.60),
}

def _choose_column(df: pd.DataFrame, candidates: list[str], label: str) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None

def _numeric_series(df: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    col = _choose_column(df, candidates, label="/".join(candidates))
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def _safe_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)

def _safe_corr(x, y) -> float:
    x = _safe_array(x)
    y = _safe_array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    sx = x[mask]
    sy = y[mask]
    if np.std(sx) <= 1e-12 or np.std(sy) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(sx, sy)[0, 1])

def _severity_level_from_series(df: pd.DataFrame) -> pd.Series:
    risk_levels = {
        "none": 0,
        "low": 1,
        "moderate": 2,
        "medium": 2,
        "high": 3,
        "severe": 3,
        "extreme": 4,
    }

    if "severity_level" in df.columns:
        return pd.to_numeric(df["severity_level"], errors="coerce").fillna(2).clip(0, 4)

    if "severity" in df.columns:
        severity = df["severity"].astype(str).str.strip().str.lower()
        mapped = severity.map(risk_levels).fillna(2)
        return mapped.astype(float).clip(0, 4)

    return pd.Series(2.0, index=df.index, dtype=float)

def _resolve_measurements(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lidar_col = _choose_column(
        df, ["lidar", "z_lidar", "lidar_measurement"], label="lidar measurement")
    ultra_col = _choose_column(
        df, ["ultrasonic", "z_ultra", "z_ultrasonic", "ultrasonic_measurement"], label="ultrasonic measurement")
    radar_col = _choose_column(
        df, ["radar", "z_radar", "radar_measurement"], label="radar measurement")

    if lidar_col is None or ultra_col is None or radar_col is None:
        raise ValueError("dataset_v2 must contain lidar, ultrasonic, and radar measurement columns")

    z_lidar = pd.to_numeric(df[lidar_col], errors="coerce").to_numpy(dtype=float)
    z_ultra = pd.to_numeric(df[ultra_col], errors="coerce").to_numpy(dtype=float)
    z_radar = pd.to_numeric(df[radar_col], errors="coerce").to_numpy(dtype=float)

    return z_lidar, z_ultra, z_radar

def _robust_row_mad(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=float)
    row_median = np.nanmedian(obs, axis=1)
    row_median = np.nan_to_num(row_median, nan=0.0)
    abs_dev = np.abs(obs - row_median[:, None])
    mad = np.nanmedian(abs_dev, axis=1)
    mad = np.nan_to_num(mad, nan=0.0)
    return 1.4826 * mad

def _compute_measurement_spread(z_lidar: np.ndarray, z_ultra: np.ndarray, z_radar: np.ndarray) -> np.ndarray:
    obs = np.column_stack([z_lidar, z_ultra, z_radar]).astype(float)
    return _robust_row_mad(obs)

def _compute_sensor_agreement(z_lidar: np.ndarray, z_ultra: np.ndarray, z_radar: np.ndarray) -> np.ndarray:
    obs = np.column_stack([z_lidar, z_ultra, z_radar]).astype(float)
    row_median = np.nanmedian(obs, axis=1)
    row_median = np.nan_to_num(row_median, nan=0.0)
    obs = np.where(np.isfinite(obs), obs, row_median[:, None])

    pairwise_mean = (
        np.abs(obs[:, 0] - obs[:, 1]) +
        np.abs(obs[:, 0] - obs[:, 2]) +
        np.abs(obs[:, 1] - obs[:, 2])) / 3.0

    mean_abs = np.mean(np.abs(obs), axis=1) + 1e-6
    agreement = 1.0 / (1.0 + pairwise_mean / (mean_abs + 1e-6))
    return np.clip(agreement, 0.0, 1.0)

def _compute_clutter_proxy(
    df: pd.DataFrame,
    water_depth: pd.Series,
    ntu: pd.Series,
    measurement_spread: np.ndarray) -> pd.Series:

    clutter_col = _choose_column(df, ["clutter_probability", "clutter", "clutter_score"], label="clutter")
    if clutter_col is not None:
        return pd.to_numeric(df[clutter_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    spread_norm = measurement_spread / (np.percentile(measurement_spread, 95) + 1e-6)
    spread_norm = np.clip(spread_norm, 0.0, 1.0)

    water_norm = water_depth.to_numpy(dtype=float)
    if float(np.nanmax(water_norm)) > 3.0:
        water_norm = water_norm / 20.0
    water_norm = np.clip(water_norm, 0.0, 1.0)

    ntu_norm = np.clip(ntu.to_numpy(dtype=float) / 500.0, 0.0, 1.0)

    clutter = 10.0 * (
        0.48 * water_norm +
        0.34 * ntu_norm +
        0.18 * spread_norm)

    return pd.Series(np.clip(clutter, 0.0, 10.0), index=df.index, dtype=float)

def _reference_depth_source_and_values(
    df: pd.DataFrame,
    z_lidar: np.ndarray,
    z_ultra: np.ndarray,
    z_radar: np.ndarray
) -> tuple[pd.Series, str]:

    candidates = [
        "true_depth",
        "ground_truth_depth",
        "reference_depth",
        "pothole_depth",
        "pothole_depth_cm",
        "depth_cm"]

    depth_col = _choose_column(df, candidates, label="reference depth")

    if depth_col is not None:
        ref = pd.to_numeric(df[depth_col], errors="coerce")
        if ref.notna().any():
            ref = ref.ffill().bfill().fillna(0.0).astype(float)
            return ref.clip(DEPTH_MIN_CM, DEPTH_MAX_CM), depth_col

    water_depth = _numeric_series(df, ["water_depth"], default=0.0).fillna(0.0).clip(lower=0.0)
    ntu = _numeric_series(df, ["ntu"], default=0.0).fillna(0.0).clip(lower=0.0)
    severity_level = _severity_level_from_series(df)

    water_scale = 18.0 if float(np.nanmax(water_depth.to_numpy(dtype=float))) <= 3.0 else 1.0
    water_cm = water_depth * water_scale

    # Conservative fallback proxy; intentionally does not use the current measurements
    ref = (0.82 * water_cm + 1.10 * severity_level + 0.03 * np.log1p(ntu))
    return ref.clip(DEPTH_MIN_CM, DEPTH_MAX_CM), "derived_context_proxy"

def _robust_sigma_from_error(
    series: Optional[pd.Series],
    fallback: float,
    lower: float,
    upper: float) -> float:

    if series is None:
        return float(np.clip(fallback, lower, upper))

    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])

    if arr.size == 0:
        return float(np.clip(fallback, lower, upper))

    sigma = float(np.nanmedian(arr) * 1.15 + 0.20)
    return float(np.clip(sigma, lower, upper))

def derive_fixed_sigmas(df: pd.DataFrame) -> tuple[float, float, float]:
    lidar_sigma = _robust_sigma_from_error(
        df["lidar_error"] if "lidar_error" in df.columns else None,
        fallback=1.80,
        lower=FIXED_SIGMA_BOUNDS["lidar"][0],
        upper=FIXED_SIGMA_BOUNDS["lidar"][1])

    ultra_sigma = _robust_sigma_from_error(
        df["ultrasonic_error"] if "ultrasonic_error" in df.columns else None,
        fallback=1.65,
        lower=FIXED_SIGMA_BOUNDS["ultrasonic"][0],
        upper=FIXED_SIGMA_BOUNDS["ultrasonic"][1])

    radar_sigma = _robust_sigma_from_error(
        df["radar_error"] if "radar_error" in df.columns else None,
        fallback=1.35,
        lower=FIXED_SIGMA_BOUNDS["radar"][0],
        upper=FIXED_SIGMA_BOUNDS["radar"][1])
    return float(lidar_sigma), float(ultra_sigma), float(radar_sigma)

def _contextual_sigmas(
    fixed_sigmas: tuple[float, float, float],
    r_lidar: float,
    r_ultra: float,
    r_radar: float,
    scene_complexity: float,
    sensor_agreement: float,
    water_depth: float, ntu: float,
    clutter_proxy: float,
) -> tuple[float, float, float]:

    fixed_lidar, fixed_ultra, fixed_radar = fixed_sigmas

    r_lidar = float(np.clip(r_lidar, 1e-3, 1.0))
    r_ultra = float(np.clip(r_ultra, 1e-3, 1.0))
    r_radar = float(np.clip(r_radar, 1e-3, 1.0))

    scene_complexity = float(np.clip(scene_complexity, 0.0, 1.0))
    sensor_agreement = float(np.clip(sensor_agreement, 0.0, 1.0))

    water_norm = water_depth / 20.0 if water_depth > 3.0 else water_depth
    water_norm = float(np.clip(water_norm, 0.0, 1.0))
    ntu_norm = float(np.clip(ntu / 500.0, 0.0, 1.0))
    clutter_norm = float(np.clip(clutter_proxy / 10.0, 0.0, 1.0))

    lidar_sigma = fixed_lidar * (1.0
        + 0.95 * np.power(1.0 - r_lidar, 1.15)
        + 0.32 * scene_complexity
        + 0.08 * ntu_norm
        + 0.10 * (1.0 - sensor_agreement))

    ultra_sigma = fixed_ultra * (1.0
        + 0.90 * np.power(1.0 - r_ultra, 1.15)
        + 0.27 * scene_complexity
        + 0.12 * water_norm
        + 0.10 * (1.0 - sensor_agreement))

    radar_sigma = fixed_radar * (1.0
        + 0.78 * np.power(1.0 - r_radar, 1.15)
        + 0.20 * scene_complexity
        + 0.16 * clutter_norm
        + 0.05 * water_norm)

    lidar_sigma = float(np.clip(
        lidar_sigma, ADAPTIVE_SIGMA_BOUNDS["lidar"][0], ADAPTIVE_SIGMA_BOUNDS["lidar"][1]))
    ultra_sigma = float(np.clip(
        ultra_sigma, ADAPTIVE_SIGMA_BOUNDS["ultrasonic"][0], ADAPTIVE_SIGMA_BOUNDS["ultrasonic"][1]))
    radar_sigma = float(np.clip(
        radar_sigma, ADAPTIVE_SIGMA_BOUNDS["radar"][0], ADAPTIVE_SIGMA_BOUNDS["radar"][1]))
    return lidar_sigma, ultra_sigma, radar_sigma

def run_fusion_pipeline():
    if not DATASET_FILE_IN.exists():
        raise FileNotFoundError(f"Input dataset not found: {DATASET_FILE_IN}")

    df = pd.read_csv(DATASET_FILE_IN).copy()

    for col in ["scenario_id", "water_depth", "ntu"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset_v2: {col}")

    z_lidar, z_ultra, z_radar = _resolve_measurements(df)
    z_lidar = np.clip(z_lidar, DEPTH_MIN_CM, DEPTH_MAX_CM)
    z_ultra = np.clip(z_ultra, DEPTH_MIN_CM, DEPTH_MAX_CM)
    z_radar = np.clip(z_radar, DEPTH_MIN_CM, DEPTH_MAX_CM)

    water_depth = pd.to_numeric(df["water_depth"], errors="coerce").fillna(0.0).clip(lower=0.0)
    ntu = pd.to_numeric(df["ntu"], errors="coerce").fillna(0.0).clip(lower=0.0)

    measurement_spread = _compute_measurement_spread(z_lidar, z_ultra, z_radar)
    sensor_agreement = _compute_sensor_agreement(z_lidar, z_ultra, z_radar)
    clutter_proxy = _compute_clutter_proxy(df, water_depth, ntu, measurement_spread)

    if "scene_complexity" in df.columns:
        scene_complexity = pd.to_numeric(
            df["scene_complexity"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    else:
        severity_level = _severity_level_from_series(df).to_numpy(dtype=float)
        water_scale = 18.0 if float(np.nanmax(water_depth.to_numpy(dtype=float))) <= 3.0 else 1.0
        water_norm = np.clip((water_depth.to_numpy(dtype=float) * water_scale) / 20.0, 0.0, 1.0) ** 1.30
        ntu_norm = np.clip(ntu.to_numpy(dtype=float) / 500.0, 0.0, 1.0)
        clutter_norm = np.clip(clutter_proxy.to_numpy(dtype=float) / 10.0, 0.0, 1.0) ** 1.30
        spread_norm = np.clip(measurement_spread / (np.percentile(measurement_spread, 95) + 1e-6), 0.0, 1.0)
        severity_norm = np.clip(severity_level / 4.0, 0.0, 1.0)

        scene_complexity = np.clip(
            0.28 * water_norm +
            0.24 * ntu_norm +
            0.18 * clutter_norm +
            0.17 * spread_norm +
            0.13 * severity_norm,
            0.0, 1.0)

    if "R_lidar_base" not in df.columns or "R_ultrasonic_base" not in df.columns or "R_radar_base" not in df.columns:
        raise ValueError("dataset_v2 must contain R_lidar_base, R_ultrasonic_base, and R_radar_base columns")

    r_lidar = pd.to_numeric(df["R_lidar"], errors="coerce").fillna(pd.to_numeric(
        df["R_lidar_base"], errors="coerce")).to_numpy(dtype=float)
    r_ultra = pd.to_numeric(df["R_ultrasonic"], errors="coerce").fillna(pd.to_numeric(
        df["R_ultrasonic_base"], errors="coerce")).to_numpy(dtype=float)
    r_radar = pd.to_numeric(df["R_radar"], errors="coerce").fillna(pd.to_numeric(
        df["R_radar_base"], errors="coerce")).to_numpy(dtype=float)

    r_lidar = np.clip(np.nan_to_num(r_lidar, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    r_ultra = np.clip(np.nan_to_num(r_ultra, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    r_radar = np.clip(np.nan_to_num(r_radar, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

    fixed_sigmas = derive_fixed_sigmas(df)

    core = BayesianFusionCore(
        depth_min=DEPTH_MIN_CM, depth_max=DEPTH_MAX_CM, resolution=DEPTH_RESOLUTION)
    adaptive_engine = AdaptiveFusionEngine(core)
    fixed_engine = FixedFusionEngine(core)
    uq = UncertaintyQuantification()
    hazard_assessor = HazardAssessment(threshold_cm=10.0, decision_boundary=0.80)

    reference_depth, reference_source = _reference_depth_source_and_values(df, z_lidar, z_ultra, z_radar)
    reference_depth = reference_depth.to_numpy(dtype=float)

    dataset = []

    for idx in range(len(df)):
        true_depth = float(reference_depth[idx])

        lidar_meas = float(z_lidar[idx])
        ultra_meas = float(z_ultra[idx])
        radar_meas = float(z_radar[idx])

        rL = float(r_lidar[idx])
        rU = float(r_ultra[idx])
        rR = float(r_radar[idx])

        scene = float(scene_complexity[idx])
        agree = float(sensor_agreement[idx])
        spread = float(measurement_spread[idx])
        clutter_val = float(clutter_proxy.iloc[idx])

        obs_errors = np.array([
            abs(lidar_meas - true_depth),
            abs(ultra_meas - true_depth),
            abs(radar_meas - true_depth)], dtype=float)

        best_idx = int(np.argmin(obs_errors))
        best_name = ["lidar", "ultrasonic", "radar"][best_idx]
        best_est = float([lidar_meas, ultra_meas, radar_meas][best_idx])
        best_err = float(obs_errors[best_idx])

        single_est = lidar_meas
        single_err = abs(single_est - true_depth)

        fixed = fixed_engine.estimate(
            lidar_meas, ultra_meas, radar_meas,
            sigmas=fixed_sigmas)

        fixed_metrics = uq.compute(core.depths, fixed["posterior"], core.delta_x)
        fixed_map_est = float(fixed_metrics["map_depth"])
        fixed_mean_est = float(fixed_metrics["expected_depth"])
        fixed_map_err = abs(fixed_map_est - true_depth)
        fixed_mean_err = abs(fixed_mean_est - true_depth)

        adaptive_sigmas = _contextual_sigmas(
            fixed_sigmas=fixed_sigmas,
            r_lidar=rL, r_ultra=rU, r_radar=rR,
            scene_complexity=scene,
            sensor_agreement=agree,
            water_depth=float(water_depth.iloc[idx]),
            ntu=float(ntu.iloc[idx]),
            clutter_proxy=clutter_val)

        adaptive = adaptive_engine.fuse(
            lidar_meas,
            ultra_meas,
            radar_meas,
            rL, rU, rR,
            sigmas=adaptive_sigmas,
            scene_complexity=scene,
            sensor_agreement=agree,
            measurement_spread=spread)

        adaptive_metrics = uq.compute(core.depths, adaptive["posterior"], core.delta_x)
        adaptive_map_est = float(adaptive_metrics["map_depth"])
        adaptive_mean_est = float(adaptive_metrics["expected_depth"])
        adaptive_map_err = abs(adaptive_map_est - true_depth)
        adaptive_mean_err = abs(adaptive_mean_est - true_depth)

        hazard = hazard_assessor.evaluate(
            core.depths,
            adaptive["posterior"],
            core.delta_x,
            fusion_confidence=float(adaptive_metrics["confidence"]),
            scene_complexity=scene)

        dataset.append({
            "scenario_id": df.iloc[idx]["scenario_id"],
            "water_depth": float(water_depth.iloc[idx]),
            "ntu": float(ntu.iloc[idx]),
            "true_depth": true_depth,
            "true_depth_source": reference_source,

            "single_sensor_est": round(single_est, 3),
            "single_mae": round(single_err, 3),

            "best_single_sensor_est": round(best_est, 3),
            "best_single_sensor_name": best_name,
            "best_single_mae": round(best_err, 3),

            "fixed_fusion_est": round(fixed_map_est, 3),
            "fixed_mean_est": round(fixed_mean_est, 3),
            "fixed_mae": round(fixed_map_err, 3),
            "fixed_mean_mae": round(fixed_mean_err, 3),

            "adaptive_fusion_est": round(adaptive_map_est, 3),
            "adaptive_depth_mean": round(adaptive_mean_est, 3),
            "adaptive_mae": round(adaptive_map_err, 3),
            "adaptive_mean_mae": round(adaptive_mean_err, 3),

            "fixed_confidence": round(float(fixed_metrics["confidence"]), 5),
            "adaptive_confidence": round(float(adaptive_metrics["confidence"]), 5),

            "fixed_entropy": round(float(fixed_metrics["entropy"]), 5),
            "adaptive_entropy": round(float(adaptive_metrics["entropy"]), 5),

            "fixed_variance": round(float(fixed_metrics["variance"]), 5),
            "adaptive_variance": round(float(adaptive_metrics["variance"]), 5),

            "fixed_posterior_peak": round(float(fixed_metrics["posterior_peak"]), 5),
            "adaptive_posterior_peak": round(float(adaptive_metrics["posterior_peak"]), 5),

            "fixed_ci_lower_95": round(float(fixed_metrics["ci_lower"]), 3),
            "fixed_ci_upper_95": round(float(fixed_metrics["ci_upper"]), 3),
            "adaptive_ci_lower_95": round(float(adaptive_metrics["ci_lower"]), 3),
            "adaptive_ci_upper_95": round(float(adaptive_metrics["ci_upper"]), 3),

            "fixed_ci_width_95": round(float(fixed_metrics["ci_upper"] - fixed_metrics["ci_lower"]), 3),
            "adaptive_ci_width_95": round(float(adaptive_metrics["ci_upper"] - adaptive_metrics["ci_lower"]), 3),

            "entropy_score": round(float(adaptive_metrics["entropy_score"]), 5),
            "peak_score": round(float(adaptive_metrics["peak_score"]), 5),
            "interval_score": round(float(adaptive_metrics["interval_score"]), 5),
            "variance_score": round(float(adaptive_metrics["variance_score"]), 5),

            "adaptive_w_lidar": round(float(adaptive["weights"]["lidar"]), 3),
            "adaptive_w_ultra": round(float(adaptive["weights"]["ultrasonic"]), 3),
            "adaptive_w_radar": round(float(adaptive["weights"]["radar"]), 3),

            "adaptive_sigma_lidar": round(float(adaptive["sigmas"]["lidar"]), 3),
            "adaptive_sigma_ultra": round(float(adaptive["sigmas"]["ultrasonic"]), 3),
            "adaptive_sigma_radar": round(float(adaptive["sigmas"]["radar"]), 3),

            "context_difficulty": round(float(adaptive["diagnostics"]["context_difficulty"]), 4),
            "adaptation_gate": round(float(adaptive["diagnostics"]["adaptation_gate"]), 4),
            "effective_sensor_count": round(float(adaptive["diagnostics"]["effective_sensor_count"]), 4),
            "dominance_ratio": round(float(adaptive["diagnostics"]["dominance_ratio"]), 4),
            "weight_entropy": round(float(adaptive["diagnostics"]["weight_entropy"]), 4),
            "measurement_consistency": round(float(adaptive["diagnostics"].get("measurement_consistency_mean", np.nan)), 4),

            "support_fraction": float(adaptive_metrics["support_fraction"]),
            "effective_support": float(adaptive_metrics["effective_support"]),
            "gap_score": float(adaptive_metrics["gap_score"]),

            "hazard_probability": round(float(hazard["hazard_probability"]), 5),
            "hazard_status": hazard["status"],
            "risk_score": round(float(hazard["risk_score"]), 2),
            "hazard_expected_depth": hazard["expected_hazard_depth"],
        })

    df_out = pd.DataFrame(dataset)

    df_out["single_abs_error"] = df_out["single_sensor_est"].sub(df_out["true_depth"]).abs()
    df_out["best_single_abs_error"] = df_out["best_single_sensor_est"].sub(df_out["true_depth"]).abs()
    df_out["fixed_abs_error"] = df_out["fixed_fusion_est"].sub(df_out["true_depth"]).abs()
    df_out["adaptive_abs_error"] = df_out["adaptive_fusion_est"].sub(df_out["true_depth"]).abs()
    df_out["fixed_mean_abs_error"] = df_out["fixed_mean_est"].sub(df_out["true_depth"]).abs()
    df_out["adaptive_mean_abs_error"] = df_out["adaptive_depth_mean"].sub(df_out["true_depth"]).abs()

    df_out["obs_lidar_abs_error"] = np.abs(z_lidar - reference_depth)
    df_out["obs_ultrasonic_abs_error"] = np.abs(z_ultra - reference_depth)
    df_out["obs_radar_abs_error"] = np.abs(z_radar - reference_depth)

    corr_conf_mae = _safe_corr(df_out["adaptive_confidence"], df_out["adaptive_abs_error"])
    corr_entropy_mae = _safe_corr(df_out["adaptive_entropy"], df_out["adaptive_abs_error"])
    corr_ciwidth_mae = _safe_corr(df_out["adaptive_ci_width_95"], df_out["adaptive_abs_error"])
    corr_variance_mae = _safe_corr(df_out["adaptive_variance"], df_out["adaptive_abs_error"])
    corr_hazard_depth = _safe_corr(df_out["hazard_probability"], df_out["true_depth"])
    corr_hazard_mae = _safe_corr(df_out["hazard_probability"], df_out["adaptive_abs_error"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(DATASET_FILE_OUT, index=False)

    rmse_single = float(np.sqrt(np.mean(df_out["single_abs_error"] ** 2)))
    rmse_best_single = float(np.sqrt(np.mean(df_out["best_single_abs_error"] ** 2)))
    rmse_fixed_map = float(np.sqrt(np.mean(df_out["fixed_abs_error"] ** 2)))
    rmse_adaptive_map = float(np.sqrt(np.mean(df_out["adaptive_abs_error"] ** 2)))
    rmse_fixed_mean = float(np.sqrt(np.mean(df_out["fixed_mean_abs_error"] ** 2)))
    rmse_adaptive_mean = float(np.sqrt(np.mean(df_out["adaptive_mean_abs_error"] ** 2)))

    rmse_gain_percent = (
        (rmse_fixed_map - rmse_adaptive_map) / rmse_fixed_map * 100.0
        if rmse_fixed_map > 1e-12 else 0.0)

    hazard_counts = df_out["hazard_status"].value_counts()
    hazard_rate_percent = 100.0 * hazard_counts.get("HAZARD", 0) / len(df_out)
    caution_rate_percent = 100.0 * hazard_counts.get("CAUTION", 0) / len(df_out)
    safe_rate_percent = 100.0 * hazard_counts.get("SAFE", 0) / len(df_out)

    confidence_gain_percent = float((df_out["adaptive_confidence"].mean() - df_out["fixed_confidence"].mean()) * 100.0)
    adaptive_win_rate = float((df_out["adaptive_abs_error"] < df_out["fixed_abs_error"]).mean())
    adaptive_mean_win_rate = float((df_out["adaptive_mean_abs_error"] < df_out["fixed_mean_abs_error"]).mean())
    best_single_counts = df_out["best_single_sensor_name"].value_counts()

    summary = {
        "samples": len(df_out),
        "rmse_single_sensor": rmse_single,
        "rmse_best_single_sensor": rmse_best_single,
        "rmse_fixed_fusion": rmse_fixed_map,
        "rmse_adaptive_fusion": rmse_adaptive_map,
        "rmse_fixed_fusion_mean": rmse_fixed_mean,
        "rmse_adaptive_fusion_mean": rmse_adaptive_mean,
        "rmse_gain_percent": rmse_gain_percent,

        "avg_single_mae": float(df_out["single_abs_error"].mean()),
        "avg_best_single_mae": float(df_out["best_single_abs_error"].mean()),
        "avg_fixed_mae": float(df_out["fixed_abs_error"].mean()),
        "avg_adaptive_mae": float(df_out["adaptive_abs_error"].mean()),
        "avg_fixed_mean_mae": float(df_out["fixed_mean_abs_error"].mean()),
        "avg_adaptive_mean_mae": float(df_out["adaptive_mean_abs_error"].mean()),

        "adaptive_win_rate": adaptive_win_rate,
        "adaptive_mean_win_rate": adaptive_mean_win_rate,

        "avg_fixed_confidence": float(df_out["fixed_confidence"].mean()),
        "avg_confidence": float(df_out["adaptive_confidence"].mean()),
        "std_confidence": float(df_out["adaptive_confidence"].std(ddof=1)),
        "confidence_gain_percent": confidence_gain_percent,

        "avg_fixed_entropy": float(df_out["fixed_entropy"].mean()),
        "avg_entropy": float(df_out["adaptive_entropy"].mean()),
        "std_entropy": float(df_out["adaptive_entropy"].std(ddof=1)),

        "avg_fixed_variance": float(df_out["fixed_variance"].mean()),
        "avg_variance": float(df_out["adaptive_variance"].mean()),

        "avg_fixed_ci_width_95": float(df_out["fixed_ci_width_95"].mean()),
        "avg_ci_width_95": float(df_out["adaptive_ci_width_95"].mean()),
        "avg_fixed_posterior_peak": float(df_out["fixed_posterior_peak"].mean()),
        "avg_posterior_peak": float(df_out["adaptive_posterior_peak"].mean()),

        "avg_adaptive_w_lidar": float(df_out["adaptive_w_lidar"].mean()),
        "avg_adaptive_w_ultra": float(df_out["adaptive_w_ultra"].mean()),
        "avg_adaptive_w_radar": float(df_out["adaptive_w_radar"].mean()),

        "avg_adaptive_sigma_lidar": float(df_out["adaptive_sigma_lidar"].mean()),
        "avg_adaptive_sigma_ultra": float(df_out["adaptive_sigma_ultra"].mean()),
        "avg_adaptive_sigma_radar": float(df_out["adaptive_sigma_radar"].mean()),

        "corr_confidence_mae": corr_conf_mae,
        "corr_entropy_mae": corr_entropy_mae,
        "corr_ciwidth_mae": corr_ciwidth_mae,
        "corr_variance_mae": corr_variance_mae,
        "corr_hazard_depth": corr_hazard_depth,
        "corr_hazard_mae": corr_hazard_mae,

        "avg_hazard_probability": float(df_out["hazard_probability"].mean()),
        "hazards_detected": int(hazard_counts.get("HAZARD", 0)),
        "hazard_rate_percent": hazard_rate_percent,
        "caution_flags": int(hazard_counts.get("CAUTION", 0)),
        "caution_rate_percent": caution_rate_percent,
        "safe_cases": int(hazard_counts.get("SAFE", 0)),
        "safe_rate_percent": safe_rate_percent,

        "mean_entropy_score": float(df_out["entropy_score"].mean()),
        "std_entropy_score": float(df_out["entropy_score"].std(ddof=1)),
        "mean_peak_score": float(df_out["peak_score"].mean()),
        "std_peak_score": float(df_out["peak_score"].std(ddof=1)),
        "mean_interval_score": float(df_out["interval_score"].mean()),
        "std_interval_score": float(df_out["interval_score"].std(ddof=1)),
        "mean_variance_score": float(df_out["variance_score"].mean()),
        "std_variance_score": float(df_out["variance_score"].std(ddof=1)),

        "avg_true_depth": float(df_out["true_depth"].mean()),
        "avg_scene_complexity": float(np.mean(scene_complexity)),
        "avg_sensor_agreement": float(np.mean(sensor_agreement)),
        "avg_measurement_spread": float(np.mean(measurement_spread)),

        "avg_reference_depth": float(np.mean(reference_depth)),
        "reference_depth_source": reference_source,

        "best_single_lidar_count": int(best_single_counts.get("lidar", 0)),
        "best_single_ultrasonic_count": int(best_single_counts.get("ultrasonic", 0)),
        "best_single_radar_count": int(best_single_counts.get("radar", 0)),

        "dominant_hazard_safe_count": int(hazard_counts.get("SAFE", 0)),
        "dominant_hazard_caution_count": int(hazard_counts.get("CAUTION", 0)),
        "dominant_hazard_hazard_count": int(hazard_counts.get("HAZARD", 0)),

        "avg_support_fraction": float(df_out["support_fraction"].mean()),
        "avg_effective_support": float(df_out["effective_support"].mean()),
        "avg_gap_score": float(df_out["gap_score"].mean()),

        "avg_context_difficulty": float(df_out["context_difficulty"].mean()),
        "avg_adaptation_gate": float(df_out["adaptation_gate"].mean()),
        "avg_effective_sensor_count": float(df_out["effective_sensor_count"].mean()),
        "avg_weight_entropy": float(df_out["weight_entropy"].mean()),
        "avg_measurement_consistency": float(df_out["measurement_consistency"].mean()),
    }

    summary_df = pd.DataFrame([summary]).round(6)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print("=" * 60)
    print("FloodTwin-HIL Fusion Validation")
    print("=" * 60)
    print()
    print(f"Samples : {len(df_out)}")
    print(f"Dataset Saved : {DATASET_FILE_OUT}")
    print(f"Summary Saved : {SUMMARY_FILE}")
    print()
    print(summary_df.T.to_string(header=False))
    print()
    print("Hazard State Distribution")
    print(df_out["hazard_status"].value_counts())
    print()
    print("Best Single Sensor Distribution")
    print(df_out["best_single_sensor_name"].value_counts())
    print()
    print("=" * 60)

    return df_out, summary_df

if __name__ == "__main__":
    run_fusion_pipeline()