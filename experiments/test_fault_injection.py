import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fusion_engine.bayesian_fusion import BayesianFusionCore
from fusion_engine.adaptive_fusion import AdaptiveFusionEngine
from fusion_engine.uncertainty import UncertaintyQuantification
from fusion_engine.hazard_assessment import HazardAssessment

from fault_injection.dropout_fault import DropoutInjector
from fault_injection.noise_fault import NoiseInjector
from fault_injection.delay_fault import DelayInjector
from fault_injection.sync_fault import SyncInjector
from fault_injection.fault_effects import FaultEffects

DATASET_FILE_IN = BASE_DIR / "datasets" / "dataset_v2.csv"
RESULTS_DIR = BASE_DIR / "results" / "fault_injection"
ROBUSTNESS_RESULTS_FILE = RESULTS_DIR / "robustness_results.csv"
SUMMARY_FILE = RESULTS_DIR / "robustness_summary.csv"

DEPTH_MIN_CM = 0.0
DEPTH_MAX_CM = 30.0
DEPTH_RESOLUTION = 300
CALIBRATION_FRACTION = 0.20
CALIBRATION_RANDOM_SEED = 42

ADAPTIVE_CALIBRATION = {
    "gamma": 1.75,
    "weight_blend": 0.60,
    "sigma_scale": 1.00,
    "sigma_offset": 0.08,
    "sigma_min": 0.45,
    "sigma_max": 2.85,
    "context_blend_boost": 0.28,
    "context_sigma_boost": 0.45,
    "weight_floor": 0.01,
    "sensor_prior": { "lidar": 1.00, "ultrasonic": 1.00, "radar": 1.00 },
    "quality_threshold": 0.24,
}

FIXED_SIGMA_BOUNDS = {
    "lidar": (0.85, 3.50),
    "ultrasonic": (0.75, 3.20),
    "radar": (0.60, 2.80),
}

REQUIRED_CANDIDATES = {
    "scenario_id": ["scenario_id"],
    "true_depth": ["true_depth"],
    "lidar": ["z_lidar", "lidar"],
    "ultra": ["z_ultra", "z_ultrasonic", "ultrasonic"],
    "radar": ["z_radar", "radar"],
    "r_lidar": ["R_lidar"],
    "r_ultra": ["R_ultrasonic", "R_ultra"],
    "r_radar": ["R_radar"],
    "water_depth": ["water_depth"],
    "ntu": ["ntu"],
}

def _resolve_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required column for '{label}'. Expected one of: {list(candidates)}")

def _safe_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)

def _safe_mean(values: Iterable[Any]) -> float:
    arr = _safe_array(values).astype(float, copy=False)
    mask = np.isfinite(arr)
    if not mask.any():
        return float("nan")
    return float(arr[mask].mean())

def _safe_rmse(values: Iterable[Any]) -> float:
    arr = _safe_array(values).astype(float, copy=False)
    mask = np.isfinite(arr)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(arr[mask] ** 2)))

def _safe_rate(flags: Iterable[Any]) -> float:
    arr = np.asarray(flags, dtype=bool)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr) * 100.0)

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

def _resolve_measurements(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lidar_col = _resolve_column(df, REQUIRED_CANDIDATES["lidar"], "lidar measurement")
    ultra_col = _resolve_column(df, REQUIRED_CANDIDATES["ultra"], "ultrasonic measurement")
    radar_col = _resolve_column(df, REQUIRED_CANDIDATES["radar"], "radar measurement")

    z_lidar = pd.to_numeric(df[lidar_col], errors="coerce").to_numpy(dtype=float)
    z_ultra = pd.to_numeric(df[ultra_col], errors="coerce").to_numpy(dtype=float)
    z_radar = pd.to_numeric(df[radar_col], errors="coerce").to_numpy(dtype=float)
    return z_lidar, z_ultra, z_radar

def _compute_measurement_spread(z_lidar: np.ndarray, z_ultra: np.ndarray, z_radar: np.ndarray) -> np.ndarray:
    obs = np.column_stack([z_lidar, z_ultra, z_radar]).astype(float)
    spread = np.zeros(obs.shape[0], dtype=float)

    for i in range(obs.shape[0]):
        valid = obs[i, np.isfinite(obs[i])]
        if valid.size < 2:
            spread[i] = 0.0
            continue
        if valid.size == 2:
            spread[i] = 0.5 * abs(valid[0] - valid[1])
            continue

        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        spread[i] = 1.4826 * mad
    return np.nan_to_num(spread, nan=0.0, posinf=0.0, neginf=0.0)

def _compute_sensor_agreement(z_lidar: np.ndarray, z_ultra: np.ndarray, z_radar: np.ndarray) -> np.ndarray:
    obs = np.column_stack([z_lidar, z_ultra, z_radar]).astype(float)
    agreement = np.ones(obs.shape[0], dtype=float)

    for i in range(obs.shape[0]):
        valid = obs[i, np.isfinite(obs[i])]
        if valid.size < 2:
            agreement[i] = 1.0
            continue

        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        scale = max(1.4826 * mad, 0.35)
        pairwise_differences = []

        for a in range(valid.size):
            for b in range(a + 1, valid.size):
                pairwise_differences.append(abs(valid[a] - valid[b]))

        pairwise_mean = float(np.mean(pairwise_differences))
        agreement[i] = np.exp(-0.25 * np.square(pairwise_mean / scale))
    return np.clip(np.nan_to_num(agreement, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

def _compute_clutter_proxy(df: pd.DataFrame, water_depth: pd.Series, ntu: pd.Series,
    measurement_spread: np.ndarray) -> pd.Series:
    
    water = pd.to_numeric(water_depth, errors="coerce").fillna(0.0).clip(lower=0.0)
    turbidity = pd.to_numeric(ntu, errors="coerce").fillna(0.0).clip(lower=0.0)
    spread = np.asarray(measurement_spread, dtype=float)
    spread = np.nan_to_num(spread, nan=0.0, posinf=0.0, neginf=0.0)

    water_norm = np.clip(water.to_numpy(dtype=float) / 20.0, 0.0, 1.0)
    ntu_norm = np.clip(turbidity.to_numpy(dtype=float) / 500.0, 0.0, 1.0)

    # Sensor disagreement influence
    spread_scale = np.percentile(spread, 95)
    if spread_scale <= 1e-6: spread_scale = 1.0

    spread_norm = np.clip(spread / spread_scale, 0.0, 1.0)
    interaction = water_norm * ntu_norm

    clutter = (
        0.30 * water_norm +
        0.35 * ntu_norm +
        0.20 * spread_norm +
        0.15 * interaction)

    clutter = np.clip(clutter, 0.0, 1.0)
    return pd.Series(clutter, index=df.index, name="clutter_proxy")

def _compute_reliability_context(r_lidar: np.ndarray, r_ultra: np.ndarray, r_radar: np.ndarray, z_lidar: np.ndarray, z_ultra:
    np.ndarray, z_radar: np.ndarray) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    reliabilities = np.column_stack([r_lidar, r_ultra, r_radar]).astype(float)
    measurements = np.column_stack([z_lidar, z_ultra, z_radar]).astype(float)
    reliabilities = np.nan_to_num(reliabilities, nan=0.0, posinf=1.0, neginf=0.0)
    reliabilities = np.clip(reliabilities, 0.0, 1.0)

    active_mask = (np.isfinite(measurements) & (reliabilities > 0.0))
    active_count = np.sum(active_mask, axis=1,)
    active_reliability = np.where(active_mask, reliabilities, 0.0)

    active_max = np.max(np.where(active_mask, active_reliability, -np.inf), axis=1)
    active_min = np.min(np.where(active_mask, active_reliability, np.inf), axis=1)
    reliability_spread = np.where(active_count >= 2, active_max - active_min, 0.0)

    reliability_sum = np.sum(active_reliability, axis=1, keepdims=True,)
    weights = np.divide(active_reliability, reliability_sum, out=np.zeros_like(active_reliability), where=reliability_sum > 1e-12)

    weight_square_sum = np.sum(np.square(weights), axis=1)
    effective_sensor_count = np.divide(1.0, weight_square_sum, out=np.zeros_like(weight_square_sum),
    where=weight_square_sum > 1e-12)

    dominant_weight = np.max(weights, axis=1)
    remaining_weight = np.maximum(np.sum(weights, axis=1) - dominant_weight, 0.0)

    other_sensor_count = np.maximum(active_count - 1, 1)
    other_mean_weight = np.divide(remaining_weight, other_sensor_count, out=np.zeros_like(remaining_weight),
    where=other_sensor_count > 0)
    dominance_ratio = np.where(active_count >= 2, dominant_weight / (other_mean_weight + 1e-12), 1.0)

    safe_weights = np.clip(weights, 1e-12, 1.0)
    entropy = -np.sum(np.where(weights > 0.0, weights * np.log(safe_weights), 0.0), axis=1)
    entropy_denominator = np.log(np.maximum(active_count, 2))
    normalized_entropy = np.divide(entropy, entropy_denominator, out=np.zeros_like(entropy), where=active_count >= 2)
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

    reliability_total = np.sum(active_reliability, axis=1)
    mean_reliability = np.divide(reliability_total, active_count, out=np.zeros_like(reliability_total), where=active_count > 0)
    confidence = np.clip(mean_reliability * (0.60 + 0.40 * (1.0 - normalized_entropy)), 0.0, 1.0)

    return (
        np.nan_to_num(reliability_spread, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(effective_sensor_count, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(dominance_ratio, nan=1.0, posinf=1.0, neginf=1.0),
        np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0))

def _build_context_bundle(
    df: pd.DataFrame,
    z_lidar: np.ndarray,
    z_ultra: np.ndarray,
    z_radar: np.ndarray,
    r_lidar: np.ndarray,
    r_ultra: np.ndarray,
    r_radar: np.ndarray,
    water_depth: pd.Series,
    ntu: pd.Series,
) -> Dict[str, np.ndarray]:

    measurement_spread = _compute_measurement_spread(z_lidar, z_ultra, z_radar)
    sensor_agreement = _compute_sensor_agreement(z_lidar, z_ultra, z_radar)
    clutter_proxy = _compute_clutter_proxy(df, water_depth, ntu, measurement_spread)
    scene_complexity = _compute_scene_complexity(df, water_depth, ntu, clutter_proxy, measurement_spread)
    reliability_spread, effective_sensor_count, dominance_ratio, confidence_proxy = _compute_reliability_context(
        r_lidar, r_ultra, r_radar, z_lidar, z_ultra, z_radar)

    return {
        "scene_complexity": scene_complexity,
        "sensor_agreement": sensor_agreement,
        "reliability_spread": reliability_spread,
        "effective_sensor_count": effective_sensor_count,
        "dominance_ratio": dominance_ratio,
        "confidence_proxy": confidence_proxy,
        "measurement_spread": measurement_spread,
        "water_depth": water_depth.to_numpy(dtype=float),
        "ntu": ntu.to_numpy(dtype=float),
        "clutter_proxy": clutter_proxy.to_numpy(dtype=float),
    }

def _compute_scene_complexity(
    df: pd.DataFrame,
    water_depth: pd.Series,
    ntu: pd.Series,
    clutter_proxy: pd.Series,
    measurement_spread: np.ndarray,
) -> np.ndarray:

    if "scene_complexity" in df.columns:
        return pd.to_numeric(df["scene_complexity"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)

    severity_level = _severity_level_from_series(df).to_numpy(dtype=float)
    water_scale = 18.0 if float(np.nanmax(water_depth.to_numpy(dtype=float))) <= 3.0 else 1.0

    water_norm = np.sqrt(np.clip((water_depth.to_numpy(dtype=float) * water_scale) / 20.0, 0.0, 1.0))
    ntu_norm = np.sqrt(np.clip(ntu.to_numpy(dtype=float) / 500.0, 0.0, 1.0))
    clutter_norm = np.sqrt(np.clip(clutter_proxy.to_numpy(dtype=float), 0.0, 1.0))
    spread_norm = np.sqrt(np.clip(measurement_spread / (np.percentile(measurement_spread, 95) + 1e-6), 0.0, 1.0))
    severity_norm = np.clip(severity_level / 4.0, 0.0, 1.0)

    # Softer and partially decorrelated composition to avoid repeatedly penalizing the same physical effect
    scene_complexity = (
        0.26 * water_norm
        + 0.24 * ntu_norm
        + 0.18 * clutter_norm
        + 0.20 * spread_norm
        + 0.12 * severity_norm)

    return np.clip(scene_complexity, 0.0, 1.0)

def _robust_sigma_from_residuals(
    measurements: np.ndarray, truth: np.ndarray,
    lower: float, upper: float, fallback: float) -> float:

    residuals = np.abs(np.asarray(measurements, dtype=float) - np.asarray(truth, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return float(np.clip(fallback, lower, upper))

    sigma = float(1.4826 * np.nanmedian(residuals) + 0.20)
    return float(np.clip(sigma, lower, upper))

def derive_fixed_sigmas(
    df: pd.DataFrame,
    true_depth: np.ndarray,
    z_lidar: np.ndarray,
    z_ultra: np.ndarray,
    z_radar: np.ndarray,
) -> Tuple[float, float, float]:

    lidar_sigma = _robust_sigma_from_residuals(z_lidar, true_depth,
        lower=FIXED_SIGMA_BOUNDS["lidar"][0],
        upper=FIXED_SIGMA_BOUNDS["lidar"][1],
        fallback=1.80)

    ultra_sigma = _robust_sigma_from_residuals(z_ultra, true_depth,
        lower=FIXED_SIGMA_BOUNDS["ultrasonic"][0],
        upper=FIXED_SIGMA_BOUNDS["ultrasonic"][1],
        fallback=1.65)

    radar_sigma = _robust_sigma_from_residuals(z_radar, true_depth,
        lower=FIXED_SIGMA_BOUNDS["radar"][0],
        upper=FIXED_SIGMA_BOUNDS["radar"][1],
        fallback=1.35)

    return float(lidar_sigma), float(ultra_sigma), float(radar_sigma)

def _sensor_sigmas(
    r_lidar: np.ndarray,
    r_ultra: np.ndarray,
    r_radar: np.ndarray,*,
    scene_complexity: Optional[np.ndarray] = None,
    sensor_agreement: Optional[np.ndarray] = None,
    measurement_spread: Optional[np.ndarray] = None,
    water_depth: Optional[np.ndarray] = None,
    ntu: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    r_lidar = np.clip(np.asarray(r_lidar, dtype=float), 1e-6, 1.0)
    r_ultra = np.clip(np.asarray(r_ultra, dtype=float), 1e-6, 1.0)
    r_radar = np.clip(np.asarray(r_radar, dtype=float), 1e-6, 1.0)

    if scene_complexity is None:
        scene_complexity = np.zeros_like(r_lidar)
    if sensor_agreement is None:
        sensor_agreement = np.ones_like(r_lidar)
    if measurement_spread is None:
        measurement_spread = np.zeros_like(r_lidar)
    if water_depth is None:
        water_depth = np.zeros_like(r_lidar)
    if ntu is None:
        ntu = np.zeros_like(r_lidar)

    scene_complexity = np.clip(np.asarray(scene_complexity, dtype=float), 0.0, 1.0)
    sensor_agreement = np.clip(np.asarray(sensor_agreement, dtype=float), 0.0, 1.0)
    measurement_spread = np.clip(np.asarray(measurement_spread, dtype=float), 0.0, None)
    water_norm = np.clip(np.asarray(water_depth, dtype=float) / 20.0, 0.0, 1.0)
    ntu_norm = np.clip(np.asarray(ntu, dtype=float) / 500.0, 0.0, 1.0)
    spread_norm = np.clip(measurement_spread / (np.percentile(measurement_spread, 95) + 1e-6), 0.0, 1.0)
    env_mix = 0.5 * water_norm + 0.5 * ntu_norm

    def _sigma(base: float, span: float, r: np.ndarray, scene_w: float, env_w: float,
        agreement_w: float, spread_w: float) -> np.ndarray:
        reliability_term = span * (1.0 - np.sqrt(r))

        context_term = (
            scene_w * scene_complexity
            + env_w * env_mix
            + agreement_w * (1.0 - sensor_agreement)
            + spread_w * spread_norm)

        sigma = base + reliability_term + context_term
        return np.clip(sigma, 0.35, 3.50)

    sigma_lidar = _sigma(0.78, 1.70, r_lidar, scene_w=0.20, env_w=0.10, agreement_w=0.08, spread_w=0.05)
    sigma_ultra = _sigma(0.72, 1.55, r_ultra, scene_w=0.18, env_w=0.10, agreement_w=0.07, spread_w=0.05)
    sigma_radar = _sigma(0.58, 1.20, r_radar, scene_w=0.14, env_w=0.08, agreement_w=0.05, spread_w=0.04)
    return sigma_lidar, sigma_ultra, sigma_radar

def _apply_sensor_fault_plan(
    sensor_name: str,
    measurements: np.ndarray,
    reliability: np.ndarray,
    sigma: np.ndarray,
    fault_plan: Sequence[Dict[str, Any]],
    injectors: Dict[str, Any],
    fault_effects: FaultEffects,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    faulted_measurements = np.asarray(measurements, dtype=float).copy()
    faulted_reliability = np.asarray(reliability, dtype=float).copy()
    faulted_sigma = np.asarray(sigma, dtype=float).copy()

    if not fault_plan:
        return faulted_measurements, faulted_reliability, faulted_sigma

    non_dropout_faults = [
        f for f in fault_plan
        if str(f.get("fault_type", "")).lower() != "dropout"]

    dropout_faults = [
        f for f in fault_plan
        if str(f.get("fault_type", "")).lower() == "dropout"]

    if non_dropout_faults:
        effect = fault_effects.apply_fault_sequence(
            sensor=sensor_name,
            fault_sequence=non_dropout_faults,
            reliability=1.0, sigma=1.0)

        faulted_reliability *= float(effect["reliability"])
        faulted_sigma *= float(effect["sigma"])

        for fault in non_dropout_faults:
            fault_type = str(fault.get("fault_type", "")).lower().strip()
            severity = str(fault.get("severity", "low")).lower().strip()

            if fault_type == "noise":
                faulted_measurements = injectors["noise"].inject(faulted_measurements, severity=severity, clip_min=0.0)

            elif fault_type == "delay":
                faulted_measurements = injectors["delay"].inject(faulted_measurements, severity=severity)

            elif fault_type == "sync":
                direction = str(fault.get("direction", "backward"))
                faulted_measurements = injectors["sync"].inject(faulted_measurements, severity=severity, direction=direction)

            elif fault_type == "drift": 
                pass # reserved for future temporal drift tests 
            else: raise ValueError(f"Unsupported fault_type '{fault_type}' for sensor '{sensor_name}'")

    for fault in dropout_faults:
        severity = str(fault.get("severity", "low")).lower().strip()
        faulted_measurements, dropout_mask = (
            injectors["dropout"].inject(faulted_measurements, severity=severity, return_mask=True))
        if np.any(dropout_mask):
            faulted_reliability[dropout_mask] = 0.0

    valid_measurement_mask = np.isfinite(faulted_measurements)
    faulted_reliability = np.where(valid_measurement_mask, np.clip(faulted_reliability, 0.0, 1.0), 0.0)
    faulted_sigma = np.clip(faulted_sigma, 1e-6, None)
    return (faulted_measurements, faulted_reliability, faulted_sigma)

def _fixed_fusion_estimate(
    measurements: Sequence[float],
    fixed_sigmas: Sequence[float],
    core: BayesianFusionCore,
    uq: UncertaintyQuantification,
    hazard_assessor: HazardAssessment,
    scene_complexity: float,
) -> Optional[Dict[str, Any]]:

    z = np.asarray(measurements, dtype=float)
    sigmas = np.asarray(fixed_sigmas, dtype=float)
    valid_mask = np.isfinite(z)

    if not valid_mask.any():
        return None

    # Equal-weight fixed pooling over the available channels, missing channels are excluded
    likelihoods = []
    weights = []
    for value, sigma, valid in zip(z, sigmas, valid_mask):
        if not valid: continue
        likelihoods.append(core.compute_likelihood(float(value), float(max(sigma, 1e-6))))
        weights.append(1.0)

    pooled_likelihood = core.combine_likelihoods(likelihoods, weights)
    posterior = core.compute_posterior(pooled_likelihood)
    metrics = uq.compute(core.depths, posterior, core.delta_x)
    hazard = hazard_assessor.evaluate(core.depths, posterior, core.delta_x,
        fusion_confidence=float(metrics["confidence"]),
        scene_complexity=float(scene_complexity))

    return {
        "posterior": posterior,
        "metrics": metrics,
        "hazard": hazard,
    }

def _adaptive_fusion_estimate(
    measurements: Sequence[float],
    reliabilities: Sequence[float],
    faulted_sigmas: Sequence[float],
    core: BayesianFusionCore,
    adaptive_engine: AdaptiveFusionEngine,
    uq: UncertaintyQuantification,
    hazard_assessor: HazardAssessment,
    scene_complexity: float,
    sensor_agreement: float,
    reliability_spread: float,
    effective_sensor_count: float,
    dominance_ratio: float,
    confidence_proxy: float,
    measurement_spread: float,
    water_depth: float, ntu: float,
) -> Optional[Dict[str, Any]]:

    z = np.asarray(measurements, dtype=float)
    r = np.asarray(reliabilities, dtype=float)
    sigmas = np.asarray(faulted_sigmas, dtype=float)

    active_mask = np.isfinite(z) & (r > 0.0)
    if not active_mask.any():
        return None

    filled_measurements = np.where(active_mask, z, np.nan)
    filled_reliabilities = np.where(active_mask, r, 0.0)

    adaptive_result = adaptive_engine.fuse(
        float(filled_measurements[0]),
        float(filled_measurements[1]),
        float(filled_measurements[2]),
        float(filled_reliabilities[0]),
        float(filled_reliabilities[1]),
        float(filled_reliabilities[2]),
        sigmas=tuple(float(s) for s in sigmas),
        scene_complexity=float(scene_complexity),
        sensor_agreement=float(sensor_agreement),
        reliability_spread=float(reliability_spread),
        effective_sensor_count=float(effective_sensor_count),
        dominance_ratio=float(dominance_ratio),
        confidence=float(confidence_proxy),
        measurement_spread=float(measurement_spread),
        water_depth=float(water_depth),
        ntu=float(ntu),
        active_mask=tuple(bool(v) for v in active_mask))

    posterior = adaptive_result["posterior"]
    metrics = uq.compute(core.depths, posterior, core.delta_x)
    hazard = hazard_assessor.evaluate(
        core.depths, posterior, core.delta_x,
        fusion_confidence=float(metrics["confidence"]),
        scene_complexity=float(scene_complexity))

    return {
        "posterior": posterior,
        "metrics": metrics,
        "hazard": hazard,
        "weights": adaptive_result["weights"],
        "sigmas": adaptive_result["sigmas"],
        "diagnostics": adaptive_result.get("diagnostics", {}),
        "active_sensor_count": adaptive_result["diagnostics"]["active_sensor_count"],
    }

def _evaluate_stream_set(
    exp_name: str,
    base_df: pd.DataFrame,
    streams: Dict[str, np.ndarray],
    reliabilities: Dict[str, np.ndarray],
    fixed_sigmas: Tuple[float, float, float],
    faulted_sigmas: Dict[str, np.ndarray],
    context: Dict[str, np.ndarray],
    true_depth: np.ndarray,
    core: BayesianFusionCore,
    adaptive_engine: AdaptiveFusionEngine,
    uq: UncertaintyQuantification,
    hazard_assessor: HazardAssessment,
) -> Dict[str, Any]:

    fixed_errors: list = []
    adaptive_errors: list = []
    single_errors: list = []
    best_single_errors: list = []

    fixed_failures: list = []
    adaptive_failures: list = []
    single_failures: list = []
    best_single_failures: list = []

    fixed_confidences: list = []
    adaptive_confidences: list = []
    fixed_entropies: list = []
    adaptive_entropies: list = []
    fixed_variances: list = []
    adaptive_variances: list = []
    fixed_hazard_probs: list = []
    adaptive_hazard_probs: list = []
    fixed_ci_widths: list = []
    adaptive_ci_widths: list = []

    adaptive_weights_l: list = []
    adaptive_weights_u: list = []
    adaptive_weights_r: list = []
    adaptive_sigmas_l: list = []
    adaptive_sigmas_u: list = []
    adaptive_sigmas_r: list = []

    adaptive_hazard_statuses: list = []
    fixed_hazard_statuses: list = []

    context_difficulties: list = []
    adaptation_gates: list = []
    measurement_consistencies: list = []
    active_sensor_counts: list = []
    weight_entropies: list = []

    for i in range(len(base_df)):
        zl = float(streams["lidar"][i])
        zu = float(streams["ultra"][i])
        zr = float(streams["radar"][i])
        truth = float(true_depth[i])

        sample_measurements = np.asarray([zl, zu, zr], dtype=float)
        sample_reliabilities = np.asarray([
            reliabilities["lidar"][i],
            reliabilities["ultrasonic"][i],
            reliabilities["radar"][i],
        ], dtype=float)

        sample_context = {
            "scene_complexity": context["scene_complexity"][i],
            "sensor_agreement": context["sensor_agreement"][i],
            "reliability_spread": context["reliability_spread"][i],
            "effective_sensor_count": context["effective_sensor_count"][i],
            "dominance_ratio": context["dominance_ratio"][i],
            "confidence_proxy": context["confidence_proxy"][i],
            "measurement_spread": context["measurement_spread"][i],
            "water_depth": context["water_depth"][i],
            "ntu": context["ntu"][i],
        }

        scene = float(sample_context["scene_complexity"])
        agree = float(sample_context["sensor_agreement"])
        rel_spread = float(sample_context["reliability_spread"])
        eff_count = float(sample_context["effective_sensor_count"])
        dom_ratio = float(sample_context["dominance_ratio"])
        conf_proxy = float(sample_context["confidence_proxy"])
        spread = float(sample_context["measurement_spread"])
        water_d = float(sample_context["water_depth"])
        ntu_val = float(sample_context["ntu"])

        active_count = int(np.sum(np.isfinite(sample_measurements) & (sample_reliabilities > 0.0)))
        active_sensor_counts.append(active_count)

        # Single sensor baseline (LiDAR only) for continuity with earlier stages
        if np.isfinite(zl):
            single_errors.append(abs(zl - truth))
            single_failures.append(False)
        else:
            single_errors.append(np.nan)
            single_failures.append(True)

        # Best single sensor benchmark among valid channels
        obs_errors = np.array([
            abs(zl - truth) if np.isfinite(zl) else np.nan,
            abs(zu - truth) if np.isfinite(zu) else np.nan,
            abs(zr - truth) if np.isfinite(zr) else np.nan,
        ], dtype=float)

        if np.isfinite(obs_errors).any():
            best_idx = int(np.nanargmin(obs_errors))
            best_name = ["lidar", "ultrasonic", "radar"][best_idx]
            best_single_meas = float(sample_measurements[best_idx])
            best_single_err = float(obs_errors[best_idx])
            best_single_errors.append(best_single_err)
            best_single_failures.append(False)
        else:
            best_name = "FAIL"
            best_single_meas = np.nan
            best_single_errors.append(np.nan)
            best_single_failures.append(True)

        fixed = _fixed_fusion_estimate(
            measurements=sample_measurements,
            fixed_sigmas=fixed_sigmas,
            core=core,
            uq=uq,
            hazard_assessor=hazard_assessor,
            scene_complexity=scene)

        if fixed is None:
            fixed_errors.append(np.nan)
            fixed_failures.append(True)
            fixed_confidences.append(np.nan)
            fixed_entropies.append(np.nan)
            fixed_variances.append(np.nan)
            fixed_ci_widths.append(np.nan)
            fixed_hazard_probs.append(np.nan)
            fixed_hazard_statuses.append("FAIL")
        else:
            fixed_uq = fixed["metrics"]
            fixed_est = float(fixed_uq["map_depth"])
            fixed_errors.append(abs(fixed_est - truth))
            fixed_failures.append(False)
            fixed_confidences.append(float(fixed_uq["confidence"]))
            fixed_entropies.append(float(fixed_uq["entropy"]))
            fixed_variances.append(float(fixed_uq["variance"]))
            fixed_ci_widths.append(float(fixed_uq["ci_upper"]) - float(fixed_uq["ci_lower"]))
            fixed_hazard_probs.append(float(fixed["hazard"]["hazard_probability"]))
            fixed_hazard_statuses.append(str(fixed["hazard"]["status"]))

        adaptive = _adaptive_fusion_estimate(
            measurements=sample_measurements,
            reliabilities=sample_reliabilities,
            faulted_sigmas=(
                faulted_sigmas["lidar"][i],
                faulted_sigmas["ultrasonic"][i],
                faulted_sigmas["radar"][i]),
            core=core,
            adaptive_engine=adaptive_engine,
            uq=uq,
            hazard_assessor=hazard_assessor,
            scene_complexity=scene,
            sensor_agreement=agree,
            reliability_spread=rel_spread,
            effective_sensor_count=eff_count,
            dominance_ratio=dom_ratio,
            confidence_proxy=conf_proxy,
            measurement_spread=spread,
            water_depth=water_d,
            ntu=ntu_val)

        if adaptive is None:
            adaptive_errors.append(np.nan)
            adaptive_failures.append(True)
            adaptive_confidences.append(np.nan)
            adaptive_entropies.append(np.nan)
            adaptive_variances.append(np.nan)
            adaptive_ci_widths.append(np.nan)
            adaptive_hazard_probs.append(np.nan)
            adaptive_hazard_statuses.append("FAIL")
            adaptive_weights_l.append(np.nan)
            adaptive_weights_u.append(np.nan)
            adaptive_weights_r.append(np.nan)
            adaptive_sigmas_l.append(np.nan)
            adaptive_sigmas_u.append(np.nan)
            adaptive_sigmas_r.append(np.nan)
            context_difficulties.append(np.nan)
            adaptation_gates.append(np.nan)
            measurement_consistencies.append(np.nan)
            weight_entropies.append(np.nan)
        else:
            adaptive_uq = adaptive["metrics"]
            adaptive_est = float(adaptive_uq["map_depth"])
            adaptive_errors.append(abs(adaptive_est - truth))
            adaptive_failures.append(False)
            adaptive_confidences.append(float(adaptive_uq["confidence"]))
            adaptive_entropies.append(float(adaptive_uq["entropy"]))
            adaptive_variances.append(float(adaptive_uq["variance"]))
            adaptive_ci_widths.append(float(adaptive_uq["ci_upper"]) - float(adaptive_uq["ci_lower"]))
            adaptive_hazard_probs.append(float(adaptive["hazard"]["hazard_probability"]))
            adaptive_hazard_statuses.append(str(adaptive["hazard"]["status"]))

            adaptive_weights_l.append(float(adaptive["weights"]["lidar"]))
            adaptive_weights_u.append(float(adaptive["weights"]["ultrasonic"]))
            adaptive_weights_r.append(float(adaptive["weights"]["radar"]))

            adaptive_sigmas_l.append(float(adaptive["sigmas"]["lidar"]))
            adaptive_sigmas_u.append(float(adaptive["sigmas"]["ultrasonic"]))
            adaptive_sigmas_r.append(float(adaptive["sigmas"]["radar"]))

            diagnostics = adaptive.get("diagnostics", {})
            context_difficulties.append(float(diagnostics.get("context_difficulty", np.nan)))
            adaptation_gates.append(float(diagnostics.get("adaptation_gate", np.nan)))
            measurement_consistencies.append(float(diagnostics.get("measurement_consistency_mean", np.nan)))
            weight_entropies.append(float(diagnostics.get("weight_entropy", np.nan)))

    single_errors_arr = np.asarray(single_errors, dtype=float)
    fixed_errors_arr = np.asarray(fixed_errors, dtype=float)
    adaptive_errors_arr = np.asarray(adaptive_errors, dtype=float)
    best_single_errors_arr = np.asarray(best_single_errors, dtype=float)

    single_rmse = _safe_rmse(single_errors_arr)
    best_single_rmse = _safe_rmse(best_single_errors_arr)
    fixed_rmse = _safe_rmse(fixed_errors_arr)
    adaptive_rmse = _safe_rmse(adaptive_errors_arr)
    single_failure_rate = _safe_rate(single_failures)

    adaptive_gain = np.nan
    adaptive_vs_single_gain = np.nan
    if np.isfinite(fixed_rmse) and fixed_rmse > 1e-12:
        adaptive_gain = ((fixed_rmse - adaptive_rmse) / fixed_rmse) * 100.0
    if np.isfinite(single_rmse) and single_rmse > 1e-12:
        adaptive_vs_single_gain = ((single_rmse - adaptive_rmse) / single_rmse) * 100.0

    valid = (np.isfinite(adaptive_errors_arr) & np.isfinite(fixed_errors_arr))
    if np.any(valid):
        adaptive_win_rate = (np.mean(adaptive_errors_arr[valid] < fixed_errors_arr[valid]) * 100.0)
    else:
        adaptive_win_rate = np.nan

    return {
        "Experiment": exp_name,
        "Samples": len(base_df),
        "Single_RMSE_cm": single_rmse,
        "Oracle_Best_Single_RMSE_cm": best_single_rmse,
        "Fixed_RMSE_cm": fixed_rmse,
        "Adaptive_RMSE_cm": adaptive_rmse,
        "Single_Failure_Rate_%": single_failure_rate,
        "Adaptive_Gain_%": adaptive_gain,
        "Adaptive_Gain_vs_Single_%": adaptive_vs_single_gain,
        "Single_Degradation_%": np.nan,
        "Fixed_Degradation_%": np.nan,
        "Adaptive_Degradation_%": np.nan,
        "Adaptive_Win_Rate_%": adaptive_win_rate,
        "Mean_Fixed_Confidence": _safe_mean(fixed_confidences),
        "Mean_Adaptive_Confidence": _safe_mean(adaptive_confidences),
        "Mean_Fixed_Entropy": _safe_mean(fixed_entropies),
        "Mean_Adaptive_Entropy": _safe_mean(adaptive_entropies),
        "Mean_Fixed_Variance": _safe_mean(fixed_variances),
        "Mean_Adaptive_Variance": _safe_mean(adaptive_variances),
        "Mean_Fixed_Hazard_Prob": _safe_mean(fixed_hazard_probs),
        "Mean_Adaptive_Hazard_Prob": _safe_mean(adaptive_hazard_probs),
        "Mean_Fixed_CI_Width": _safe_mean(fixed_ci_widths),
        "Mean_Adaptive_CI_Width": _safe_mean(adaptive_ci_widths),
        "Mean_W_Lidar": _safe_mean(adaptive_weights_l),
        "Mean_W_Ultrasonic": _safe_mean(adaptive_weights_u),
        "Mean_W_Radar": _safe_mean(adaptive_weights_r),
        "Mean_Sigma_Lidar": _safe_mean(adaptive_sigmas_l),
        "Mean_Sigma_Ultrasonic": _safe_mean(adaptive_sigmas_u),
        "Mean_Sigma_Radar": _safe_mean(adaptive_sigmas_r),
        "Fixed_Predicted_Hazard_Rate_%": _safe_rate(
            np.asarray(fixed_hazard_statuses, dtype=object) == "HAZARD"),
        "Adaptive_Predicted_Hazard_Rate_%": _safe_rate(
            np.asarray(adaptive_hazard_statuses, dtype=object) == "HAZARD"),
        "Mean_Context_Difficulty": _safe_mean(context_difficulties),
        "Mean_Adaptation_Gate": _safe_mean(adaptation_gates),
        "Mean_Measurement_Consistency": _safe_mean(measurement_consistencies),
        "Mean_Active_Sensor_Count": _safe_mean(active_sensor_counts),
        "Mean_Weight_Entropy": _safe_mean(weight_entropies),
    }

def _split_calibration_evaluation(df: pd.DataFrame, calibration_fraction: float = CALIBRATION_FRACTION,
    random_seed: int = CALIBRATION_RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    n_calibration = int(round(len(df) * calibration_fraction))
    if n_calibration <= 0 or n_calibration >= len(df):
        raise ValueError("Calibration split must leave samples for both calibration and evaluation")

    calibration_indices = indices[:n_calibration]
    evaluation_indices = indices[n_calibration:]
    calibration_df = (df.iloc[calibration_indices].copy().reset_index(drop=True))
    evaluation_df = (df.iloc[evaluation_indices].copy().reset_index(drop=True))
    return calibration_df, evaluation_df

def evaluate_robustness(dataset_path: Path = DATASET_FILE_IN, verbose: bool = True):
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found. Run fusion evaluation first")
    full_df = pd.read_csv(dataset_path).copy()

    _resolve_column(full_df, REQUIRED_CANDIDATES["scenario_id"], "scenario_id")
    truth_col = _resolve_column(full_df, REQUIRED_CANDIDATES["true_depth"],  "true_depth")

    r_lidar_col = _resolve_column(full_df, REQUIRED_CANDIDATES["r_lidar"], "r_lidar")
    r_ultra_col = _resolve_column(full_df, REQUIRED_CANDIDATES["r_ultra"], "r_ultra")
    r_radar_col = _resolve_column(full_df, REQUIRED_CANDIDATES["r_radar"], "r_radar")

    _resolve_column(full_df, REQUIRED_CANDIDATES["water_depth"], "water_depth")
    _resolve_column(full_df, REQUIRED_CANDIDATES["ntu"], "ntu")

    calibration_df, df = _split_calibration_evaluation(full_df)
    calibration_truth = pd.to_numeric(calibration_df[truth_col], errors="coerce").to_numpy(dtype=float)
    cal_z_lidar, cal_z_ultra, cal_z_radar = _resolve_measurements(calibration_df)
    fixed_sigmas = derive_fixed_sigmas(calibration_df, calibration_truth, cal_z_lidar, cal_z_ultra, cal_z_radar)

    true_depth = pd.to_numeric(df[truth_col], errors="coerce").to_numpy(dtype=float)
    z_lidar, z_ultra, z_radar = _resolve_measurements(df)
    r_lidar = pd.to_numeric(df[r_lidar_col], errors="coerce").to_numpy(dtype=float)
    r_ultra = pd.to_numeric(df[r_ultra_col], errors="coerce").to_numpy(dtype=float)
    r_radar = pd.to_numeric(df[r_radar_col], errors="coerce").to_numpy(dtype=float)

    base_streams = {
        "lidar": z_lidar.copy(),
        "ultra": z_ultra.copy(),
        "radar": z_radar.copy(),
    }

    # Fault-aware sigma seed before contextual expansion
    sigma_lidar, sigma_ultra, sigma_radar = _sensor_sigmas(r_lidar, r_ultra, r_radar)
    base_sigmas = {
        "lidar": sigma_lidar,
        "ultrasonic": sigma_ultra,
        "radar": sigma_radar,
    }

    base_reliabilities = {
        "lidar": r_lidar.copy(),
        "ultrasonic": r_ultra.copy(),
        "radar": r_radar.copy(),
    }

    water_depth = pd.to_numeric(df["water_depth"], errors="coerce").fillna(0.0).clip(lower=0.0)
    ntu = pd.to_numeric(df["ntu"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Clean reference context for baseline reporting
    clean_context_bundle = _build_context_bundle(df,
        z_lidar, z_ultra, z_radar,
        r_lidar, r_ultra, r_radar,
        water_depth, ntu)

    dropout = DropoutInjector(random_seed=101)
    noise = NoiseInjector(random_seed=202)
    delay = DelayInjector(fps=20)
    fault_effects = FaultEffects()
    sync = SyncInjector()
    injectors = {
        "dropout": dropout,
        "noise": noise,
        "delay": delay,
        "sync": sync,
    }

    experiment_fault_plans = [
        ("Baseline (No Faults)", {}),
        ("LiDAR Low Dropout (10%)", {"lidar": [{"fault_type": "dropout", "severity": "low"}]}),
        ("LiDAR High Dropout (50%)", {"lidar": [{"fault_type": "dropout", "severity": "high"}]}),
        ("Ultrasonic High Dropout (50%)", {"ultrasonic": [{"fault_type": "dropout", "severity": "high"}]}),
        ("Radar High Dropout (50%)", {"radar": [{"fault_type": "dropout", "severity": "high"}]}),
        ("Radar Severe Noise", {"radar": [{"fault_type": "noise", "severity": "severe"}]}),
        ("Ultrasonic High Delay", {"ultrasonic": [{"fault_type": "delay", "severity": "high"}]}),
        ("Radar High Sync Error", {"radar": [{"fault_type": "sync", "severity": "high", "direction": "backward"}]}),
        
        ("Compound Scenario A", {
            "lidar": [{"fault_type": "dropout", "severity": "high"}],
            "radar": [{"fault_type": "noise", "severity": "severe"}]}),

        ("Compound Scenario C", {
            "lidar": [{"fault_type": "dropout", "severity": "medium"}],
            "ultrasonic": [{"fault_type": "noise", "severity": "severe"}],
            "radar": [{"fault_type": "sync", "severity": "high", "direction": "backward"}]})]

    core = BayesianFusionCore(
        depth_min=DEPTH_MIN_CM,
        depth_max=DEPTH_MAX_CM,
        resolution=DEPTH_RESOLUTION)

    adaptive_engine = AdaptiveFusionEngine(core,
        gamma=ADAPTIVE_CALIBRATION["gamma"],
        weight_blend=ADAPTIVE_CALIBRATION["weight_blend"],
        sigma_scale=ADAPTIVE_CALIBRATION["sigma_scale"],
        sigma_offset=ADAPTIVE_CALIBRATION["sigma_offset"],
        sigma_min=ADAPTIVE_CALIBRATION["sigma_min"],
        sigma_max=ADAPTIVE_CALIBRATION["sigma_max"],
        sensor_prior=ADAPTIVE_CALIBRATION["sensor_prior"],
        context_blend_boost=ADAPTIVE_CALIBRATION["context_blend_boost"],
        context_sigma_boost=ADAPTIVE_CALIBRATION["context_sigma_boost"],
        weight_floor=ADAPTIVE_CALIBRATION["weight_floor"],
        quality_threshold=ADAPTIVE_CALIBRATION["quality_threshold"])

    uq = UncertaintyQuantification()
    hazard_assessor = HazardAssessment(threshold_cm=10.0, decision_boundary=0.8)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp_name, fault_plan in experiment_fault_plans:
        streams: Dict[str, np.ndarray] = {}
        reliabilities: Dict[str, np.ndarray] = {}
        faulted_sigmas: Dict[str, np.ndarray] = {}

        for sensor_name, stream_key in (("lidar", "lidar"), ("ultrasonic", "ultra"), ("radar", "radar")):
            faulted_measurements, faulted_reliability, faulted_sigma = _apply_sensor_fault_plan(
                sensor_name=sensor_name,
                measurements=base_streams[stream_key],
                reliability=base_reliabilities[sensor_name],
                sigma=base_sigmas[sensor_name],
                fault_plan=fault_plan.get(sensor_name, []),
                injectors=injectors,
                fault_effects=fault_effects)

            streams[stream_key] = faulted_measurements
            reliabilities[sensor_name] = faulted_reliability
            faulted_sigmas[sensor_name] = faulted_sigma

        # Recompute context per experiment so faulted measurements and reliabilities actually influence adaptation
        experiment_context_bundle = _build_context_bundle(df,
            streams["lidar"],
            streams["ultra"],
            streams["radar"],
            reliabilities["lidar"],
            reliabilities["ultrasonic"],
            reliabilities["radar"],
            water_depth, ntu)

        row = _evaluate_stream_set(
            exp_name=exp_name,
            base_df=df,
            streams=streams,
            reliabilities=reliabilities,
            fixed_sigmas=fixed_sigmas,
            faulted_sigmas=faulted_sigmas,
            context=experiment_context_bundle,
            true_depth=true_depth,
            core=core,
            adaptive_engine=adaptive_engine,
            uq=uq,
            hazard_assessor=hazard_assessor)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    baseline_mask = results_df["Experiment"] == "Baseline (No Faults)"

    if not baseline_mask.any():
        raise RuntimeError("Baseline experiment was not generated")
    baseline_row = results_df.loc[baseline_mask].iloc[0]

    results_df["Fixed_Degradation_%"] = (
        (results_df["Fixed_RMSE_cm"] - baseline_row["Fixed_RMSE_cm"]) / max(
            float(baseline_row["Fixed_RMSE_cm"]), 1e-12)) * 100.0

    results_df["Adaptive_Degradation_%"] = (
        (results_df["Adaptive_RMSE_cm"] - baseline_row["Adaptive_RMSE_cm"]) / max(
            float(baseline_row["Adaptive_RMSE_cm"]), 1e-12)) * 100.0

    results_df["Single_Degradation_%"] = (
        (results_df["Single_RMSE_cm"] - baseline_row["Single_RMSE_cm"]) / max(
            float(baseline_row["Single_RMSE_cm"]), 1e-12)) * 100.0

    results_df["Oracle_Best_Single_Degradation_%"] = (
        (results_df["Oracle_Best_Single_RMSE_cm"] - baseline_row["Oracle_Best_Single_RMSE_cm"]) / max(
            float(baseline_row["Oracle_Best_Single_RMSE_cm"]), 1e-12)) * 100.0

    results_df.to_csv(ROBUSTNESS_RESULTS_FILE, index=False)

    summary = {
        "total_dataset_samples": int(len(full_df)),
        "experiments": int(len(results_df)),
        "calibration_samples": int(len(calibration_df)),
        "evaluation_samples": int(len(df)),
        "calibration_fraction": float(CALIBRATION_FRACTION),
        "calibration_random_seed": int(CALIBRATION_RANDOM_SEED),
        "calibrated_fixed_sigma_lidar_cm": float(fixed_sigmas[0]),
        "calibrated_fixed_sigma_ultrasonic_cm": float(fixed_sigmas[1]),
        "calibrated_fixed_sigma_radar_cm": float(fixed_sigmas[2]),
        "baseline_single_rmse_cm": float(baseline_row["Single_RMSE_cm"]),
        "baseline_oracle_best_single_rmse_cm": float(baseline_row["Oracle_Best_Single_RMSE_cm"]),
        "baseline_fixed_rmse_cm": float(baseline_row["Fixed_RMSE_cm"]),
        "baseline_adaptive_rmse_cm": float(baseline_row["Adaptive_RMSE_cm"]),
        "mean_single_rmse_cm": _safe_mean(results_df["Single_RMSE_cm"]),
        "mean_oracle_best_single_rmse_cm": _safe_mean(results_df["Oracle_Best_Single_RMSE_cm"]),
        "mean_fixed_rmse_cm": _safe_mean(results_df["Fixed_RMSE_cm"]),
        "mean_adaptive_rmse_cm": _safe_mean(results_df["Adaptive_RMSE_cm"]),
        "mean_single_failure_rate_%": _safe_mean(results_df["Single_Failure_Rate_%"]),
        "mean_adaptive_gain_%": _safe_mean(results_df["Adaptive_Gain_%"]),
        "mean_adaptive_gain_vs_single_%": _safe_mean(results_df["Adaptive_Gain_vs_Single_%"]),
        "mean_adaptive_win_rate_%": _safe_mean(results_df["Adaptive_Win_Rate_%"]),
        "mean_fixed_confidence": _safe_mean(results_df["Mean_Fixed_Confidence"]),
        "mean_adaptive_confidence": _safe_mean(results_df["Mean_Adaptive_Confidence"]),
        "mean_fixed_entropy": _safe_mean(results_df["Mean_Fixed_Entropy"]),
        "mean_adaptive_entropy": _safe_mean(results_df["Mean_Adaptive_Entropy"]),
        "mean_fixed_variance": _safe_mean(results_df["Mean_Fixed_Variance"]),
        "mean_adaptive_variance": _safe_mean(results_df["Mean_Adaptive_Variance"]),
        "mean_fixed_hazard_probability": _safe_mean(results_df["Mean_Fixed_Hazard_Prob"]),
        "mean_adaptive_hazard_probability": _safe_mean(results_df["Mean_Adaptive_Hazard_Prob"]),
        "mean_fixed_ci_width": _safe_mean(results_df["Mean_Fixed_CI_Width"]),
        "mean_adaptive_ci_width": _safe_mean(results_df["Mean_Adaptive_CI_Width"]),
        "mean_weight_lidar": _safe_mean(results_df["Mean_W_Lidar"]),
        "mean_weight_ultrasonic": _safe_mean(results_df["Mean_W_Ultrasonic"]),
        "mean_weight_radar": _safe_mean(results_df["Mean_W_Radar"]),
        "mean_sigma_lidar": _safe_mean(results_df["Mean_Sigma_Lidar"]),
        "mean_sigma_ultrasonic": _safe_mean(results_df["Mean_Sigma_Ultrasonic"]),
        "mean_sigma_radar": _safe_mean(results_df["Mean_Sigma_Radar"]),
        "mean_fixed_hazard_rate_%": _safe_mean(results_df["Fixed_Predicted_Hazard_Rate_%"]),
        "mean_adaptive_hazard_rate_%": _safe_mean(results_df["Adaptive_Predicted_Hazard_Rate_%"]),
        "max_fixed_degradation_%": float(results_df["Fixed_Degradation_%"].max()),
        "max_adaptive_degradation_%": float(results_df["Adaptive_Degradation_%"].max()),
        "max_oracle_best_single_degradation_%": float(results_df[
            "Oracle_Best_Single_Degradation_%"].max()),
        "baseline_scene_complexity": float(np.mean(clean_context_bundle["scene_complexity"])),
        "baseline_sensor_agreement": float(np.mean(clean_context_bundle["sensor_agreement"])),
        "baseline_measurement_spread": float(np.mean(clean_context_bundle["measurement_spread"])),
        "baseline_reliability_spread": float(np.mean(clean_context_bundle["reliability_spread"])),
        "baseline_effective_sensor_count": float(np.mean(clean_context_bundle["effective_sensor_count"])),
        "mean_context_difficulty": _safe_mean(results_df["Mean_Context_Difficulty"]),
        "mean_adaptation_gate": _safe_mean(results_df["Mean_Adaptation_Gate"]),
        "mean_measurement_consistency": _safe_mean(results_df["Mean_Measurement_Consistency"]),
        "mean_active_sensor_count": _safe_mean(results_df["Mean_Active_Sensor_Count"]),
        "mean_weight_entropy": _safe_mean(results_df["Mean_Weight_Entropy"]),
        "reference_depth_source": "true_depth",
    }

    summary_df = pd.DataFrame([summary]).round(6)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    if verbose:
        print()
        print("=" * 60)
        print("FloodTwin-HIL Fault Robustness Results")
        print("=" * 60)
        print()

        columns_to_show = [
            "Experiment",
            "Single_RMSE_cm",
            "Oracle_Best_Single_RMSE_cm",
            "Fixed_RMSE_cm",
            "Adaptive_RMSE_cm",
            "Single_Failure_Rate_%",
            "Adaptive_Gain_%",
            "Adaptive_Win_Rate_%",
        ]

        print(results_df[columns_to_show].round(3).to_string(index=False))
        print()
        print("=" * 60)
        print(f"Results Saved : {ROBUSTNESS_RESULTS_FILE}")
        print(f"Summary Saved : {SUMMARY_FILE}")
        print("-" * 60)
        print(summary_df.T.to_string(header=False))
        print()
        print("=" * 60)

    return results_df, summary_df

if __name__ == "__main__":
    evaluate_robustness()