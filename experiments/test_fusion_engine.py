import hashlib
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd

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

CALIBRATION_MODULO = 5
CALIBRATION_SEED = 20260906
BOOTSTRAP_SAMPLES = 5000
MIN_CALIBRATION_SAMPLES_PER_SENSOR = 100
MAX_EVALUATION_ROWS = 0

# ------------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------------
def _choose_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> Optional[str]:
    for col in candidates:
        if col in df.columns: return col
    return None

def _numeric_series(df: pd.DataFrame, candidates: Sequence[str],
    default: float = np.nan) -> pd.Series:
    col = _choose_column(df, candidates, label="/".join(candidates))
    if col is None: return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def _safe_float(value: Any, default: float = np.nan) -> float:
    try: x = float(value)
    except (TypeError, ValueError): return float(default)
    return x if np.isfinite(x) else float(default)

def _safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)    
    if int(mask.sum()) < 2: return float("nan")
    sx = x[mask]
    sy = y[mask]
    if np.std(sx) <= 1e-12 or np.std(sy) <= 1e-12: return float("nan")
    return float(np.corrcoef(sx, sy)[0, 1])

def _nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")

def _nanstd(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")

def _rmse(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")

def _calibration_mask(scenario_ids: pd.Series) -> np.ndarray:
    namespace = str(CALIBRATION_SEED).encode("utf-8")
    flags = []    
    for value in scenario_ids.astype(str):
        digest = hashlib.sha256(namespace + b"|" + value.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % CALIBRATION_MODULO
        flags.append(bucket == 0)
    return np.asarray(flags, dtype=bool)

# ------------------------------------------------------------------
# Dataset / inference boundary
# ------------------------------------------------------------------
def _validate_dataset(df: pd.DataFrame) -> None:
    required = ["scenario_id", "true_depth", "lidar",
        "ultrasonic", "radar", "R_lidar", "R_ultrasonic", "R_radar"]
    missing = [c for c in required if c not in df.columns]

    if missing: raise ValueError(f"dataset_v2 is missing required columns: {missing}")
    if df["scenario_id"].isna().any(): raise ValueError("scenario_id contains missing values")
    if not df["scenario_id"].is_unique:
        raise ValueError("scenario_id must be unique for a disjoint calibration/evaluation split")

def _resolve_measurements(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = {
        "lidar": _choose_column(df, ["lidar", "z_lidar", "lidar_measurement"], "lidar"),
        "ultrasonic": _choose_column(df, ["ultrasonic", "z_ultra", "z_ultrasonic", "ultrasonic_measurement"], "ultrasonic"),
        "radar": _choose_column(df, ["radar", "z_radar", "radar_measurement"], "radar")}
    
    if any(v is None for v in columns.values()):
        raise ValueError("dataset_v2 must contain lidar, ultrasonic, and radar measurement columns")

    z_lidar = pd.to_numeric(df[columns["lidar"]], errors="coerce").to_numpy(dtype=float)
    z_ultra = pd.to_numeric(df[columns["ultrasonic"]], errors="coerce").to_numpy(dtype=float)
    z_radar = pd.to_numeric(df[columns["radar"]], errors="coerce").to_numpy(dtype=float)
    return z_lidar, z_ultra, z_radar

def _prepare_measurements(*measurements: np.ndarray) -> Tuple[np.ndarray, ...]:
    prepared = []    
    for values in measurements:
        values = np.asarray(values, dtype=float).copy()
        valid = np.isfinite(values) & (values >= DEPTH_MIN_CM) & (values <= DEPTH_MAX_CM)
        values[~valid] = np.nan
        prepared.append(values)        
    return tuple(prepared)

def _measurement_validity_mask(*measurements: np.ndarray) -> np.ndarray:
    raw = np.column_stack(measurements).astype(float)
    return np.isfinite(raw) & (raw >= DEPTH_MIN_CM) & (raw <= DEPTH_MAX_CM)

def _observable_water_context(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "water_context_depth_proxy" not in df.columns:
        values = pd.Series(np.nan, index=df.index, dtype=float)
    else:
        values = pd.to_numeric(df["water_context_depth_proxy"], errors="coerce")
    available = values.notna()
    values = values.fillna(0.0).clip(DEPTH_MIN_CM, 20.0)
    return values.astype(float), available.astype(bool)

def _observable_ntu(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    column = _choose_column(df, ["ntu_observed", "ntu"], label="ntu_observed/ntu")
    if column is None:
        values = pd.Series(np.nan, index=df.index, dtype=float)
    else:
        values = pd.to_numeric(df[column], errors="coerce")
    available = values.notna()
    values = (values.fillna(0.0).clip(0.0, 500.0).astype(float))
    return values, available.astype(bool)

def _observable_scene(
    df: pd.DataFrame, water_proxy: pd.Series, ntu: pd.Series, clutter: pd.Series,
    agreement: np.ndarray, measurement_spread: np.ndarray, spread_ref: float) -> pd.Series:

    if "scene_complexity" in df.columns:
        return (pd.to_numeric(df["scene_complexity"],
        errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float))

    water_norm = (np.clip(water_proxy.to_numpy(dtype=float) / 20.0, 0.0, 1.0,) ** 1.20)
    ntu_norm = np.clip(ntu.to_numpy(dtype=float) / 500.0, 0.0, 1.0)
    clutter_norm = (np.clip(clutter.to_numpy(dtype=float) / 0.30, 0.0, 1.0) ** 1.20)

    measurement_spread = np.asarray(measurement_spread, dtype=float)
    spread_ref = float(spread_ref)
    if not np.isfinite(spread_ref) or spread_ref <= 0.0:
        spread_ref = 1e-6
    spread_norm = np.clip(measurement_spread / spread_ref, 0.0, 1.0)
    agreement_difficulty = (1.0 - np.clip(np.asarray(agreement, dtype=float), 0.0, 1.0))

    scene = np.clip(0.22 * water_norm
        + 0.20 * ntu_norm
        + 0.16 * clutter_norm
        + 0.22 * spread_norm
        + 0.20 * agreement_difficulty, 0.0, 1.0)

    return pd.Series(scene, index=df.index, dtype=float)

def _reliability_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = []    
    for sensor in ("lidar", "ultrasonic", "radar"):
        col = f"R_{sensor}"
        base_col = f"R_{sensor}_base"        
        if col in df.columns:
            r = pd.to_numeric(df[col], errors="coerce")
            if base_col in df.columns:
                r = r.fillna(pd.to_numeric(df[base_col], errors="coerce"))
        elif base_col in df.columns:
            r = pd.to_numeric(df[base_col], errors="coerce")
        else:
            raise ValueError(f"Missing {col} and {base_col}")            
        r = np.clip(np.nan_to_num(r.to_numpy(dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        values.append(r)        
    return tuple(values)

def _reliability_spread(r_lidar: float, r_ultra: float, r_radar: float) -> float:
    return float(np.clip(max(r_lidar, r_ultra, r_radar) - min(r_lidar, r_ultra, r_radar), 0.0, 1.0))

def _observable_sensor_agreement(measurements: Sequence[float], active_mask: Sequence[bool]) -> float:
    values = np.asarray(measurements, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    values = values[active & np.isfinite(values)]    
    if values.size < 2: return 1.0 if values.size == 1 else 0.0    
    pairwise = []

    for i in range(values.size):
        for j in range(i + 1, values.size):
            pairwise.append(abs(float(values[i] - values[j])))            
    pairwise_mean = float(np.mean(pairwise))
    mean_abs = float(np.mean(np.abs(values))) + 1e-6
    return float(np.clip(1.0 / (1.0 + pairwise_mean / mean_abs), 0.0, 1.0))

def _observable_measurement_spread(
    measurements: Sequence[float], active_mask: Sequence[bool]) -> float:
    values = np.asarray(measurements, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    values = values[active & np.isfinite(values)]    
    if values.size < 2: return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return float(1.4826 * mad)

# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------
def _derive_fixed_sigmas_from_calibration(
    calibration_df: pd.DataFrame, z_lidar: np.ndarray, z_ultra: np.ndarray,
    z_radar: np.ndarray) -> Tuple[Tuple[float, float, float], Dict[str, float]]:

    truth = pd.to_numeric(calibration_df["true_depth"], errors="coerce").to_numpy(dtype=float)
    if truth.size == 0 or not np.isfinite(truth).all():
        raise ValueError("Calibration partition contains invalid true_depth values")

    measurements = {
        "lidar": np.asarray(z_lidar, dtype=float),
        "ultrasonic": np.asarray(z_ultra, dtype=float),
        "radar": np.asarray(z_radar, dtype=float),
    }
    sigmas = []
    diagnostics: Dict[str, float] = {}    
    for sensor, values in measurements.items():
        valid = np.isfinite(values) & np.isfinite(truth)
        count = int(valid.sum())
        
        if count < MIN_CALIBRATION_SAMPLES_PER_SENSOR:
            raise ValueError(f"Insufficient calibration samples for {sensor}: {count} "
                f"< {MIN_CALIBRATION_SAMPLES_PER_SENSOR}")
        
        residuals = values[valid] - truth[valid]
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        lo, hi = FIXED_SIGMA_BOUNDS[sensor]
        sigma = float(np.clip(rms, lo, hi))
        
        sigmas.append(sigma)
        diagnostics[f"calibration_rms_{sensor}"] = rms
        diagnostics[f"fixed_sigma_{sensor}"] = sigma
        diagnostics[f"calibration_count_{sensor}"] = count

    diagnostics["calibration_mean_depth"] = float(np.mean(truth))
    diagnostics["calibration_std_depth"] = float(np.std(truth, ddof=1)) if truth.size > 1 else 0.0
    return (float(sigmas[0]), float(sigmas[1]), float(sigmas[2])), diagnostics

# ------------------------------------------------------------------
# Baseline and posterior scoring
# ------------------------------------------------------------------
def _fixed_fusion_estimate(
    fixed_engine: FixedFusionEngine, core: BayesianFusionCore,
    measurements: Sequence[float], active_mask: Sequence[bool],
    fixed_sigmas: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
    
    active = np.asarray(active_mask, dtype=bool)
    z = np.asarray(measurements, dtype=float)
    
    if active.shape != (3,) or z.shape != (3,):
        raise ValueError("measurements and active_mask must each describe three sensors")
    if not np.any(active): return None

    if int(np.sum(active)) == 3:
        return fixed_engine.estimate(
            float(z[0]), float(z[1]), float(z[2]), sigmas=fixed_sigmas)

    likelihoods = []
    active_indices = []
    names = ("lidar", "ultrasonic", "radar")
    
    for i in range(3):
        if active[i] and np.isfinite(z[i]):
            likelihoods.append(core.compute_likelihood(float(z[i]), fixed_sigmas[i]))
            active_indices.append(i)
            
    if not likelihoods: return None
    weights = np.ones(len(likelihoods), dtype=float) / len(likelihoods)
    pooled = core.combine_likelihoods(likelihoods, weights)
    posterior = core.compute_posterior(pooled)
    
    equal_weight = 1.0 / len(active_indices)
    sensor_weights = {name: 0.0 for name in names}
    for i in active_indices: sensor_weights[names[i]] = equal_weight

    return {
        "posterior": posterior,
        "depth_map": float(core.map_estimate(posterior)),
        "depth_mean": float(core.expected_depth(posterior)),
        "variance": float(core.posterior_variance(posterior)),
        "entropy": float(core.entropy(posterior)),
        "weights": sensor_weights,
    }

def _posterior_metrics(
    uq: UncertaintyQuantification, core: BayesianFusionCore,
    posterior: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:    
    if posterior is None: return None
    return uq.compute(core.depths, posterior, core.delta_x)

def _posterior_scoring_metrics(
    core: BayesianFusionCore, posterior: Optional[np.ndarray], truth: float) -> Dict[str, float]:    
    if posterior is None or not np.isfinite(truth):
        return {"coverage_95": np.nan, "nll": np.nan, "crps": np.nan}

    mass = core.posterior_mass(posterior)
    cdf = np.cumsum(mass)
    ci_lower, ci_upper = core.credible_interval(posterior, mass_level=0.95)
    coverage = float(ci_lower <= truth <= ci_upper)

    # Density score on the same discretized depth grid used for fusion
    density_at_truth = float(np.interp(truth, core.depths, np.asarray(posterior, dtype=float)))
    nll = float(-np.log(max(density_at_truth, 1e-15)))

    # Standard discrete-grid CRPS approximation
    indicator = (core.depths >= truth).astype(float)
    crps = float(np.sum((cdf - indicator) ** 2) * core.delta_x)
    return {"coverage_95": coverage, "nll": nll, "crps": crps}

def _bootstrap_mean_difference(
    adaptive_error: np.ndarray, fixed_error: np.ndarray,
    seed: int = CALIBRATION_SEED, samples: int = BOOTSTRAP_SAMPLES) -> Tuple[float, float, float]:
    
    a = np.asarray(adaptive_error, dtype=float)
    f = np.asarray(fixed_error, dtype=float)
    mask = np.isfinite(a) & np.isfinite(f)    
    a = a[mask]
    f = f[mask]
    
    if a.size < 2: return float("nan"), float("nan"), float("nan")
    observed = float(np.mean(a - f))
    rng = np.random.default_rng(seed)
    n = a.size
    paired_difference = a - f
    bootstrap = np.empty(samples, dtype=float)
    batch_size = 25
    
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        sample_idx = rng.integers(0, n, size=(count, n))
        bootstrap[start:start + count] = np.mean(paired_difference[sample_idx], axis=1)

    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return observed, float(low), float(high)

# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def run_fusion_pipeline():
    if not DATASET_FILE_IN.exists():
        raise FileNotFoundError(f"Input dataset not found: {DATASET_FILE_IN}")

    df = pd.read_csv(DATASET_FILE_IN).copy()
    _validate_dataset(df)

    calibration_mask = _calibration_mask(df["scenario_id"])
    if calibration_mask.sum() == 0 or (~calibration_mask).sum() == 0:
        raise ValueError("Calibration/evaluation split produced an empty partition")

    calibration_df = df.loc[calibration_mask].reset_index(drop=True).copy()
    z_lidar_raw, z_ultra_raw, z_radar_raw = _resolve_measurements(df)
    z_lidar, z_ultra, z_radar = _prepare_measurements(z_lidar_raw, z_ultra_raw, z_radar_raw)
    validity_all = _measurement_validity_mask(z_lidar, z_ultra, z_radar)

    # Truth is used here only for the dedicated calibration operation
    calibration_indices = np.flatnonzero(calibration_mask)
    fixed_sigmas, calibration_diag = _derive_fixed_sigmas_from_calibration(calibration_df,
        z_lidar[calibration_indices], z_ultra[calibration_indices], z_radar[calibration_indices])

    water_proxy_all, water_available_all = _observable_water_context(df)
    ntu_all, ntu_available_all = _observable_ntu(df)
    r_lidar_all, r_ultra_all, r_radar_all = _reliability_values(df)

    # A positive reliability plus a valid measurement is required for participation
    active_all = validity_all & np.column_stack([
        r_lidar_all > 0.0, r_ultra_all > 0.0, r_radar_all > 0.0])

    clutter_all = pd.Series(0.10, index=df.index, dtype=float)
    if "clutter_probability" in df.columns:
        clutter_all = pd.to_numeric(
            df["clutter_probability"], errors="coerce").fillna(0.10).clip(0.0, 0.30)
    elif "clutter" in df.columns:
        clutter_all = pd.to_numeric(df["clutter"], errors="coerce").fillna(0.10).clip(0.0, 0.30)
        
    temp_agreement = np.array([
        _observable_sensor_agreement((z_lidar[i], z_ultra[i], z_radar[i]), tuple(active_all[i]))
        for i in range(len(df))], dtype=float)
        
    temp_spread = np.array([
        _observable_measurement_spread((z_lidar[i], z_ultra[i], z_radar[i]), tuple(active_all[i]))
        for i in range(len(df))], dtype=float)
        
    calibration_spread = temp_spread[calibration_mask]
    spread_ref = (float(np.percentile(calibration_spread, 95)) if calibration_spread.size else 0.0)

    scene_all = _observable_scene(
        df=df, water_proxy=water_proxy_all, ntu=ntu_all, clutter=clutter_all,
        agreement=temp_agreement, measurement_spread=temp_spread, spread_ref=spread_ref)

    core = BayesianFusionCore(
        depth_min=DEPTH_MIN_CM, depth_max=DEPTH_MAX_CM, resolution=DEPTH_RESOLUTION)
    fixed_engine = FixedFusionEngine(core)
    adaptive_engine = AdaptiveFusionEngine(core)
    uq = UncertaintyQuantification()
    hazard_assessor = HazardAssessment(threshold_cm=10.0, decision_boundary=0.80)

    eval_indices = np.flatnonzero(~calibration_mask)
    if MAX_EVALUATION_ROWS > 0: eval_indices = eval_indices[:MAX_EVALUATION_ROWS]
    
    eval_df = df.iloc[eval_indices].reset_index(drop=True).copy()
    eval_z_lidar = z_lidar[eval_indices]
    eval_z_ultra = z_ultra[eval_indices]
    eval_z_radar = z_radar[eval_indices]
    eval_r_lidar = r_lidar_all[eval_indices]
    eval_r_ultra = r_ultra_all[eval_indices]
    eval_r_radar = r_radar_all[eval_indices]
    eval_active = active_all[eval_indices]
    eval_scene = scene_all.iloc[eval_indices].to_numpy(dtype=float)
    eval_water = water_proxy_all.iloc[eval_indices].to_numpy(dtype=float)
    eval_ntu = ntu_all.iloc[eval_indices].to_numpy(dtype=float)
    eval_water_available = water_available_all.iloc[eval_indices].to_numpy(dtype=bool)
    eval_ntu_available = ntu_available_all.iloc[eval_indices].to_numpy(dtype=bool)

    rows = []
    failure_count_fixed = 0
    failure_count_adaptive = 0
    names = ("lidar", "ultrasonic", "radar")

    for j, row_src in enumerate(eval_df.itertuples(index=False)):
        z = (float(eval_z_lidar[j]) if np.isfinite(eval_z_lidar[j]) else np.nan,
            float(eval_z_ultra[j]) if np.isfinite(eval_z_ultra[j]) else np.nan,
            float(eval_z_radar[j]) if np.isfinite(eval_z_radar[j]) else np.nan)
             
        r = (float(eval_r_lidar[j]), float(eval_r_ultra[j]), float(eval_r_radar[j]))
        row_active = tuple(bool(v) for v in eval_active[j])
        scene = float(eval_scene[j])
        water_proxy = float(eval_water[j])
        ntu_value = float(eval_ntu[j])
        rel_spread = _reliability_spread(*r)
        observable_agreement = _observable_sensor_agreement(z, row_active)
        observable_measurement_spread = _observable_measurement_spread(z, row_active)

        # Inference path: only measurements + observable context
        fixed = _fixed_fusion_estimate(
            fixed_engine=fixed_engine,
            core=core, measurements=z,
            active_mask=row_active,
            fixed_sigmas=fixed_sigmas)

        if fixed is None:
            failure_count_fixed += 1
            fixed_metrics = None
        else:
            fixed_metrics = _posterior_metrics(uq, core, fixed["posterior"])

        adaptive = adaptive_engine.fuse(
            z_lidar=z[0], z_ultra=z[1], z_radar=z[2],
            r_lidar=r[0], r_ultra=r[1], r_radar=r[2],
            sigmas=fixed_sigmas,
            scene_complexity=scene,
            reliability_spread=rel_spread,
            water_depth=water_proxy,
            ntu=ntu_value, active_mask=row_active)

        if adaptive.get("status") != "OK":
            failure_count_adaptive += 1
            adaptive_metrics = None
        else:
            adaptive_metrics = _posterior_metrics(uq, core, adaptive["posterior"])

        # Evaluation truth is accessed only after both inference paths finish
        true_depth = _safe_float(getattr(row_src, "true_depth"), np.nan)
        if not np.isfinite(true_depth): raise ValueError("Evaluation row contains invalid true_depth")
        true_depth = float(np.clip(true_depth, DEPTH_MIN_CM, DEPTH_MAX_CM))

        observations = np.asarray(z, dtype=float)
        abs_observation_error = np.abs(observations - true_depth)
        
        if np.isfinite(abs_observation_error).any():
            best_idx = int(np.nanargmin(abs_observation_error))
            best_name = names[best_idx]
            best_estimate = float(observations[best_idx])
            best_error = float(abs_observation_error[best_idx])
        else:
            best_name = "FAIL"
            best_estimate = np.nan
            best_error = np.nan

        single_estimate = float(observations[0]) if np.isfinite(observations[0]) else np.nan
        single_error = abs(single_estimate - true_depth) if np.isfinite(single_estimate) else np.nan

        if fixed_metrics is None:
            fixed_map_estimate = np.nan
            fixed_mean_estimate = np.nan
            fixed_map_error = np.nan
            fixed_mean_error = np.nan
            fixed_scoring = {"coverage_95": np.nan, "nll": np.nan, "crps": np.nan}
        else:
            fixed_map_estimate = float(fixed_metrics["map_depth"])
            fixed_mean_estimate = float(fixed_metrics["expected_depth"])
            fixed_map_error = abs(fixed_map_estimate - true_depth)
            fixed_mean_error = abs(fixed_mean_estimate - true_depth)
            fixed_scoring = _posterior_scoring_metrics(core, fixed["posterior"], true_depth)

        if adaptive_metrics is None:
            adaptive_map_estimate = np.nan
            adaptive_mean_estimate = np.nan
            adaptive_map_error = np.nan
            adaptive_mean_error = np.nan
            adaptive_scoring = {"coverage_95": np.nan, "nll": np.nan, "crps": np.nan}
            
            hazard = {
                "hazard_probability": np.nan,
                "status": "FAIL",
                "risk_score": np.nan,
                "expected_hazard_depth": np.nan,
            }
            
            adaptive_values = {
                "confidence": np.nan,
                "entropy": np.nan,
                "variance": np.nan,
                "posterior_peak": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "ci_width_95": np.nan,
                "entropy_score": np.nan,
                "peak_score": np.nan,
                "interval_score": np.nan,
                "variance_score": np.nan,
                "effective_support": np.nan,
                "support_fraction": np.nan,
                "gap_score": np.nan,
            }
            adaptive_diag = {}
        else:
            adaptive_map_estimate = float(adaptive_metrics["map_depth"])
            adaptive_mean_estimate = float(adaptive_metrics["expected_depth"])
            adaptive_map_error = abs(adaptive_map_estimate - true_depth)
            adaptive_mean_error = abs(adaptive_mean_estimate - true_depth)
            adaptive_scoring = _posterior_scoring_metrics(core, adaptive["posterior"], true_depth)
            
            hazard = hazard_assessor.evaluate(
                core.depths, adaptive["posterior"], core.delta_x,
                fusion_confidence=float(adaptive_metrics["confidence"]),
                scene_complexity=scene)
                
            adaptive_values = {
                "confidence": float(adaptive_metrics["confidence"]),
                "entropy": float(adaptive_metrics["entropy"]),
                "variance": float(adaptive_metrics["variance"]),
                "posterior_peak": float(adaptive_metrics["posterior_peak"]),
                "ci_lower": float(adaptive_metrics["ci_lower"]),
                "ci_upper": float(adaptive_metrics["ci_upper"]),
                "ci_width_95": float(adaptive_metrics.get("ci_width_95",
                adaptive_metrics["ci_upper"] - adaptive_metrics["ci_lower"])),
                "entropy_score": float(adaptive_metrics.get("entropy_score", np.nan)),
                "peak_score": float(adaptive_metrics.get("peak_score", np.nan)),
                "interval_score": float(adaptive_metrics.get("interval_score", np.nan)),
                "variance_score": float(adaptive_metrics.get("variance_score", np.nan)),
                "effective_support": float(adaptive_metrics.get("effective_support", np.nan)),
                "support_fraction": float(adaptive_metrics.get("support_fraction", np.nan)),
                "gap_score": float(adaptive_metrics.get("gap_score", np.nan)),
            }
            adaptive_diag = adaptive["diagnostics"]

        row = {
            "scenario_id": getattr(row_src, "scenario_id"),
            "split": "evaluation",
            "true_depth": true_depth,
            "true_hazard": float(true_depth >= 10.0),

            "water_context_depth_proxy": water_proxy,
            "water_context_available": bool(eval_water_available[j]),
            "ntu_observed": ntu_value,
            "ntu_available": bool(eval_ntu_available[j]),
            "scene_complexity": scene,
            "sensor_agreement": observable_agreement,
            "measurement_spread": observable_measurement_spread,
            "reliability_spread": rel_spread,

            "lidar_available": bool(validity_all[eval_indices[j], 0]),
            "ultrasonic_available": bool(validity_all[eval_indices[j], 1]),
            "radar_available": bool(validity_all[eval_indices[j], 2]),
            "active_lidar": bool(row_active[0]),
            "active_ultrasonic": bool(row_active[1]),
            "active_radar": bool(row_active[2]),
            "active_sensor_count": int(np.sum(row_active)),

            # Baselines
            "single_sensor_est": single_estimate,
            "single_abs_error": single_error,
            "best_single_sensor_est": best_estimate,
            "best_single_sensor_name": best_name,
            "best_single_abs_error": best_error,

            # Fixed fusion
            "fixed_fusion_est": fixed_map_estimate,
            "fixed_mean_est": fixed_mean_estimate,
            "fixed_abs_error": fixed_map_error,
            "fixed_mean_abs_error": fixed_mean_error,
            "fixed_confidence": float(fixed_metrics["confidence"]) if fixed_metrics else np.nan,
            "fixed_entropy": float(fixed_metrics["entropy"]) if fixed_metrics else np.nan,
            "fixed_variance": float(fixed_metrics["variance"]) if fixed_metrics else np.nan,
            "fixed_posterior_peak": float(fixed_metrics[
                "posterior_peak"]) if fixed_metrics else np.nan,
            "fixed_ci_lower_95": float(fixed_metrics["ci_lower"]) if fixed_metrics else np.nan,
            "fixed_ci_upper_95": float(fixed_metrics["ci_upper"]) if fixed_metrics else np.nan,
            "fixed_ci_width_95": (float(fixed_metrics.get("ci_width_95", fixed_metrics["ci_upper"] - fixed_metrics["ci_lower"])) if fixed_metrics else np.nan),
            "fixed_ci_coverage_95": fixed_scoring["coverage_95"],
            "fixed_nll": fixed_scoring["nll"],
            "fixed_crps": fixed_scoring["crps"],

            # Adaptive fusion
            "adaptive_fusion_est": adaptive_map_estimate,
            "adaptive_depth_mean": adaptive_mean_estimate,
            "adaptive_abs_error": adaptive_map_error,
            "adaptive_mean_abs_error": adaptive_mean_error,
            "adaptive_confidence": adaptive_values["confidence"],
            "adaptive_entropy": adaptive_values["entropy"],
            "adaptive_variance": adaptive_values["variance"],
            "adaptive_posterior_peak": adaptive_values["posterior_peak"],
            "adaptive_ci_lower_95": adaptive_values["ci_lower"],
            "adaptive_ci_upper_95": adaptive_values["ci_upper"],
            "adaptive_ci_width_95": adaptive_values["ci_width_95"],
            "adaptive_ci_coverage_95": adaptive_scoring["coverage_95"],
            "adaptive_nll": adaptive_scoring["nll"],
            "adaptive_crps": adaptive_scoring["crps"],
            "entropy_score": adaptive_values["entropy_score"],
            "peak_score": adaptive_values["peak_score"],
            "interval_score": adaptive_values["interval_score"],
            "variance_score": adaptive_values["variance_score"],
            "effective_support": adaptive_values["effective_support"],
            "support_fraction": adaptive_values["support_fraction"],
            "gap_score": adaptive_values["gap_score"],

            # Adaptive diagnostics
            "adaptive_w_lidar": float(adaptive["weights"]["lidar"]) if adaptive_values["confidence"] == adaptive_values["confidence"] and adaptive.get("weights") else np.nan,
            "adaptive_w_ultra": float(adaptive["weights"]["ultrasonic"]) if adaptive_values["confidence"] == adaptive_values["confidence"] and adaptive.get("weights") else np.nan,
            "adaptive_w_radar": float(adaptive["weights"]["radar"]) if adaptive_values["confidence"] == adaptive_values["confidence"] and adaptive.get("weights") else np.nan,
            "adaptive_sigma_lidar": float(adaptive["sigmas"]["lidar"]) if adaptive_values["confidence"] == adaptive_values["confidence"] and adaptive.get("sigmas") else np.nan,
            "adaptive_sigma_ultra": float(adaptive["sigmas"]["ultrasonic"]) if adaptive_values["confidence"] == adaptive_values["confidence"] and adaptive.get("sigmas") else np.nan,
            "adaptive_sigma_radar": float(adaptive["sigmas"]["radar"]) if adaptive_values["confidence"] == adaptive_values["confidence"] and adaptive.get("sigmas") else np.nan,
            "context_difficulty": _safe_float(adaptive_diag.get("context_difficulty"), np.nan),
            "adaptation_gate": _safe_float(adaptive_diag.get("adaptation_gate"), np.nan),
            "effective_sensor_count": _safe_float(
                adaptive_diag.get("effective_sensor_count"), np.nan),
            "dominance_ratio": _safe_float(adaptive_diag.get("dominance_ratio"), np.nan),
            "weight_entropy": _safe_float(adaptive_diag.get("weight_entropy"), np.nan),
            "measurement_consistency": _safe_float(
                adaptive_diag.get("measurement_consistency_mean"), np.nan),

            # Hazard output is downstream of the posterior only
            "hazard_probability": float(hazard["hazard_probability"]),
            "hazard_status": str(hazard["status"]),
            "risk_score": float(hazard["risk_score"]),
            "hazard_expected_depth": float(hazard["expected_hazard_depth"]),
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if df_out.empty: raise RuntimeError("No evaluation rows were produced")
    ok_adaptive = df_out["adaptive_fusion_est"].notna()

    if ok_adaptive.any():
        weight_cols = ["adaptive_w_lidar", "adaptive_w_ultra", "adaptive_w_radar"]
        weight_sum = df_out.loc[ok_adaptive, weight_cols].sum(axis=1).to_numpy(dtype=float)

        if not np.all(np.isfinite(weight_sum)) or not np.allclose(weight_sum, 1.0, atol=1e-6):
            raise RuntimeError("Adaptive fusion weights failed normalization integrity check")
            
        sigma_cols = ["adaptive_sigma_lidar", "adaptive_sigma_ultra", "adaptive_sigma_radar"]
        sigma_values = df_out.loc[ok_adaptive, sigma_cols].to_numpy(dtype=float)

        if not np.isfinite(sigma_values).all():
            raise RuntimeError("Adaptive fusion produced non-finite sigma values")
        if np.any(sigma_values < 0.45 - 1e-9) or np.any(sigma_values > 2.85 + 1e-9):
            raise RuntimeError("Adaptive sigma output exceeded the frozen engine bounds")

    # Evaluation-only diagnostics
    corr_conf_mae = _safe_corr(df_out["adaptive_confidence"], df_out["adaptive_abs_error"])
    corr_entropy_mae = _safe_corr(df_out["adaptive_entropy"], df_out["adaptive_abs_error"])
    corr_ciwidth_mae = _safe_corr(df_out["adaptive_ci_width_95"], df_out["adaptive_abs_error"])
    corr_variance_mae = _safe_corr(df_out["adaptive_variance"], df_out["adaptive_abs_error"])
    corr_hazard_depth = _safe_corr(df_out["hazard_probability"], df_out["true_depth"])
    corr_hazard_mae = _safe_corr(df_out["hazard_probability"], df_out["adaptive_abs_error"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(DATASET_FILE_OUT, index=False)

    rmse_single = _rmse(df_out["single_abs_error"])
    rmse_best_single = _rmse(df_out["best_single_abs_error"])
    rmse_fixed_map = _rmse(df_out["fixed_abs_error"])
    rmse_adaptive_map = _rmse(df_out["adaptive_abs_error"])
    rmse_fixed_mean = _rmse(df_out["fixed_mean_abs_error"])
    rmse_adaptive_mean = _rmse(df_out["adaptive_mean_abs_error"])

    rmse_gain_map = ((rmse_fixed_map - rmse_adaptive_map) / rmse_fixed_map * 100.0
        if np.isfinite(rmse_fixed_map) and rmse_fixed_map > 1e-12 and np.isfinite(rmse_adaptive_map)
        else float("nan"))
        
    rmse_gain_mean = ((rmse_fixed_mean - rmse_adaptive_mean) / rmse_fixed_mean * 100.0
        if np.isfinite(rmse_fixed_mean) and rmse_fixed_mean > 1e-12 and np.isfinite(rmse_adaptive_mean) else float("nan"))

    paired_map = df_out[["adaptive_abs_error", "fixed_abs_error"]].dropna()
    paired_mean = df_out[["adaptive_mean_abs_error", "fixed_mean_abs_error"]].dropna()
    
    adaptive_win_map = (float(np.mean(paired_map["adaptive_abs_error"] < paired_map["fixed_abs_error"])) if len(paired_map) else float("nan"))        
    adaptive_win_mean = (float(np.mean(paired_mean["adaptive_mean_abs_error"] < paired_mean["fixed_mean_abs_error"])) if len(paired_mean) else float("nan"))

    map_diff, map_low, map_high = _bootstrap_mean_difference(
        paired_map["adaptive_abs_error"].to_numpy(dtype=float),
        paired_map["fixed_abs_error"].to_numpy(dtype=float))
        
    mean_diff, mean_low, mean_high = _bootstrap_mean_difference(
        paired_mean["adaptive_mean_abs_error"].to_numpy(dtype=float),
        paired_mean["fixed_mean_abs_error"].to_numpy(dtype=float))

    best_counts = df_out["best_single_sensor_name"].value_counts(dropna=False)
    hazard_counts = df_out["hazard_status"].value_counts(dropna=False)
    total_eval = len(df_out)

    hazard_probability = df_out["hazard_probability"].to_numpy(dtype=float)
    true_hazard = df_out["true_hazard"].to_numpy(dtype=float)
    hazard_mask = np.isfinite(hazard_probability) & np.isfinite(true_hazard)
    
    hazard_brier = (float(np.mean((hazard_probability[hazard_mask] - true_hazard[hazard_mask]) ** 2))
        if hazard_mask.any() else float("nan"))
        
    predicted_hazard = hazard_probability[hazard_mask] >= 0.80    
    hazard_accuracy = (float(np.mean(predicted_hazard == true_hazard[hazard_mask]))
        if hazard_mask.any() else float("nan"))

    summary = {
        "samples_total": len(df),
        "calibration_samples": int(calibration_mask.sum()),
        "evaluation_samples_available": int((~calibration_mask).sum()),
        "evaluation_samples": int(len(df_out)),
        "calibration_fraction": float(np.mean(calibration_mask)),
        "fixed_sigma_lidar": fixed_sigmas[0],
        "fixed_sigma_ultrasonic": fixed_sigmas[1],
        "fixed_sigma_radar": fixed_sigmas[2], **calibration_diag,

        "rmse_single_sensor": rmse_single,
        "rmse_best_single_sensor_oracle": rmse_best_single,
        "rmse_fixed_fusion_map": rmse_fixed_map,
        "rmse_adaptive_fusion_map": rmse_adaptive_map,
        "rmse_fixed_fusion_mean": rmse_fixed_mean,
        "rmse_adaptive_fusion_mean": rmse_adaptive_mean,
        "rmse_gain_map_percent": rmse_gain_map,
        "rmse_gain_mean_percent": rmse_gain_mean,

        "avg_single_mae": _nanmean(df_out["single_abs_error"]),
        "avg_best_single_mae_oracle": _nanmean(df_out["best_single_abs_error"]),
        "avg_fixed_mae": _nanmean(df_out["fixed_abs_error"]),
        "avg_adaptive_mae": _nanmean(df_out["adaptive_abs_error"]),
        "avg_fixed_mean_mae": _nanmean(df_out["fixed_mean_abs_error"]),
        "avg_adaptive_mean_mae": _nanmean(df_out["adaptive_mean_abs_error"]),
        "adaptive_win_rate_map": adaptive_win_map,
        "adaptive_win_rate_mean": adaptive_win_mean,
        "paired_mean_abs_error_difference_map": map_diff,
        "paired_mean_abs_error_difference_map_ci95_low": map_low,
        "paired_mean_abs_error_difference_map_ci95_high": map_high,
        "paired_mean_abs_error_difference_mean": mean_diff,
        "paired_mean_abs_error_difference_mean_ci95_low": mean_low,
        "paired_mean_abs_error_difference_mean_ci95_high": mean_high,

        "avg_fixed_confidence": _nanmean(df_out["fixed_confidence"]),
        "avg_adaptive_confidence": _nanmean(df_out["adaptive_confidence"]),
        "std_adaptive_confidence": _nanstd(df_out["adaptive_confidence"]),
        "avg_fixed_entropy": _nanmean(df_out["fixed_entropy"]),
        "avg_adaptive_entropy": _nanmean(df_out["adaptive_entropy"]),
        "std_adaptive_entropy": _nanstd(df_out["adaptive_entropy"]),
        "avg_fixed_variance": _nanmean(df_out["fixed_variance"]),
        "avg_adaptive_variance": _nanmean(df_out["adaptive_variance"]),
        "avg_fixed_ci_width_95": _nanmean(df_out["fixed_ci_width_95"]),
        "avg_adaptive_ci_width_95": _nanmean(df_out["adaptive_ci_width_95"]),
        "fixed_ci_coverage_95": _nanmean(df_out["fixed_ci_coverage_95"]),
        "adaptive_ci_coverage_95": _nanmean(df_out["adaptive_ci_coverage_95"]),
        "avg_fixed_nll": _nanmean(df_out["fixed_nll"]),
        "avg_adaptive_nll": _nanmean(df_out["adaptive_nll"]),
        "avg_fixed_crps": _nanmean(df_out["fixed_crps"]),
        "avg_adaptive_crps": _nanmean(df_out["adaptive_crps"]),
        "avg_fixed_posterior_peak": _nanmean(df_out["fixed_posterior_peak"]),
        "avg_adaptive_posterior_peak": _nanmean(df_out["adaptive_posterior_peak"]),

        "avg_adaptive_w_lidar": _nanmean(df_out["adaptive_w_lidar"]),
        "avg_adaptive_w_ultra": _nanmean(df_out["adaptive_w_ultra"]),
        "avg_adaptive_w_radar": _nanmean(df_out["adaptive_w_radar"]),
        "avg_adaptive_sigma_lidar": _nanmean(df_out["adaptive_sigma_lidar"]),
        "avg_adaptive_sigma_ultra": _nanmean(df_out["adaptive_sigma_ultra"]),
        "avg_adaptive_sigma_radar": _nanmean(df_out["adaptive_sigma_radar"]),

        "corr_confidence_mae": corr_conf_mae,
        "corr_entropy_mae": corr_entropy_mae,
        "corr_ciwidth_mae": corr_ciwidth_mae,
        "corr_variance_mae": corr_variance_mae,
        "corr_hazard_depth": corr_hazard_depth,
        "corr_hazard_mae": corr_hazard_mae,

        "avg_hazard_probability": _nanmean(df_out["hazard_probability"]),
        "hazard_brier_score": hazard_brier,
        "hazard_accuracy_at_0.80": hazard_accuracy,
        "hazards_detected": int(hazard_counts.get("HAZARD", 0)),
        "hazard_rate_percent": 100.0 * hazard_counts.get("HAZARD", 0) / total_eval,
        "caution_flags": int(hazard_counts.get("CAUTION", 0)),
        "caution_rate_percent": 100.0 * hazard_counts.get("CAUTION", 0) / total_eval,
        "safe_cases": int(hazard_counts.get("SAFE", 0)),
        "safe_rate_percent": 100.0 * hazard_counts.get("SAFE", 0) / total_eval,

        "mean_entropy_score": _nanmean(df_out["entropy_score"]),
        "std_entropy_score": _nanstd(df_out["entropy_score"]),
        "mean_peak_score": _nanmean(df_out["peak_score"]),
        "std_peak_score": _nanstd(df_out["peak_score"]),
        "mean_interval_score": _nanmean(df_out["interval_score"]),
        "std_interval_score": _nanstd(df_out["interval_score"]),
        "mean_variance_score": _nanmean(df_out["variance_score"]),
        "std_variance_score": _nanstd(df_out["variance_score"]),

        "avg_true_depth": _nanmean(df_out["true_depth"]),
        "avg_scene_complexity": _nanmean(df_out["scene_complexity"]),
        "avg_sensor_agreement": _nanmean(df_out["sensor_agreement"]),
        "avg_measurement_spread": _nanmean(df_out[
            "measurement_spread"]) if "measurement_spread" in df_out else float("nan"),
        "avg_reliability_spread": _nanmean(df_out["reliability_spread"]),
        "avg_water_context_proxy": _nanmean(df_out["water_context_depth_proxy"]),

        "best_single_lidar_count": int(best_counts.get("lidar", 0)),
        "best_single_ultrasonic_count": int(best_counts.get("ultrasonic", 0)),
        "best_single_radar_count": int(best_counts.get("radar", 0)),

        "avg_context_difficulty": _nanmean(df_out["context_difficulty"]),
        "avg_adaptation_gate": _nanmean(df_out["adaptation_gate"]),
        "avg_effective_sensor_count": _nanmean(df_out["effective_sensor_count"]),
        "avg_dominance_ratio": _nanmean(df_out["dominance_ratio"]),
        "avg_weight_entropy": _nanmean(df_out["weight_entropy"]),
        "avg_measurement_consistency": _nanmean(df_out["measurement_consistency"]),

        "adaptive_failures_evaluation": int(df_out["adaptive_fusion_est"].isna().sum()),
        "fixed_failures_evaluation": int(df_out["fixed_fusion_est"].isna().sum()),
        "adaptive_failures_total": int(failure_count_adaptive),
        "fixed_failures_total": int(failure_count_fixed),
        "observable_water_context_available_rate": float(np.mean(eval_water_available)),
        "observable_ntu_available_rate": float(np.mean(eval_ntu_available)),
    }
    summary_df = pd.DataFrame([summary]).round(6)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print("=" * 60)
    print("FloodTwin-HIL Fusion Validation")
    print("=" * 60)
    print()
    print(f"Total Scenarios       : {len(df)}")
    print(f"Calibration Scenarios : {int(calibration_mask.sum())}")
    print(f"Evaluation Scenarios  : {len(df_out)}")
    print(f"Results Saved         : {DATASET_FILE_OUT}")
    print(f"Summary Saved         : {SUMMARY_FILE}")
    print()
    print(summary_df.T.to_string(header=False))
    print()
    print("Hazard State Distribution:")
    print(df_out["hazard_status"].value_counts())
    print()
    print("Best Single Sensor Distribution:")
    print(df_out["best_single_sensor_name"].value_counts())
    print()
    print("=" * 60)

    return df_out, summary_df

if __name__ == "__main__":
    run_fusion_pipeline()