import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Repository Path
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path: sys.path.insert(0, str(BASE_DIR))

from fusion_engine.bayesian_fusion import BayesianFusionCore
from fusion_engine.fixed_fusion import FixedFusionEngine
from fusion_engine.adaptive_fusion import AdaptiveFusionEngine
from fusion_engine.uncertainty import UncertaintyQuantification
from fusion_engine.hazard_assessment import HazardAssessment

from benchmarking.ablation_study import AblationFramework
from benchmarking.statistical_analysis import StatisticalEvaluator
from benchmarking.validation_confidence import ValidationEvidenceScorer

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATASET_FILE_IN = BASE_DIR / "datasets" / "dataset_v2.csv"
DEFAULT_FAULT_RESULTS_FILE = BASE_DIR / "results" / "fault_injection" / "robustness_results.csv"
RESULTS_DIR = BASE_DIR / "results" / "benchmarking"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"
ABLATION_RESULTS_FILE = RESULTS_DIR / "ablation_results.csv"
VALIDATION_SCORES_FILE = RESULTS_DIR / "validation_scores.csv"
STATISTICAL_SUMMARY_FILE = RESULTS_DIR / "statistical_summary.csv"
BENCHMARK_SUMMARY_FILE = RESULTS_DIR / "benchmark_summary.csv"
CAMPAIGN_METADATA_FILE = RESULTS_DIR / "campaign_metadata.json"

DEPTH_MIN_CM = 0.0
DEPTH_MAX_CM = 30.0
DEPTH_RESOLUTION = 300
DEFAULT_RANDOM_SEED = 101
DEFAULT_ABLATION_LIMIT = 5000
CALIBRATED_BASE_SIGMAS_CM: Tuple[float, float, float] = (0.85, 0.75, 0.60)

SENSOR_NAMES = ("lidar", "ultrasonic", "radar")
SENSOR_MEASUREMENT_COLUMNS = {"lidar": "lidar", "ultrasonic": "ultrasonic", "radar": "radar"}
SENSOR_RELIABILITY_COLUMNS = {"lidar": "R_lidar", "ultrasonic": "R_ultrasonic", "radar": "R_radar"}

# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------
@dataclass(frozen=True)
class CampaignConfig:

    dataset_path: Path = DATASET_FILE_IN
    results_dir: Path = RESULTS_DIR
    benchmark_limit: Optional[int] = None
    ablation_limit: Optional[int] = DEFAULT_ABLATION_LIMIT
    random_seed: int = DEFAULT_RANDOM_SEED
    fixed_sigmas_cm: Tuple[float, float, float] = CALIBRATED_BASE_SIGMAS_CM
    hazard_threshold_cm: float = 10.0
    hazard_decision_boundary: float = 0.80
    statistical_alpha: float = 0.05
    bootstrap_samples: int = 5000
    validation_target_samples: int = 30000
    fault_results_path: Optional[Path] = DEFAULT_FAULT_RESULTS_FILE

@dataclass
class CampaignArtifacts:

    benchmark_df: pd.DataFrame
    ablation_df: pd.DataFrame
    statistical_df: pd.DataFrame
    validation_df: pd.DataFrame
    summary_df: pd.DataFrame

# ------------------------------------------------------------------
# Numerical Utilities
# ------------------------------------------------------------------
def _safe_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)

def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = _safe_array(values)
    return arr[np.isfinite(arr)]

def _safe_mean(values: Iterable[Any]) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")

def _safe_std(values: Iterable[Any], ddof: int = 1) -> float:
    arr = _finite(values)
    if arr.size <= ddof: return float("nan")
    return float(np.std(arr, ddof=ddof))

def _safe_rmse(values: Iterable[Any]) -> float:
    arr = _finite(values)
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else float("nan")

def _safe_mae(values: Iterable[Any]) -> float:
    arr = np.abs(_finite(values))
    return float(np.mean(arr)) if arr.size else float("nan")

def _safe_rate(flags: Iterable[Any]) -> float:
    arr = np.asarray(flags, dtype=bool).reshape(-1)
    return float(np.mean(arr) * 100.0) if arr.size else float("nan")

def _safe_fraction(flags: Iterable[Any]) -> float:
    arr = np.asarray(flags, dtype=bool).reshape(-1)
    return float(np.mean(arr)) if arr.size else float("nan")

def _safe_corr(x: Iterable[Any], y: Iterable[Any]) -> float:
    xx, yy = _safe_array(x), _safe_array(y)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if mask.sum() < 2: return float("nan")
    x_valid, y_valid = xx[mask], yy[mask]
    if np.std(x_valid) <= 1e-12 or np.std(y_valid) <= 1e-12: return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])

def _quantile_safe(values: Iterable[Any], q: float) -> float:
    arr = _finite(values)
    return float(np.quantile(arr, q)) if arr.size else float("nan")

def _normalize_probability(value: Any) -> float:
    try: x = float(value)
    except (TypeError, ValueError): return float("nan")
    if not np.isfinite(x): return float("nan")
    if x > 1.0: x /= 100.0
    return float(np.clip(x, 0.0, 1.0))

def _clip01(value: Any, default: float = 0.0) -> float:
    try: x = float(value)
    except (TypeError, ValueError): x = float(default)
    if not np.isfinite(x): x = float(default)
    return float(np.clip(x, 0.0, 1.0))

def _depth_axis(core: BayesianFusionCore) -> np.ndarray:
    if hasattr(core, "depths"):
        depths = np.asarray(core.depths, dtype=float).reshape(-1)
        if depths.size >= 2 and np.all(np.isfinite(depths)): return depths
    return np.linspace(DEPTH_MIN_CM, DEPTH_MAX_CM, DEPTH_RESOLUTION, dtype=float)

def _extract_posterior(result: Any) -> np.ndarray:
    if isinstance(result, dict) and "posterior" in result:
        posterior = np.asarray(result["posterior"], dtype=float).reshape(-1)
    else:
        posterior = np.asarray(result, dtype=float).reshape(-1)
    if posterior.size == 0: raise ValueError("Fusion result contains an empty posterior")
    return posterior

def _posterior_grid_metrics(
    uq: UncertaintyQuantification, depth_axis: np.ndarray,
    posterior: np.ndarray, delta_x: float) -> Dict[str, float]:

    metrics = uq.compute(depth_axis, posterior, delta_x)
    return {
        "map_depth": float(metrics.get("map_depth", np.nan)),
        "expected_depth": float(metrics.get("expected_depth", np.nan)),
        "variance": float(metrics.get("variance", np.nan)),
        "std": float(metrics.get("std", np.nan)),
        "entropy": float(metrics.get("entropy", np.nan)),
        "confidence": float(metrics.get("confidence", np.nan)),
        "posterior_peak": float(metrics.get("posterior_peak", np.nan)),
        "ci_lower": float(metrics.get("ci_lower", np.nan)),
        "ci_upper": float(metrics.get("ci_upper", np.nan)),
        "ci_width": float(metrics.get("ci_width_95", metrics.get("ci_upper", np.nan) - metrics.get("ci_lower", np.nan))),
    }

def _posterior_cdf_at(depth_axis: np.ndarray, posterior: np.ndarray, x: float) -> float:
    p = np.asarray(posterior, dtype=float).reshape(-1)
    xgrid = np.asarray(depth_axis, dtype=float).reshape(-1)
    if p.size != xgrid.size or p.size == 0 or not np.isfinite(x): return float("nan")

    p = np.clip(p, 0.0, None)
    total = float(np.sum(p))
    if total <= 1e-12: return float("nan")

    cdf = np.cumsum(p) / total
    idx = np.searchsorted(xgrid, float(x), side="right") - 1
    if idx < 0: return 0.0
    if idx >= cdf.size - 1: return 1.0
    return float(np.clip(cdf[idx], 0.0, 1.0))

def _posterior_interval_probability(
    depth_axis: np.ndarray, posterior: np.ndarray,
    lower: float, upper: float) -> float:

    p = np.asarray(posterior, dtype=float).reshape(-1)
    xgrid = np.asarray(depth_axis, dtype=float).reshape(-1)
    if p.size != xgrid.size: return float("nan")

    p = np.clip(p, 0.0, None)
    total = float(np.sum(p))
    if total <= 1e-12: return float("nan")

    mask = (xgrid >= float(lower)) & (xgrid <= float(upper))
    return float(np.sum(p[mask]) / total)

def _posterior_nll_approx(
    depth_axis: np.ndarray, posterior: np.ndarray, true_depth: float) -> float:

    p = np.asarray(posterior, dtype=float).reshape(-1)
    xgrid = np.asarray(depth_axis, dtype=float).reshape(-1)
    if p.size != xgrid.size or not np.isfinite(true_depth): return float("nan")

    p = np.clip(p, 0.0, None)
    total = float(np.sum(p))
    if total <= 1e-12: return float("nan")

    idx = int(np.argmin(np.abs(xgrid - float(true_depth))))
    mass = max(float(p[idx] / total), 1e-12)
    return float(-math.log(mass))

def _posterior_crps_approx(
    depth_axis: np.ndarray, posterior: np.ndarray, true_depth: float) -> float:

    p = np.asarray(posterior, dtype=float).reshape(-1)
    xgrid = np.asarray(depth_axis, dtype=float).reshape(-1)
    if p.size != xgrid.size or p.size < 2 or not np.isfinite(true_depth): return float("nan")

    p = np.clip(p, 0.0, None)
    total = float(np.sum(p))
    if total <= 1e-12: return float("nan")

    cdf = np.cumsum(p) / total
    indicator = (xgrid >= float(true_depth)).astype(float)
    integrand = np.square(cdf - indicator)

    try: crps = float(np.trapezoid(integrand, xgrid))
    except AttributeError: crps = float(np.trapz(integrand, xgrid))
    return max(crps, 0.0)

# ------------------------------------------------------------------
# Data Handling / Observable Context
# ------------------------------------------------------------------
def _coalesce_numeric(
    df: pd.DataFrame, candidates: Sequence[str], default: float = np.nan) -> pd.Series:
    for col in candidates:
        if col in df.columns: return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)

def _coalesce_bool(
    df: pd.DataFrame, candidates: Sequence[str], default: bool = False) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            series = df[col]
            if series.dtype == bool: return series.fillna(default).astype(bool)
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any(): return numeric.fillna(float(default)).astype(bool)
            text = series.astype(str).str.strip().str.lower()
            return text.isin({"true", "1", "yes", "available", "active"})
    return pd.Series(default, index=df.index, dtype=bool)

def _observable_water_context(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    proxy = _coalesce_numeric(df, ("water_context_depth_proxy", "observable_water_depth",
        "water_depth_proxy", "water_proxy"), default=np.nan).clip(lower=0.0)
    available = proxy.notna()
    contact = _coalesce_bool(df, ("water_contact", "water_present",
        "water_state_available"), default=False)
    proxy = proxy.where(available, contact.astype(float) * 1.0)
    return proxy.astype(float), (available | contact).astype(bool)

def _observable_ntu(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    ntu = _coalesce_numeric(df, ("ntu_observed", "observed_ntu", "ntu_sensor",
        "ntu_proxy", "ntu", "ntu_measurement"), default=np.nan).clip(lower=0.0)
    return ntu, ntu.notna()

def _sensor_measurements(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for sensor, col in SENSOR_MEASUREMENT_COLUMNS.items():
        out[sensor] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(
            np.nan, index=df.index)
    return out

def _sensor_reliabilities(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for sensor, col in SENSOR_RELIABILITY_COLUMNS.items():
        out[sensor] = (pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)
            if col in df.columns else pd.Series(0.0, index=df.index))
    return out

def _sensor_active_masks(
    df: pd.DataFrame, measurements: Mapping[str, pd.Series],
    reliabilities: Mapping[str, pd.Series]) -> Dict[str, pd.Series]:

    out: Dict[str, pd.Series] = {}
    for sensor in SENSOR_NAMES:
        explicit = _coalesce_bool(df, (f"{sensor}_active", f"{sensor}_available",
            f"active_{sensor}", f"available_{sensor}"), default=True)
        measurement_valid = measurements[sensor].notna()
        reliability_valid = reliabilities[sensor].fillna(0.0) > 0.0
        out[sensor] = (explicit & measurement_valid & reliability_valid).astype(bool)
    return out

def _sensor_health(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for sensor in SENSOR_NAMES:
        health = _coalesce_numeric(df, (f"{sensor}_health",
            f"health_{sensor}", f"H_{sensor}"), default=1.0).clip(0.0, 1.0)
        out[sensor] = health
    return out

def _observable_context_arrays(
    df: pd.DataFrame, measurements: Mapping[str, pd.Series], reliabilities: Mapping[str, pd.Series],
    active_masks: Mapping[str, pd.Series]) -> Dict[str, np.ndarray]:
    n = len(df)
    z = np.column_stack([
        measurements["lidar"].to_numpy(dtype=float),
        measurements["ultrasonic"].to_numpy(dtype=float),
        measurements["radar"].to_numpy(dtype=float)])

    r = np.column_stack([
        reliabilities["lidar"].fillna(0.0).to_numpy(dtype=float),
        reliabilities["ultrasonic"].fillna(0.0).to_numpy(dtype=float),
        reliabilities["radar"].fillna(0.0).to_numpy(dtype=float)])

    active = np.column_stack([
        active_masks["lidar"].to_numpy(dtype=bool),
        active_masks["ultrasonic"].to_numpy(dtype=bool),
        active_masks["radar"].to_numpy(dtype=bool)])

    measurement_spread = np.zeros(n, dtype=float)
    agreement = np.zeros(n, dtype=float)
    reliability_spread = np.zeros(n, dtype=float)
    effective_count = active.sum(axis=1).astype(float)
    dominance_ratio = np.ones(n, dtype=float)

    for i in range(n):
        values = z[i, active[i] & np.isfinite(z[i])]
        if values.size >= 2:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            measurement_spread[i] = float(1.4826 * mad)
            pairwise = []

            for a in range(values.size):
                for b in range(a + 1, values.size):
                    pairwise.append(abs(float(values[a] - values[b])))
            pairwise_mean = float(np.mean(pairwise)) if pairwise else 0.0
            mean_abs = float(np.mean(np.abs(values))) + 1e-6
            agreement[i] = float(np.clip(1.0 / (1.0 + pairwise_mean / mean_abs), 0.0, 1.0))
        elif values.size == 1: agreement[i] = 1.0

        rel_values = r[i, active[i]]
        if rel_values.size >= 2:
            reliability_spread[i] = float(np.clip(np.max(rel_values) - np.min(rel_values), 0.0, 1.0))
            rel_sum = float(np.sum(rel_values))
            if rel_sum > 1e-12:
                dominance_ratio[i] = float(np.clip(np.max(rel_values) / rel_sum, 0.0, 1.0))
        elif rel_values.size == 1: dominance_ratio[i] = 1.0

    water_proxy, water_available = _observable_water_context(df)
    ntu_observed, ntu_available = _observable_ntu(df)

    water_arr = water_proxy.to_numpy(dtype=float)
    ntu_arr = ntu_observed.to_numpy(dtype=float)

    water_term = np.nan_to_num(water_arr / 20.0, nan=0.0)
    ntu_term = np.nan_to_num(ntu_arr / 500.0, nan=0.0)
    spread_term = np.clip(np.nan_to_num(measurement_spread, nan=0.0) / 5.0, 0.0, 1.0)

    scene_complexity = np.clip(0.35 * water_term + 0.35 * ntu_term + 0.30 * spread_term, 0.0, 1.0)
    confidence_proxy = np.full(n, 0.5, dtype=float)

    return {
        "scene_complexity": scene_complexity,
        "sensor_agreement": agreement,
        "reliability_spread": reliability_spread,
        "effective_sensor_count": effective_count,
        "dominance_ratio": dominance_ratio,
        "confidence_proxy": confidence_proxy,
        "measurement_spread": measurement_spread,
        "water_context_depth_proxy": np.nan_to_num(water_arr, nan=0.0),
        "water_context_available": water_available.to_numpy(dtype=bool),
        "ntu_observed": np.nan_to_num(ntu_arr, nan=0.0),
        "ntu_available": ntu_available.to_numpy(dtype=bool),
        "active_lidar": active[:, 0],
        "active_ultrasonic": active[:, 1],
        "active_radar": active[:, 2],
    }

# ------------------------------------------------------------------
# Sampling / Benchmark Split Helpers
# ------------------------------------------------------------------
def _stable_stratified_sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n >= len(df): return df.copy().reset_index(drop=True)
    n = max(int(n), 1)
    category_col = next((col for col in ("benchmark_category", "fault_scenario", "scenario_category", "category") if col in df.columns), None)

    if category_col is None: return df.sample(n=n, random_state=seed).reset_index(drop=True)
    groups = df.groupby(df[category_col].astype(str), dropna=False, sort=True)
    rng = np.random.default_rng(seed)
    selected: List[int] = []

    sizes = {str(k): len(v) for k, v in groups}
    total = sum(sizes.values())
    if total <= 0: return df.sample(n=n, random_state=seed).reset_index(drop=True)

    allocations: Dict[str, int] = {}
    fractional: List[Tuple[float, str]] = []

    for key, size in sizes.items():
        raw = n * size / total
        base = int(math.floor(raw))
        allocations[key] = min(base, size)
        fractional.append((raw - base, key))

    remaining = n - sum(allocations.values())
    fractional.sort(reverse=True)

    for _, key in fractional:
        if remaining <= 0: break
        if allocations[key] < sizes[key]:
            allocations[key] += 1
            remaining -= 1

    for key, group in groups:
        key_str = str(key)
        count = allocations.get(key_str, 0)
        if count <= 0: continue

        local_seed = int(rng.integers(0, 2**32 - 1))
        sampled = group.sample(n=count, random_state=local_seed)
        selected.extend(sampled.index.tolist())

    if len(selected) < n:
        missing = n - len(selected)
        remaining_pool = df.loc[~df.index.isin(selected)]
        if len(remaining_pool) > 0:
            extra = remaining_pool.sample(
                n=min(missing, len(remaining_pool)),
                random_state=int(rng.integers(0, 2**32 - 1)))
            selected.extend(extra.index.tolist())
    return df.loc[selected].sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ------------------------------------------------------------------
# Fusion / Evaluation
# ------------------------------------------------------------------
def _fixed_posterior_with_mask(
    core: BayesianFusionCore, measurements: Sequence[float],
    active_mask: Sequence[bool], sigmas: Sequence[float]) -> Tuple[np.ndarray, Dict[str, float]]:

    z = np.asarray(measurements, dtype=float).reshape(-1)
    active = np.asarray(active_mask, dtype=bool)
    sigma = np.asarray(sigmas, dtype=float).reshape(-1)

    if z.shape != (3,) or active.shape != (3,) or sigma.shape != (3,):
        raise ValueError("Fixed pooling expects three sensors")

    likelihoods: List[np.ndarray] = []
    names: List[str] = []

    for idx, name in enumerate(SENSOR_NAMES):
        if not active[idx]: continue
        value = float(z[idx])
        if not np.isfinite(value): continue
        likelihoods.append(core.compute_likelihood(value, float(sigma[idx])))
        names.append(name)

    if not likelihoods: raise RuntimeError("No active finite sensor is available for fixed fusion")
    weights = [1.0] * len(likelihoods)
    pooled = core.combine_likelihoods(likelihoods, weights)
    posterior = core.compute_posterior(pooled)

    equal_weight_value = 1.0 / len(likelihoods)
    weight_dict = {name: 0.0 for name in SENSOR_NAMES}
    for name in names: weight_dict[name] = equal_weight_value
    return np.asarray(posterior, dtype=float), weight_dict

def _evaluate_single_row(
    row: pd.Series, core: BayesianFusionCore, fixed_engine: FixedFusionEngine,
    adaptive_engine: AdaptiveFusionEngine, uq: UncertaintyQuantification,
    hazard_assessor: HazardAssessment, fixed_sigmas: Tuple[float, float, float],
    context_row: Mapping[str, Any]) -> Dict[str, Any]:

    true_depth = float(row["true_depth"])
    depth_axis = _depth_axis(core)
    delta_x = float(getattr(core, "delta_x", depth_axis[1] - depth_axis[0]))

    measurements = np.array([
        float(row["lidar"]) if np.isfinite(row["lidar"]) else np.nan,
        float(row["ultrasonic"]) if np.isfinite(row["ultrasonic"]) else np.nan,
        float(row["radar"]) if np.isfinite(row["radar"]) else np.nan,
    ], dtype=float)

    reliabilities = np.array([
        float(row["R_lidar"]),
        float(row["R_ultrasonic"]),
        float(row["R_radar"]),
    ], dtype=float)

    active_mask = np.array([
        bool(context_row["active_lidar"]),
        bool(context_row["active_ultrasonic"]),
        bool(context_row["active_radar"]),
    ], dtype=bool)

    sensor_health = {
        "lidar": float(context_row["health_lidar"]),
        "ultrasonic": float(context_row["health_ultrasonic"]),
        "radar": float(context_row["health_radar"]),
    }

    scene_complexity = float(context_row["scene_complexity"])
    sensor_agreement = float(context_row["sensor_agreement"])
    reliability_spread = float(context_row["reliability_spread"])
    effective_sensor_count = float(context_row["effective_sensor_count"])
    dominance_ratio = float(context_row["dominance_ratio"])
    measurement_spread = float(context_row["measurement_spread"])
    water_context_proxy = float(context_row["water_context_depth_proxy"])
    ntu_observed = float(context_row["ntu_observed"])

    try:
        if np.all(active_mask):
            fixed_result = fixed_engine.estimate(
                float(measurements[0]), float(measurements[1]), float(measurements[2]), sigmas=fixed_sigmas)
            fixed_posterior = _extract_posterior(fixed_result)
            fixed_weights = {k: float(v) for k, v in fixed_result.get("weights", {}).items()} if isinstance(fixed_result, dict) else {}
            if not fixed_weights:
                _, fixed_weights = _fixed_posterior_with_mask(core, measurements, active_mask, fixed_sigmas)
        else:
            fixed_posterior, fixed_weights = _fixed_posterior_with_mask(core, measurements, active_mask, fixed_sigmas)
    except Exception:
        fixed_posterior, fixed_weights = _fixed_posterior_with_mask(core, measurements, active_mask, fixed_sigmas)

    fixed_metrics = _posterior_grid_metrics(uq, depth_axis, fixed_posterior, delta_x)
    fixed_hazard = hazard_assessor.evaluate(
        depth_axis, fixed_posterior, delta_x,
        fusion_confidence=float(fixed_metrics["confidence"]),
        scene_complexity=scene_complexity)

    adaptive_result = adaptive_engine.fuse(
        z_lidar=float(measurements[0]) if np.isfinite(measurements[0]) else np.nan,
        z_ultra=float(measurements[1]) if np.isfinite(measurements[1]) else np.nan,
        z_radar=float(measurements[2]) if np.isfinite(measurements[2]) else np.nan,
        r_lidar=float(reliabilities[0]),
        r_ultra=float(reliabilities[1]),
        r_radar=float(reliabilities[2]),
        sigmas=fixed_sigmas,
        scene_complexity=scene_complexity,
        sensor_agreement=sensor_agreement,
        reliability_spread=reliability_spread,
        effective_sensor_count=effective_sensor_count,
        dominance_ratio=dominance_ratio,
        confidence=0.5,
        measurement_spread=measurement_spread,
        water_depth=water_context_proxy,
        ntu=ntu_observed,
        active_mask=tuple(bool(v) for v in active_mask),
        sensor_health=sensor_health)

    adaptive_posterior = _extract_posterior(adaptive_result)
    adaptive_metrics = _posterior_grid_metrics(uq, depth_axis, adaptive_posterior, delta_x)

    adaptive_hazard = hazard_assessor.evaluate(
        depth_axis, adaptive_posterior, delta_x,
        fusion_confidence=float(adaptive_metrics["confidence"]),
        scene_complexity=scene_complexity)

    hazard_probability = _normalize_probability(adaptive_hazard.get("hazard_probability", np.nan))
    true_hazard = bool(true_depth >= hazard_assessor.threshold)

    adaptive_ci_lower = adaptive_metrics["ci_lower"]
    adaptive_ci_upper = adaptive_metrics["ci_upper"]

    result = {
        "scenario_id": row["scenario_id"],
        "true_depth": true_depth,
        "water_context_depth_proxy": water_context_proxy,
        "water_context_available": bool(context_row["water_context_available"]),
        "ntu_observed": ntu_observed,
        "ntu_available": bool(context_row["ntu_available"]),
        "scene_complexity": scene_complexity,
        "sensor_agreement": sensor_agreement,
        "measurement_spread": measurement_spread,
        "reliability_spread": reliability_spread,
        "effective_sensor_count": effective_sensor_count,
        "dominance_ratio": dominance_ratio,
        "active_lidar": bool(active_mask[0]),
        "active_ultrasonic": bool(active_mask[1]),
        "active_radar": bool(active_mask[2]),
        "active_sensor_count": int(active_mask.sum()),
        "single_sensor_est": float(measurements[0]) if np.isfinite(measurements[0]) else np.nan,
        "fixed_map_est": fixed_metrics["map_depth"],
        "fixed_expected_est": fixed_metrics["expected_depth"],
        "adaptive_map_est": adaptive_metrics["map_depth"],
        "adaptive_expected_est": adaptive_metrics["expected_depth"],
        "single_error": (abs(float(measurements[0]) - true_depth)
            if np.isfinite(measurements[0]) else np.nan),
        "fixed_error": abs(fixed_metrics["map_depth"] - true_depth),
        "adaptive_error": abs(adaptive_metrics["map_depth"] - true_depth),
        "fixed_mean_error": abs(fixed_metrics["expected_depth"] - true_depth),
        "adaptive_mean_error": abs(adaptive_metrics["expected_depth"] - true_depth),
        "fixed_confidence": fixed_metrics["confidence"],
        "adaptive_confidence": adaptive_metrics["confidence"],
        "fixed_entropy": fixed_metrics["entropy"],
        "adaptive_entropy": adaptive_metrics["entropy"],
        "fixed_variance": fixed_metrics["variance"],
        "adaptive_variance": adaptive_metrics["variance"],
        "fixed_ci_lower_95": fixed_metrics["ci_lower"],
        "fixed_ci_upper_95": fixed_metrics["ci_upper"],
        "fixed_ci_width_95": fixed_metrics["ci_width"],
        "adaptive_ci_lower_95": adaptive_metrics["ci_lower"],
        "adaptive_ci_upper_95": adaptive_metrics["ci_upper"],
        "adaptive_ci_width_95": adaptive_metrics["ci_width"],
        "fixed_posterior_peak": fixed_metrics["posterior_peak"],
        "adaptive_posterior_peak": adaptive_metrics["posterior_peak"],
        "adaptive_nll": _posterior_nll_approx(depth_axis, adaptive_posterior, true_depth),
        "adaptive_crps": _posterior_crps_approx(depth_axis, adaptive_posterior, true_depth),
        "fixed_nll": _posterior_nll_approx(depth_axis, fixed_posterior, true_depth),
        "fixed_crps": _posterior_crps_approx(depth_axis, fixed_posterior, true_depth),
        "adaptive_ci_contains_truth": bool(np.isfinite(adaptive_ci_lower)
            and np.isfinite(adaptive_ci_upper)
            and adaptive_ci_lower <= true_depth <= adaptive_ci_upper),
        "fixed_ci_contains_truth": bool(np.isfinite(fixed_metrics["ci_lower"])
            and np.isfinite(fixed_metrics["ci_upper"])
            and fixed_metrics["ci_lower"] <= true_depth <= fixed_metrics["ci_upper"]),
        "hazard_probability": hazard_probability,
        "hazard_status": str(adaptive_hazard.get("status", "UNKNOWN")),
        "risk_score": float(adaptive_hazard.get("risk_score", np.nan)),
        "true_hazard": true_hazard,
        "hazard_correct_at_boundary": bool(np.isfinite(hazard_probability)
            and ((hazard_probability >= hazard_assessor.decision_boundary) == true_hazard)),
        "fixed_hazard_probability": _normalize_probability(fixed_hazard.get(
            "hazard_probability", np.nan)),
        "fixed_hazard_status": str(fixed_hazard.get("status", "UNKNOWN")),
        "adaptive_weights_json": json.dumps({k: float(v) for k, v in (adaptive_result.get(
            "weights", {}) or {}).items()}, sort_keys=True),
        "adaptive_sigmas_json": json.dumps({k: float(v) for k, v in (adaptive_result.get(
            "sigmas", {}) or {}).items()}, sort_keys=True),
        "fixed_weights_json": json.dumps(fixed_weights, sort_keys=True),
        "adaptive_effective_sensor_count": float(adaptive_result.get(
            "diagnostics", {}).get("effective_sensor_count", effective_sensor_count)),
        "adaptive_dominance_ratio": float(adaptive_result.get(
            "diagnostics", {}).get("dominance_ratio", dominance_ratio)),
        "adaptive_context_difficulty": float(adaptive_result.get(
            "diagnostics", {}).get("context_difficulty", np.nan)),
        "adaptive_adaptation_gate": float(adaptive_result.get(
            "diagnostics", {}).get("adaptation_gate", np.nan)),
        "adaptive_measurement_consistency": float(adaptive_result.get(
            "diagnostics", {}).get("measurement_consistency_mean", np.nan)),
    }

    return result

def _prepare_context_frame(df: pd.DataFrame) -> pd.DataFrame:
    measurements = _sensor_measurements(df)
    reliabilities = _sensor_reliabilities(df)
    active_masks = _sensor_active_masks(df, measurements, reliabilities)
    health = _sensor_health(df)
    arrays = _observable_context_arrays(df, measurements, reliabilities, active_masks)

    context = pd.DataFrame(index=df.index)
    for sensor in SENSOR_NAMES:
        context[f"active_{sensor}"] = active_masks[sensor].astype(bool)
        context[f"health_{sensor}"] = health[sensor].fillna(1.0).clip(0.0, 1.0)

    for key, values in arrays.items():
        context[key] = values
    return context

def _validate_dataset_columns(df: pd.DataFrame) -> None:
    required = {
        "scenario_id", "true_depth", "lidar", "ultrasonic",
        "radar", "R_lidar", "R_ultrasonic", "R_radar"}

    missing = sorted(required - set(df.columns))
    if missing: raise ValueError(
        "dataset_v2 is missing required benchmark columns: " + ", ".join(missing))

    valid_truth = pd.to_numeric(df["true_depth"], errors="coerce").notna()
    if not valid_truth.any(): raise ValueError("No finite true_depth reference values are available")

# ------------------------------------------------------------------
# Statistical Consolidation
# ------------------------------------------------------------------
def _build_pairwise_stats(
    evaluator: StatisticalEvaluator,
    benchmark_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    comparisons = [
        ("adaptive_vs_fixed", benchmark_df["fixed_error"], benchmark_df["adaptive_error"]),
        ("adaptive_vs_single", benchmark_df["single_error"], benchmark_df["adaptive_error"])]

    rows: List[Dict[str, Any]] = []
    for name, baseline, proposed in comparisons:
        out = dict(evaluator.compute_significance(baseline, proposed))
        out["comparison"] = name
        rows.append(out)

    stats_df = pd.DataFrame(rows)
    raw_p = stats_df["p_value"].to_numpy(dtype=float)
    adjusted = evaluator.adjust_pvalues(raw_p, method="holm")
    stats_df["holm_adjusted_p_value"] = adjusted
    stats_df["holm_significant"] = np.isfinite(adjusted) & (adjusted < evaluator.alpha)

    stats_df["direction_supports_adaptive"] = pd.to_numeric(stats_df[
        "improvement_percent"], errors="coerce") > 0.0
    stats_df["adjusted_and_directionally_supported"] = stats_df[
        "holm_significant"] & stats_df["direction_supports_adaptive"]

    metadata = {
        "multiple_comparison_method": "holm",
        "hypothesis_family": ["adaptive_vs_fixed", "adaptive_vs_single"],
        "family_size": int(len(stats_df)),
        "alpha": float(evaluator.alpha)}

    return stats_df, metadata

# ------------------------------------------------------------------
# Validation Evidence Integration
# ------------------------------------------------------------------
def _load_optional_fault_results(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not Path(path).exists(): return pd.DataFrame()
    try: fault_df = pd.read_csv(path)
    except Exception: return pd.DataFrame()
    if fault_df.empty: return pd.DataFrame()
    return fault_df.copy()

def _build_validation_input(
    benchmark_df: pd.DataFrame, ablation_df: pd.DataFrame, fault_df: pd.DataFrame) -> pd.DataFrame:
    pieces = [benchmark_df.copy()]
    if not ablation_df.empty:
        ab = ablation_df.copy()
        if "experiment" in ab.columns: pieces.append(ab)
        elif "category" in ab.columns: pieces.append(ab)

    if not fault_df.empty: pieces.append(fault_df)
    combined = pd.concat(pieces, ignore_index=True, sort=False)
    return combined

def _call_validation_evidence(
    scorer: Any, validation_df: pd.DataFrame, summary_df: pd.DataFrame,
    target_samples: int, stats_df: Optional[pd.DataFrame] = None,
    fault_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:

    try:
        report = scorer.score_dataframe(
            validation_df, summary_df=summary_df,
            target_samples=target_samples,
            stats_df=stats_df, fault_df=fault_df)

        components = report.get("components", {})
        row = {"validation_evidence_score": float(report.get("score", np.nan)),
            "validation_evidence_label": str(report.get("label", "Unknown"))}

        for key, value in components.items(): row[key] = float(value)
        return row
    except Exception as exc:
        return {
            "validation_evidence_score": np.nan,
            "validation_evidence_label": "Unavailable",
            "validation_evidence_error": str(exc),
        }

# ------------------------------------------------------------------
# Benchmark Summary
# ------------------------------------------------------------------
def _summarize_benchmark(
    input_df: pd.DataFrame, benchmark_df: pd.DataFrame, ablation_df: pd.DataFrame,
    stats_df: pd.DataFrame, validation_row: Mapping[str, Any],
    statistical_metadata: Mapping[str, Any]) -> Dict[str, Any]:

    adaptive_error = benchmark_df["adaptive_error"]
    fixed_error = benchmark_df["fixed_error"]
    single_error = benchmark_df["single_error"]

    summary: Dict[str, Any] = {
        "input_rows": int(len(input_df)),
        "evaluated_rows": int(len(benchmark_df)),
        "ablation_rows": int(len(ablation_df)),

        "true_depth_mean_cm": _safe_mean(benchmark_df["true_depth"]),
        "true_depth_std_cm": _safe_std(benchmark_df["true_depth"]),
        "true_depth_p05_cm": _quantile_safe(benchmark_df["true_depth"], 0.05),
        "true_depth_p95_cm": _quantile_safe(benchmark_df["true_depth"], 0.95),

        "single_rmse_cm": _safe_rmse(single_error),
        "fixed_rmse_cm": _safe_rmse(fixed_error),
        "adaptive_rmse_cm": _safe_rmse(adaptive_error),

        "single_mae_cm": _safe_mae(single_error),
        "fixed_mae_cm": _safe_mae(fixed_error),
        "adaptive_mae_cm": _safe_mae(adaptive_error),

        "adaptive_vs_fixed_improvement_percent": ((_safe_rmse(fixed_error) - _safe_rmse(
            adaptive_error)) / max(abs(_safe_rmse(fixed_error)), 1e-12) * 100.0),
        "adaptive_vs_single_improvement_percent": ((_safe_rmse(single_error) - _safe_rmse(
            adaptive_error)) / max(abs(_safe_rmse(single_error)), 1e-12) * 100.0),

        "adaptive_win_rate_vs_fixed_percent": _safe_rate(
            benchmark_df["adaptive_error"] < benchmark_df["fixed_error"]),
        "adaptive_win_rate_vs_single_percent": _safe_rate(
            benchmark_df["adaptive_error"] < benchmark_df["single_error"]),

        "adaptive_confidence_mean": _safe_mean(benchmark_df["adaptive_confidence"]),
        "fixed_confidence_mean": _safe_mean(benchmark_df["fixed_confidence"]),
        "adaptive_entropy_mean": _safe_mean(benchmark_df["adaptive_entropy"]),
        "fixed_entropy_mean": _safe_mean(benchmark_df["fixed_entropy"]),
        "adaptive_variance_mean": _safe_mean(benchmark_df["adaptive_variance"]),
        "fixed_variance_mean": _safe_mean(benchmark_df["fixed_variance"]),

        "adaptive_ci_coverage_95_percent": _safe_rate(benchmark_df["adaptive_ci_contains_truth"]),
        "fixed_ci_coverage_95_percent": _safe_rate(benchmark_df["fixed_ci_contains_truth"]),
        "adaptive_nll_mean": _safe_mean(benchmark_df["adaptive_nll"]),
        "fixed_nll_mean": _safe_mean(benchmark_df["fixed_nll"]),
        "adaptive_crps_mean": _safe_mean(benchmark_df["adaptive_crps"]),
        "fixed_crps_mean": _safe_mean(benchmark_df["fixed_crps"]),

        "adaptive_hazard_brier": _safe_mean(np.square(benchmark_df[
            "hazard_probability"] - benchmark_df["true_hazard"].astype(float))),
        "fixed_hazard_brier": _safe_mean(np.square(benchmark_df[
            "fixed_hazard_probability"] - benchmark_df["true_hazard"].astype(float))),
        "adaptive_hazard_accuracy_percent": _safe_rate(benchmark_df["hazard_correct_at_boundary"]),
        "hazard_rate_percent": _safe_rate(benchmark_df["true_hazard"]),
        "hazard_flags_percent": _safe_rate(benchmark_df[
            "hazard_status"].astype(str).str.upper().eq("HAZARD")),

        "observable_water_context_availability_percent": _safe_rate(
            benchmark_df["water_context_available"]),
        "observable_ntu_availability_percent": _safe_rate(benchmark_df["ntu_available"]),
        "mean_active_sensor_count": _safe_mean(benchmark_df["active_sensor_count"]),

        "adaptive_failure_count": int(benchmark_df["adaptive_map_est"].isna().sum()),
        "fixed_failure_count": int(benchmark_df["fixed_map_est"].isna().sum()),

        "single_to_adaptive_error_correlation": _safe_corr(
            benchmark_df["single_error"], benchmark_df["adaptive_error"]),
        "fixed_to_adaptive_error_correlation": _safe_corr(
            benchmark_df["fixed_error"], benchmark_df["adaptive_error"]),

        "calibrated_sigma_lidar_cm": float(CALIBRATED_BASE_SIGMAS_CM[0]),
        "calibrated_sigma_ultrasonic_cm": float(CALIBRATED_BASE_SIGMAS_CM[1]),
        "calibrated_sigma_radar_cm": float(CALIBRATED_BASE_SIGMAS_CM[2]),

        "statistical_multiple_comparison_method": statistical_metadata.get(
            "multiple_comparison_method", "holm"),
        "statistical_family_size": int(statistical_metadata.get("family_size", len(stats_df))),

        "validation_evidence_score": float(validation_row.get("validation_evidence_score", np.nan)),
        "validation_evidence_label": str(validation_row.get("validation_evidence_label", "Unknown")),
    }

    for _, row in stats_df.iterrows():
        prefix = str(row.get("comparison", "comparison"))
        summary[f"{prefix}_p_value"] = row.get("p_value")
        summary[f"{prefix}_holm_adjusted_p_value"] = row.get("holm_adjusted_p_value")
        summary[f"{prefix}_hedges_g"] = row.get("hedges_g")
        summary[f"{prefix}_improvement_percent"] = row.get("improvement_percent")
        summary[f"{prefix}_ci95_mean_difference"] = row.get("ci95_mean_difference")
        summary[f"{prefix}_bootstrap_ci95_mean_difference"] = row.get(
            "bootstrap_ci95_mean_difference")
        summary[f"{prefix}_is_significant"] = bool(row.get("is_significant", False))
        summary[f"{prefix}_holm_significant"] = bool(row.get("holm_significant", False))
    return summary

# ------------------------------------------------------------------
# Campaign Metadata
# ------------------------------------------------------------------
def _write_campaign_metadata(
    path: Path, config: CampaignConfig, input_df: pd.DataFrame,
    benchmark_df: pd.DataFrame, ablation_df: pd.DataFrame,
    stats_metadata: Mapping[str, Any]) -> None:

    payload = {
        "pipeline": "FloodTwin-HIL Final Validation Benchmark",
        "dataset_path": str(config.dataset_path),
        "random_seed": int(config.random_seed),
        "benchmark_limit": config.benchmark_limit,
        "ablation_limit": config.ablation_limit,
        "input_rows": int(len(input_df)),
        "benchmark_rows": int(len(benchmark_df)),
        "ablation_rows": int(len(ablation_df)),
        "depth_range_cm": [DEPTH_MIN_CM, DEPTH_MAX_CM],
        "depth_resolution": int(DEPTH_RESOLUTION),
        "fixed_base_sigmas_cm": list(config.fixed_sigmas_cm),
        "truth_used_only_for_evaluation": True,

        "observable_context_fields": [
            "water_context_depth_proxy",
            "water_contact",
            "ntu_observed",
            "scene_complexity",
            "sensor_agreement",
            "measurement_spread",
            "reliability_spread",
        ],

        "ground_truth_water_depth_excluded_from_context": True,
        "statistical_metadata": dict(stats_metadata),
        "validation_score_semantics": (
            "Validation Evidence Score is an engineering-defined composite "
            "evidence summary, not statistical confidence or probability of research correctness."),
    }

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

# ------------------------------------------------------------------
# Main Campaign
# ------------------------------------------------------------------
def run_campaign(
    dataset_path: Path = DATASET_FILE_IN, benchmark_limit: Optional[int] = None,
    ablation_limit: Optional[int] = DEFAULT_ABLATION_LIMIT, random_seed: int = DEFAULT_RANDOM_SEED,
    verbose: bool = True, fixed_sigmas: Tuple[float, float, float] = CALIBRATED_BASE_SIGMAS_CM,
    hazard_threshold_cm: float = 10.0, hazard_decision_boundary: float = 0.80,
    bootstrap_samples: int = 5000, fault_results_path: Optional[Path] = DEFAULT_FAULT_RESULTS_FILE) -> CampaignArtifacts:

    dataset_path = Path(dataset_path)
    if not dataset_path.exists(): raise FileNotFoundError(f"{dataset_path} not found")

    df = pd.read_csv(dataset_path).copy()
    _validate_dataset_columns(df)

    df["true_depth"] = pd.to_numeric(df["true_depth"], errors="coerce")
    df = df.loc[df["true_depth"].notna()].copy()

    for sensor in SENSOR_NAMES:
        measurement_col = SENSOR_MEASUREMENT_COLUMNS[sensor]
        reliability_col = SENSOR_RELIABILITY_COLUMNS[sensor]

        df[measurement_col] = pd.to_numeric(df[measurement_col], errors="coerce")
        df[reliability_col] = pd.to_numeric(df[reliability_col], errors="coerce").clip(0.0, 1.0)

    context_df = _prepare_context_frame(df)
    for col in context_df.columns: df[col] = context_df[col].to_numpy()
    benchmark_source = _stable_stratified_sample(df, benchmark_limit, seed=random_seed)

    core = BayesianFusionCore(
        depth_min=DEPTH_MIN_CM,
        depth_max=DEPTH_MAX_CM,
        resolution=DEPTH_RESOLUTION)

    fixed_engine = FixedFusionEngine(core)
    adaptive_engine = AdaptiveFusionEngine(core)

    uq = UncertaintyQuantification()
    hazard_assessor = HazardAssessment(
        threshold_cm=float(hazard_threshold_cm),
        decision_boundary=float(hazard_decision_boundary))

    fixed_sigmas = tuple(float(x) for x in fixed_sigmas)
    if len(fixed_sigmas) != 3 or any(not np.isfinite(x) or x <= 0.0 for x in fixed_sigmas):
        raise ValueError("fixed_sigmas must contain three finite positive calibrated values")

    benchmark_eval_rows: List[Dict[str, Any]] = []
    health_frame = _sensor_health(benchmark_source)

    for idx, (_, row) in enumerate(benchmark_source.iterrows()):
        context_row = {
            "scene_complexity": float(benchmark_source.iloc[idx]["scene_complexity"]),
            "sensor_agreement": float(benchmark_source.iloc[idx]["sensor_agreement"]),
            "reliability_spread": float(benchmark_source.iloc[idx]["reliability_spread"]),
            "effective_sensor_count": float(benchmark_source.iloc[idx]["effective_sensor_count"]),
            "dominance_ratio": float(benchmark_source.iloc[idx]["dominance_ratio"]),
            "measurement_spread": float(benchmark_source.iloc[idx]["measurement_spread"]),
            "water_context_depth_proxy": float(benchmark_source.iloc[idx][
                "water_context_depth_proxy"]),
            "water_context_available": bool(benchmark_source.iloc[idx]["water_context_available"]),
            "ntu_observed": float(benchmark_source.iloc[idx]["ntu_observed"]),
            "ntu_available": bool(benchmark_source.iloc[idx]["ntu_available"]),
            "active_lidar": bool(benchmark_source.iloc[idx]["active_lidar"]),
            "active_ultrasonic": bool(benchmark_source.iloc[idx]["active_ultrasonic"]),
            "active_radar": bool(benchmark_source.iloc[idx]["active_radar"]),
            "health_lidar": float(health_frame["lidar"].loc[row.name]),
            "health_ultrasonic": float(health_frame["ultrasonic"].loc[row.name]),
            "health_radar": float(health_frame["radar"].loc[row.name]),
        }

        benchmark_eval_rows.append(_evaluate_single_row(
            row=row, core=core,
            fixed_engine=fixed_engine,
            adaptive_engine=adaptive_engine, uq=uq,
            hazard_assessor=hazard_assessor,
            fixed_sigmas=fixed_sigmas,
            context_row=context_row))

    benchmark_df = pd.DataFrame(benchmark_eval_rows)
    if benchmark_df.empty: raise RuntimeError("Benchmark evaluation produced no rows")

    if ablation_limit is None:
        ablation_source = benchmark_source.copy()
    else:
        ablation_source = _stable_stratified_sample(
            benchmark_source,
            min(int(ablation_limit), len(benchmark_source)),
            seed=random_seed + 7)

    ablation_framework = AblationFramework(
        adaptive_engine=adaptive_engine,
        fixed_engine=fixed_engine,
        core=core, uq=uq,
        hazard_assessor=hazard_assessor)

    ablation_records: List[Dict[str, Any]] = []

    for _, row in ablation_source.iterrows():
        active_mask = (
            bool(row["active_lidar"]),
            bool(row["active_ultrasonic"]),
            bool(row["active_radar"]))

        sensor_health_map = {
            "lidar": float(row["health_lidar"]),
            "ultrasonic": float(row["health_ultrasonic"]),
            "radar": float(row["health_radar"]),
        }

        ablation_results = ablation_framework.run_experiments(
            z_lidar=row["lidar"],
            z_ultra=row["ultrasonic"],
            z_radar=row["radar"],
            r_lidar=row["R_lidar"],
            r_ultra=row["R_ultrasonic"],
            r_radar=row["R_radar"],
            true_depth=float(row["true_depth"]),
            scene_complexity=float(row["scene_complexity"]),
            sensor_agreement=float(row["sensor_agreement"]),
            reliability_spread=float(row["reliability_spread"]),
            effective_sensor_count=float(row["effective_sensor_count"]),
            dominance_ratio=float(row["dominance_ratio"]),
            confidence=0.5,
            measurement_spread=float(row["measurement_spread"]),
            water_context_depth_proxy=float(row["water_context_depth_proxy"]),
            ntu=float(row["ntu_observed"]),
            sensor_health=sensor_health_map,
            base_sigmas=fixed_sigmas,
            active_mask=active_mask)

        for experiment, result in ablation_results.items():
            record: Dict[str, Any] = {
                "scenario_id": row["scenario_id"],
                "experiment": experiment,
                "category": str(row.get("benchmark_category", "All")),
                "true_depth": float(row["true_depth"]),
                "estimate": float(result.get("map_depth", np.nan)),
                "expected_estimate": float(result.get("expected_depth", np.nan)),
                "error": float(result.get("abs_error_map", np.nan)),
                "mean_error": float(result.get("abs_error_mean", np.nan)),
                "confidence": float(result.get("confidence", np.nan)),
                "entropy": float(result.get("entropy", np.nan)),
                "variance": float(result.get("variance", np.nan)),
                "ci_width_95": float(result.get("ci_width_95", np.nan)),
                "status": str(result.get("status", result.get(
                    "diagnostics", {}).get("status", "OK"))),
                "weights_json": json.dumps(result.get("weights", {}), sort_keys=True),
                "sigmas_json": json.dumps(result.get("sigmas", {}), sort_keys=True),
            }

            ablation_records.append(record)
    ablation_df = pd.DataFrame(ablation_records)

    evaluator = StatisticalEvaluator(
        alpha=0.05,
        bootstrap_samples=max(500, int(bootstrap_samples)),
        random_seed=random_seed)

    stats_df, stats_metadata = _build_pairwise_stats(evaluator, benchmark_df)
    fault_df = _load_optional_fault_results(fault_results_path)

    preliminary_summary = pd.DataFrame([{
        "samples": int(len(benchmark_df)),
        "adaptive_rmse_cm": _safe_rmse(benchmark_df["adaptive_error"]),
        "fixed_rmse_cm": _safe_rmse(benchmark_df["fixed_error"]),
        "adaptive_mae_cm": _safe_mae(benchmark_df["adaptive_error"]),
        "fixed_mae_cm": _safe_mae(benchmark_df["fixed_error"]),
        "adaptive_ci_coverage_95": _safe_fraction(benchmark_df["adaptive_ci_contains_truth"]),
        "adaptive_nll": _safe_mean(benchmark_df["adaptive_nll"]),
        "fixed_nll": _safe_mean(benchmark_df["fixed_nll"]),
        "adaptive_crps": _safe_mean(benchmark_df["adaptive_crps"]),
        "fixed_crps": _safe_mean(benchmark_df["fixed_crps"]),
        "adaptive_hazard_brier": _safe_mean(np.square(benchmark_df[
            "hazard_probability"] - benchmark_df["true_hazard"].astype(float))),
        "adaptive_hazard_accuracy": _safe_fraction(benchmark_df["hazard_correct_at_boundary"]),
        "fault_experiments": (int(fault_df["experiment"].nunique(dropna=True))
            if "experiment" in fault_df.columns and not fault_df.empty else 0),
    }])

    validation_scorer = ValidationEvidenceScorer()
    validation_row = _call_validation_evidence(
        scorer=validation_scorer,
        validation_df=benchmark_df,
        summary_df=preliminary_summary,
        target_samples=30000,
        stats_df=stats_df,
        fault_df=fault_df)

    summary = _summarize_benchmark(
        input_df=df,
        benchmark_df=benchmark_df,
        ablation_df=ablation_df,
        stats_df=stats_df,
        validation_row=validation_row,
        statistical_metadata=stats_metadata)

    summary_df = pd.DataFrame([summary])
    validation_df = pd.DataFrame([validation_row])

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df.to_csv(BENCHMARK_RESULTS_FILE, index=False)
    ablation_df.to_csv(ABLATION_RESULTS_FILE, index=False)
    validation_df.to_csv(VALIDATION_SCORES_FILE, index=False)
    stats_df.to_csv(STATISTICAL_SUMMARY_FILE, index=False)
    summary_df.to_csv(BENCHMARK_SUMMARY_FILE, index=False)

    _write_campaign_metadata(CAMPAIGN_METADATA_FILE, CampaignConfig(
        dataset_path=dataset_path,
        results_dir=results_dir,
        benchmark_limit=benchmark_limit,
        ablation_limit=ablation_limit,
        random_seed=random_seed,
        fixed_sigmas_cm=fixed_sigmas,
        hazard_threshold_cm=hazard_threshold_cm,
        hazard_decision_boundary=hazard_decision_boundary,
        statistical_alpha=evaluator.alpha,
        bootstrap_samples=evaluator.bootstrap_samples,
        validation_target_samples=30000,
        fault_results_path=fault_results_path),
        input_df=df,
        benchmark_df=benchmark_df,
        ablation_df=ablation_df,
        stats_metadata=stats_metadata)

    # ------------------------------------------------------------------
    # Console Reporting
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print(f"{'FloodTwin-HIL Final Validation Benchmark':^60}")
        print("=" * 60)

        print("\n[Dataset / Execution]")
        print(f"  Source Rows                 : {len(df):,}")
        print(f"  Clean Benchmark Rows        : {len(benchmark_df):,}")
        print(f"  Ablation Rows               : {len(ablation_df):,}")
        print(f"  Observable Water Context    : {summary['observable_water_context_availability_percent']:.1f}%")
        print(f"  Observable NTU Availability : {summary['observable_ntu_availability_percent']:.1f}%")

        print("\n[Depth Estimation]")
        print(f"  LiDAR-Only RMSE             : {summary['single_rmse_cm']:.3f} cm")
        print(f"  Fixed Bayesian RMSE         : {summary['fixed_rmse_cm']:.3f} cm")
        print(f"  Adaptive Bayesian RMSE      : {summary['adaptive_rmse_cm']:.3f} cm")
        print(f"  Adaptive vs Fixed Reduction : {summary['adaptive_vs_fixed_improvement_percent']:.2f}%")
        print(f"  Adaptive vs Fixed Win Rate  : {summary['adaptive_win_rate_vs_fixed_percent']:.2f}%")

        print("\n[Uncertainty / Calibration]")
        print(f"  Adaptive 95% CI Coverage    : {summary['adaptive_ci_coverage_95_percent']:.2f}%")
        print(f"  Adaptive Mean NLL           : {summary['adaptive_nll_mean']:.4f}")
        print(f"  Adaptive Mean CRPS          : {summary['adaptive_crps_mean']:.4f}")

        print("\n[Hazard Assessment]")
        print(f"  True Hazardous Cases        : {summary['hazard_rate_percent']:.2f}%")
        print(f"  Adaptive Hazard Brier       : {summary['adaptive_hazard_brier']:.5f}")
        print(f"  Adaptive Hazard Accuracy    : {summary['adaptive_hazard_accuracy_percent']:.2f}%")

        print("\n[Statistical Evidence]")
        for _, stat in stats_df.iterrows():
            print(f"  {stat['comparison']:<22} "
                f"raw p={stat['p_value']:.3e} | "
                f"Holm p={stat['holm_adjusted_p_value']:.3e} | "
                f"g={stat['hedges_g']:.3f} | "
                f"Δ={stat['improvement_percent']:.2f}%")

        print("\n[Validation Evidence]")
        print(f"  Evidence Score              : {validation_row.get('validation_evidence_score', np.nan):.3f}")
        print(f"  Evidence Band               : {validation_row.get('validation_evidence_label', 'Unknown')}")

        print("\n[Outputs]")
        print(f"  Results Directory            : {results_dir}")
        print(f"  Benchmark Results            : {BENCHMARK_RESULTS_FILE.name}")
        print(f"  Ablation Results             : {ABLATION_RESULTS_FILE.name}")
        print(f"  Statistical Summary          : {STATISTICAL_SUMMARY_FILE.name}")
        print(f"  Validation Evidence          : {VALIDATION_SCORES_FILE.name}")
        print(f"  Benchmark Summary            : {BENCHMARK_SUMMARY_FILE.name}")
        print(f"  Campaign Metadata            : {CAMPAIGN_METADATA_FILE.name}")
        print("\n" + "=" * 60)

    return CampaignArtifacts(benchmark_df=benchmark_df, ablation_df=ablation_df,
        statistical_df=stats_df, validation_df=validation_df, summary_df=summary_df)

if __name__ == "__main__":
    run_campaign()