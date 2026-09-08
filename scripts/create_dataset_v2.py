import json
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reliability_engine.lidar_reliability import LidarReliabilityModel
from reliability_engine.ultrasonic_reliability import UltrasonicReliabilityModel
from reliability_engine.radar_reliability import RadarReliabilityModel
from reliability_engine.reliability_fusion import (compute_adaptive_weights, compute_diagnostics)

DATASET_V1 = BASE_DIR / "datasets" / "dataset_v1.csv"
DATASET_V2 = BASE_DIR / "datasets" / "dataset_v2.csv"

RESULTS_DIR = BASE_DIR / "results" / "reliability_engine"
SUMMARY_FILE = RESULTS_DIR / "reliability_summary.csv"
CORR_PEARSON_FILE = RESULTS_DIR / "reliability_correlations_pearson.csv"
CORR_SPEARMAN_FILE = RESULTS_DIR / "reliability_correlations_spearman.csv"
QUANTILES_FILE = RESULTS_DIR / "reliability_quantiles.csv"
METADATA_FILE = BASE_DIR / "datasets" / "metadata.json"

# Constants
EPS = 1e-12
MAX_WATER_DEPTH_CONTEXT = 20.0
MAX_NTU_CONTEXT = 500.0
MAX_CLUTTER_CONTEXT = 0.30
NOMINAL_MAX_SPREAD_CM = 15.0
RISK_LEVELS = {"none": 0, "low": 1, "moderate": 2, "medium": 2, "high": 3, "severe": 3, "extreme": 4}

# ------------------------------------------------------------------
# Generic Utilities
# ------------------------------------------------------------------
def _severity_to_level(value):
    return int(RISK_LEVELS.get(str(value).strip().lower(), 2))

def _safe_series(df, column, default=0.0):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(float(default), index=df.index, dtype=float)

def _safe_radar_rcs(df):
    if "rcs" not in df.columns: return None
    return pd.to_numeric(df["rcs"], errors="coerce")

def _clip01(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.clip(0.0, 1.0)

def _entropy_from_weights(weights):
    values = np.asarray(weights, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))

    if total <= EPS: return 0.0
    values = values / total
    positive = values > EPS
    return float(-np.sum(values[positive] * np.log(values[positive])))

def _consistency_index(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if int(np.sum(valid)) <= 1: return np.nan
    values = values[valid]
    spread = float(np.std(values))
    magnitude = float(np.mean(np.abs(values)))
    consistency = 1.0 - (spread / (magnitude + spread + EPS))
    return float(np.clip(consistency, 0.0, 1.0))

def _safe_finite(values, default=0.0):
    values = np.asarray(values, dtype=float)
    return np.nan_to_num(values, nan=default, posinf=default, neginf=default)

# ------------------------------------------------------------------
# Observable Water-Context Estimation
# ------------------------------------------------------------------
def _water_state_score(value):
    text = str(value).strip().lower()
    if text in {"dry", "none", "no_water", "no water"}: return 0.0
    if text in {"shallow", "low", "wet"}: return 0.35
    if text in {"flooded", "deep", "high"}: return 0.75
    if text in {"severe", "extreme"}: return 1.0
    return 0.0

def _build_water_context_depth_proxy(df):
    # Water Presence
    if "water_present" in df.columns:
        water_present = _clip01(df["water_present"]).fillna(0.0)
    elif "water_contact" in df.columns:
        water_present = _clip01(df["water_contact"]).fillna(0.0)
    else:
        water_present = pd.Series(0.0, index=df.index, dtype=float)

    # Qualitative State
    if "water_state" in df.columns:
        state_score = pd.Series(
            [_water_state_score(v) for v in df["water_state"]],
            index=df.index, dtype=float)
    else:
        state_score = pd.Series(0.0, index=df.index, dtype=float)

    # Echo Observables
    if "surface_echo" in df.columns:
        surface_echo = _clip01(df["surface_echo"]).fillna(0.0)
    else:
        surface_echo = pd.Series(0.0, index=df.index, dtype=float)

    if "bottom_echo" in df.columns:
        bottom_echo = _clip01(df["bottom_echo"]).fillna(0.0)
    else:
        bottom_echo = pd.Series(1.0, index=df.index, dtype=float)

    if "bottom_confidence" in df.columns:
        bottom_confidence = _clip01(df["bottom_confidence"]).fillna(1.0)
    else:
        bottom_confidence = pd.Series(1.0, index=df.index, dtype=float)

    if "echo_ambiguity" in df.columns:
        echo_ambiguity = _clip01(df["echo_ambiguity"]).fillna(0.0)
    else:
        echo_ambiguity = (surface_echo * (1.0 - bottom_echo)).clip(0.0, 1.0)

    # Context Score
    water_context_score = (0.35 * water_present
        + 0.35 * state_score
        + 0.15 * surface_echo
        + 0.05 * (1.0 - bottom_confidence)
        + 0.10 * echo_ambiguity)

    water_context_score = water_context_score.clip(0.0, 1.0)
    estimate = water_context_score * MAX_WATER_DEPTH_CONTEXT
    estimate = np.where(water_present.to_numpy(dtype=float) > 0.5, estimate, 0.0)
    estimate = np.nan_to_num(estimate, nan=0.0, posinf=MAX_WATER_DEPTH_CONTEXT, neginf=0.0)
    return np.clip(estimate, 0.0, MAX_WATER_DEPTH_CONTEXT).astype(float)

# ------------------------------------------------------------------
# Observable Scene-Complexity Model
# ------------------------------------------------------------------
def _build_scene_complexity(
    observable_water_depth, ntu, clutter, measurement_spread,
    measurement_consistency, snr, echo_ambiguity):

    observable_water_depth = _safe_finite(observable_water_depth, default=0.0)
    ntu = _safe_finite(ntu, default=0.0)
    clutter = _safe_finite(clutter, default=0.10)
    measurement_spread = _safe_finite(measurement_spread, default=0.0)
    snr = _safe_finite(snr, default=15.0)
    echo_ambiguity = _safe_finite(echo_ambiguity, default=0.0)

    # If consistency is missing (due to 1 or 0 sensors), it implies zero disagreement penalty
    measurement_consistency = _safe_finite(measurement_consistency, default=1.0)

    water_norm = np.clip(observable_water_depth / MAX_WATER_DEPTH_CONTEXT, 0.0, 1.0)
    water_norm = np.power(water_norm, 1.20)

    ntu_norm = np.clip(ntu / MAX_NTU_CONTEXT, 0.0, 1.0)
    clutter_norm = np.clip(clutter / MAX_CLUTTER_CONTEXT, 0.0, 1.0)
    clutter_norm = np.power(clutter_norm, 1.20)
    spread_norm = np.clip(measurement_spread / NOMINAL_MAX_SPREAD_CM, 0.0, 1.0)

    snr_norm = np.clip((snr - 5.0) / 25.0, 0.0, 1.0)
    snr_difficulty = 1.0 - snr_norm

    disagreement = 1.0 - np.clip(measurement_consistency, 0.0, 1.0)
    echo_ambiguity = np.clip(echo_ambiguity, 0.0, 1.0)

    scene = (0.18 * water_norm
        + 0.18 * ntu_norm
        + 0.14 * clutter_norm
        + 0.20 * spread_norm
        + 0.12 * disagreement
        + 0.10 * snr_difficulty
        + 0.08 * echo_ambiguity)

    return np.clip(scene, 0.0, 1.0)

def _sanitize_reliability(values, floor=0.0, ceiling=1.0):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Reliability model generated non-finite values")
    return np.clip(values, floor, ceiling)

# ------------------------------------------------------------------
# Reliability Dataset Generation
# ------------------------------------------------------------------
def generate_reliability_dataset():
    if not DATASET_V1.exists(): raise FileNotFoundError(f"Dataset V1 not found: {DATASET_V1}")
    df = pd.read_csv(DATASET_V1).copy()

    required_cols = [
        "scenario_id", "water_depth", "ntu", "severity", "lidar", "ultrasonic", "radar", "snr"]

    missing = [column for column in required_cols if column not in df.columns]
    if missing: raise ValueError(f"Missing required columns in dataset_v1: {missing}")
    if df.empty: raise ValueError("dataset_v1 is empty")

    # Canonical Reliability Models
    lidar_model = LidarReliabilityModel()
    ultrasonic_model = UltrasonicReliabilityModel()
    radar_model = RadarReliabilityModel()

    lidar_values = pd.to_numeric(df["lidar"], errors="coerce")
    ultrasonic_values = pd.to_numeric(df["ultrasonic"], errors="coerce")
    radar_values = pd.to_numeric(df["radar"], errors="coerce")

    lidar_values = lidar_values.where(np.isfinite(lidar_values), np.nan)
    ultrasonic_values = ultrasonic_values.where(np.isfinite(ultrasonic_values), np.nan)
    radar_values = radar_values.where(np.isfinite(radar_values), np.nan)

    df["lidar_available"] = lidar_values.notna()
    df["ultrasonic_available"] = ultrasonic_values.notna()
    df["radar_available"] = radar_values.notna()

    df["sensor_count_valid"] = df[["lidar_available",
    "ultrasonic_available", "radar_available"]].sum(axis=1)

    # Observable Context & Availability
    ntu_raw = pd.to_numeric(df["ntu"], errors="coerce")
    ntu_raw = ntu_raw.where(np.isfinite(ntu_raw), np.nan)

    df["ntu_available"] = ntu_raw.notna()
    ntu = ntu_raw.fillna(0.0).clip(lower=0.0, upper=MAX_NTU_CONTEXT)
    df["ntu"] = ntu.to_numpy(dtype=float)

    snr_raw = pd.to_numeric(df["snr"], errors="coerce")
    snr_raw = snr_raw.where(np.isfinite(snr_raw), np.nan)

    df["snr_available"] = snr_raw.notna()
    snr = snr_raw.fillna(15.0)
    df["snr"] = snr.to_numpy(dtype=float)

    if "clutter_probability" in df.columns:
        df["clutter_probability_available"] = df["clutter_probability"].notna()
        clutter = _safe_series(df, "clutter_probability", default=np.nan)
    else:
        df["clutter_probability_available"] = False
        clutter = pd.Series(np.nan, index=df.index, dtype=float)

    if clutter.isna().all():
        clutter = pd.Series(0.10, index=df.index, dtype=float)
    else:
        clutter = clutter.fillna(0.10)

    clutter = clutter.clip(lower=0.0, upper=MAX_CLUTTER_CONTEXT)
    df["clutter_probability"] = clutter.to_numpy(dtype=float)

    # Observable Echo Context
    if "surface_echo" in df.columns:
        df["surface_echo_available"] = df["surface_echo"].notna()
        surface_echo = _clip01(df["surface_echo"]).fillna(0.0)
    else:
        df["surface_echo_available"] = False
        surface_echo = pd.Series(0.0, index=df.index, dtype=float)
    df["surface_echo"] = surface_echo.to_numpy(dtype=float)

    if "bottom_echo" in df.columns:
        df["bottom_echo_available"] = df["bottom_echo"].notna()
        bottom_echo = _clip01(df["bottom_echo"]).fillna(0.5)
    else:
        df["bottom_echo_available"] = False
        bottom_echo = pd.Series(0.5, index=df.index, dtype=float)
    df["bottom_echo"] = bottom_echo.to_numpy(dtype=float)

    if "bottom_confidence" in df.columns:
        df["bottom_confidence_available"] = df["bottom_confidence"].notna()
        bottom_confidence = _clip01(df["bottom_confidence"]).fillna(0.5)
    else:
        df["bottom_confidence_available"] = False
        bottom_confidence = pd.Series(0.5, index=df.index, dtype=float)
    df["bottom_confidence"] = bottom_confidence.to_numpy(dtype=float)

    if "echo_ambiguity" in df.columns:
        df["echo_ambiguity_available"] = df["echo_ambiguity"].notna()
        echo_ambiguity = _clip01(df["echo_ambiguity"]).fillna(0.0)
    else:
        df["echo_ambiguity_available"] = False
        echo_ambiguity = (surface_echo * (1.0 - bottom_echo)).clip(0.0, 1.0)
    df["echo_ambiguity"] = echo_ambiguity.to_numpy(dtype=float)

    # Observable Water Context Estimate
    water_context_depth_proxy = _build_water_context_depth_proxy(df)
    df["water_context_depth_proxy"] = water_context_depth_proxy

    # Measurement Errors
    lidar_abs_error = _safe_series(df, "lidar_error", default=np.nan).abs()
    ultrasonic_abs_error = _safe_series(df, "ultrasonic_error", default=np.nan).abs()
    radar_abs_error = _safe_series(df, "radar_error", default=np.nan).abs()

    df["lidar_error_abs"] = lidar_abs_error
    df["ultrasonic_error_abs"] = ultrasonic_abs_error
    df["radar_error_abs"] = radar_abs_error

    # Measurement Spread / Consistency
    measurement_matrix = pd.DataFrame({
        "lidar": lidar_values, "ultrasonic": ultrasonic_values, "radar": radar_values
        }, index=df.index)

    measurement_spread = measurement_matrix.std(axis=1, skipna=True, ddof=1).fillna(0.0)
    measurement_consistency = pd.Series([_consistency_index([lidar, ultrasonic, radar])
        for lidar, ultrasonic, radar in zip(lidar_values, ultrasonic_values, radar_values)],
        index=df.index, dtype=float)

    # Observable Scene Complexity
    scene_complexity = _build_scene_complexity(
        observable_water_depth=water_context_depth_proxy,
        ntu=ntu.to_numpy(dtype=float),
        clutter=clutter.to_numpy(dtype=float),
        measurement_spread=measurement_spread.to_numpy(dtype=float),
        measurement_consistency=measurement_consistency.to_numpy(dtype=float),
        snr=snr.to_numpy(dtype=float),
        echo_ambiguity=echo_ambiguity.to_numpy(dtype=float))

    # Canonical Sensor Reliability
    lidar_noise_sigma = (0.36
        + 0.0012 * ntu.to_numpy(dtype=float)
        + 0.0065 * np.asarray(water_context_depth_proxy, dtype=float))

    lidar_noise_sigma = np.clip(lidar_noise_sigma, 0.05, 5.0)

    if "rcs" in df.columns:
        radar_rcs = pd.to_numeric(df["rcs"], errors="coerce")
        radar_rcs = radar_rcs.where(np.isfinite(radar_rcs), np.nan)
        df["rcs_available"] = radar_rcs.notna()
    else:
        df["rcs_available"] = False
        radar_rcs = pd.Series(np.nan, index=df.index, dtype=float)

    radar_rcs_filled = radar_rcs.fillna(0.0).clip(lower=0.0)

    r_lidar_raw = pd.Series(lidar_model.compute_reliability(
        ntu=ntu.to_numpy(dtype=float),
        water_depth=np.asarray(water_context_depth_proxy, dtype=float),
        noise_sigma=lidar_noise_sigma),
        index=df.index, dtype=float)

    r_ultrasonic_raw = pd.Series(ultrasonic_model.compute_reliability(
        water_depth=np.asarray(water_context_depth_proxy, dtype=float),
        surface_echo_prob=surface_echo.to_numpy(dtype=float),
        bottom_echo_prob=bottom_echo.to_numpy(dtype=float)),
        index=df.index, dtype=float)

    r_radar_no_rcs = pd.Series(radar_model.compute_reliability(
        snr=snr.to_numpy(dtype=float),
        clutter=clutter.to_numpy(dtype=float),
        water_depth=np.asarray(water_context_depth_proxy, dtype=float)),
        index=df.index, dtype=float)

    r_radar_with_rcs = pd.Series(radar_model.compute_reliability(
        snr=snr.to_numpy(dtype=float),
        clutter=clutter.to_numpy(dtype=float),
        water_depth=np.asarray(water_context_depth_proxy, dtype=float),
        rcs=radar_rcs_filled.to_numpy(dtype=float)),
        index=df.index, dtype=float)

    r_radar_raw = np.where(df["rcs_available"], r_radar_with_rcs, r_radar_no_rcs)

    # Reliability Sanitization and Availability Masking
    r_lidar = np.where(df["lidar_available"], _sanitize_reliability(r_lidar_raw), 0.0)
    r_ultrasonic = np.where(df["ultrasonic_available"], _sanitize_reliability(r_ultrasonic_raw), 0.0)
    r_radar = np.where(df["radar_available"], _sanitize_reliability(r_radar_raw), 0.0)

    df["R_lidar"] = pd.Series(r_lidar, index=df.index, dtype=float)
    df["R_ultrasonic"] = pd.Series(r_ultrasonic, index=df.index, dtype=float)
    df["R_radar"] = pd.Series(r_radar, index=df.index, dtype=float)

    # Existing Context Reliabilities
    if "R_imu" not in df.columns: df["R_imu"] = 0.96
    else: df["R_imu"] = (pd.to_numeric(df["R_imu"], errors="coerce").fillna(0.0).clip(0.0, 1.0))

    if "R_water" not in df.columns: df["R_water"] = 0.95
    else: df["R_water"] = (pd.to_numeric(df["R_water"], errors="coerce").fillna(0.0).clip(0.0, 1.0))

    # Severity Metadata
    severity_level = df["severity"].map(_severity_to_level).astype(int)
    severity_score = (severity_level / 4.0).clip(0.0, 1.0)
    df["severity_level"] = severity_level
    df["severity_score"] = severity_score

    # Context Diagnostics
    df["scene_complexity"] = np.asarray(scene_complexity, dtype=float)
    df["measurement_spread"] = measurement_spread.to_numpy(dtype=float)
    df["measurement_consistency"] = measurement_consistency.to_numpy(dtype=float)

    # Reliability Distribution Diagnostics
    reliability_matrix = df[["R_lidar", "R_ultrasonic", "R_radar"]]

    df["reliability_spread"] = (reliability_matrix.max(axis=1)
        - reliability_matrix.min(axis=1)).clip(0.0, 1.0)
    df["reliability_entropy"] = [
        _entropy_from_weights(row) for row in reliability_matrix.to_numpy(dtype=float)]

    # Adaptive Reliability Weight Diagnostics
    final_weights = []
    final_diagnostics = []

    for (rl, ru, rr, scene) in zip(
        df["R_lidar"], df["R_ultrasonic"], df["R_radar"], df["scene_complexity"]):

        wf = compute_adaptive_weights(
            float(rl), float(ru), float(rr), scene_complexity=float(scene))

        dfinal = compute_diagnostics(
            float(rl), float(ru), float(rr), scene_complexity=float(scene))

        final_weights.append(wf)
        final_diagnostics.append(dfinal)

    # Final Fusion Diagnostics
    df["W_lidar"] = [float(w[0]) for w in final_weights]
    df["W_ultrasonic"] = [float(w[1]) for w in final_weights]
    df["W_radar"] = [float(w[2]) for w in final_weights]

    df["fusion_entropy"] = [float(d["entropy"]) for d in final_diagnostics]
    df["fusion_confidence"] = [float(d["confidence"]) for d in final_diagnostics]
    df["effective_sensor_count"] = [float(d["effective_sensor_count"]) for d in final_diagnostics]

    df["dominant_sensor"] = [d["dominant_sensor"] for d in final_diagnostics]
    df["dominance_ratio"] = [float(d["dominance_ratio"]) for d in final_diagnostics]
    df["dominant_sensor"] = df["dominant_sensor"].astype("string")

    # Backward Compatibility Layer
    df["sensor_agreement"] = df["measurement_consistency"]
    df["water_depth_estimate"] = df["water_context_depth_proxy"]

    df["R_lidar_base"] = df["R_lidar"]
    df["R_ultrasonic_base"] = df["R_ultrasonic"]
    df["R_radar_base"] = df["R_radar"]

    df["W_lidar_base"] = df["W_lidar"]
    df["W_ultrasonic_base"] = df["W_ultrasonic"]
    df["W_radar_base"] = df["W_radar"]

    df["fusion_entropy_base"] = df["fusion_entropy"]
    df["effective_sensor_count_base"] = df["effective_sensor_count"]
    df["dominant_sensor_base"] = df["dominant_sensor"]
    df["dominance_ratio_base"] = df["dominance_ratio"]
    df["fusion_confidence_base"] = df["fusion_confidence"]

    df["reliability_spread_base"] = df["reliability_spread"]
    df["reliability_entropy_base"] = df["reliability_entropy"]

    # Final Fused Estimation & Accuracy
    valid_mask = df["sensor_count_valid"] > 0

    df["fusion_depth_linear"] = np.where(valid_mask,
        (df["W_lidar"] * lidar_values.fillna(0.0)
        + df["W_ultrasonic"] * ultrasonic_values.fillna(0.0)
        + df["W_radar"] * radar_values.fillna(0.0)), np.nan)

    df["fusion_error"] = np.where(valid_mask,
        (df["W_lidar"] * _safe_series(df, "lidar_error", default=0.0)
        + df["W_ultrasonic"] * _safe_series(df, "ultrasonic_error", default=0.0)
        + df["W_radar"] * _safe_series(df, "radar_error", default=0.0)), np.nan)
    
    df["fusion_error_abs"] = df["fusion_error"].abs()

    df["equal_fusion_depth"] = np.where(valid_mask,
        (lidar_values.fillna(0.0)
        + ultrasonic_values.fillna(0.0)
        + radar_values.fillna(0.0))
        / df["sensor_count_valid"].replace(0, np.nan), np.nan)

    df["equal_fusion_error"] = np.where(valid_mask,
        (_safe_series(df, "lidar_error", default=0.0)
        + _safe_series(df, "ultrasonic_error", default=0.0)
        + _safe_series(df, "radar_error", default=0.0))
        / df["sensor_count_valid"].replace(0, np.nan), np.nan)
    
    df["equal_fusion_error_abs"] = df["equal_fusion_error"].abs()

    # Oracle Best Sensor Baseline
    df["oracle_error_abs"] = df[["lidar_error_abs",
    "ultrasonic_error_abs", "radar_error_abs"]].min(axis=1)

    # Data Integrity Checks
    reliability_columns = ["R_lidar", "R_ultrasonic", "R_radar"]

    for column in reliability_columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)): raise ValueError(f"Non-finite values found in {column}")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"Reliability outside [0, 1] in {column}")

    weight_columns = ["W_lidar", "W_ultrasonic", "W_radar"]

    for column in weight_columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)): raise ValueError(f"Non-finite values found in {column}")
        if np.any(values < -EPS): raise ValueError(f"Negative weight found in {column}")

    weight_sum = df[["W_lidar", "W_ultrasonic", "W_radar"]].sum(axis=1).to_numpy(dtype=float)
    valid_sum_mask = np.isclose(weight_sum, 1.0, atol=1e-5) | np.isclose(weight_sum, 0.0, atol=1e-5)
    if not np.all(valid_sum_mask): raise ValueError("Fusion weights do not sum to 1 or 0")

    if np.any((df["fusion_confidence"] < 0.0) | (df["fusion_confidence"] > 1.0)):
        raise ValueError("fusion_confidence outside [0, 1]")

    if np.any((df["scene_complexity"] < 0.0) | (df["scene_complexity"] > 1.0)):
        raise ValueError("scene_complexity outside [0, 1]")

    if np.any(df["fusion_entropy"] > np.log(3) + EPS):
        raise ValueError("fusion_entropy exceeds ln(3)")

    if np.any(df["effective_sensor_count"] > 3 + EPS):
        raise ValueError("effective_sensor_count exceeds 3")

    # Dominance-ratio validation
    dominance = df["dominance_ratio"].to_numpy(dtype=float)
    active_count = df["sensor_count_valid"].to_numpy(dtype=int)

    # 0 active sensors -> dominance ratio must be 0
    zero_active = active_count == 0
    if np.any(np.abs(dominance[zero_active]) > EPS):
        raise ValueError("dominance_ratio must be 0 when no sensors are active")

    # 1 active sensor -> dominance ratio is mathematically infinite
    single_active = active_count == 1
    if np.any(~np.isinf(dominance[single_active])):
        raise ValueError("dominance_ratio must be infinite when exactly one sensor is active")

    # 2 or 3 active sensors -> dominance ratio must be finite and >= 1
    multi_active = active_count >= 2
    if np.any(~np.isfinite(dominance[multi_active])):
        raise ValueError("dominance_ratio must be finite when at least two sensors are active")

    if np.any(dominance[multi_active] < (1.0 - EPS)):
        raise ValueError("dominance_ratio must be >= 1 when at least two sensors are active")

    if np.any((df["reliability_spread"] < 0.0) | (df["reliability_spread"] > 1.0)):
        raise ValueError("reliability_spread outside [0, 1]")

    # Save Dataset
    DATASET_V2.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_V2, index=False)

    # Summary Generation
    wd_err = df["water_context_depth_proxy"] - pd.to_numeric(df["water_depth"], errors="coerce")
    finite_dominance = df["dominance_ratio"].replace([np.inf, -np.inf], np.nan)
    
    summary = {
        "samples": len(df),
        "avg_R_lidar": df["R_lidar"].mean(),
        "avg_R_ultrasonic": df["R_ultrasonic"].mean(),
        "avg_R_radar": df["R_radar"].mean(),
        "avg_R_imu": df["R_imu"].mean(),
        "avg_R_water": df["R_water"].mean(),

        "std_R_lidar": df["R_lidar"].std(),
        "std_R_ultrasonic": df["R_ultrasonic"].std(),
        "std_R_radar": df["R_radar"].std(),

        "median_R_lidar": df["R_lidar"].median(),
        "median_R_ultrasonic": df["R_ultrasonic"].median(),
        "median_R_radar": df["R_radar"].median(),

        "min_R_lidar": df["R_lidar"].min(),
        "min_R_ultrasonic": df["R_ultrasonic"].min(),
        "min_R_radar": df["R_radar"].min(),

        "max_R_lidar": df["R_lidar"].max(),
        "max_R_ultrasonic": df["R_ultrasonic"].max(),
        "max_R_radar": df["R_radar"].max(),

        "p05_R_lidar": df["R_lidar"].quantile(0.05),
        "p05_R_ultrasonic": df["R_ultrasonic"].quantile(0.05),
        "p05_R_radar": df["R_radar"].quantile(0.05),

        "p95_R_lidar": df["R_lidar"].quantile(0.95),
        "p95_R_ultrasonic": df["R_ultrasonic"].quantile(0.95),
        "p95_R_radar": df["R_radar"].quantile(0.95),

        "avg_scene_complexity": df["scene_complexity"].mean(),
        "avg_severity_level": df["severity_level"].mean(),
        "avg_measurement_spread": df["measurement_spread"].mean(),
        "avg_measurement_consistency": df["measurement_consistency"].mean(),
        "avg_water_context_depth_proxy": df["water_context_depth_proxy"].mean(),

        "water_context_proxy_mae": wd_err.abs().mean(),
        "water_context_proxy_rmse": np.sqrt((wd_err**2).mean()),
        "water_context_proxy_bias": wd_err.mean(),

        "avg_reliability_spread": df["reliability_spread"].mean(),
        "avg_reliability_entropy": df["reliability_entropy"].mean(),

        "avg_W_lidar": df["W_lidar"].mean(),
        "avg_W_ultrasonic": df["W_ultrasonic"].mean(),
        "avg_W_radar": df["W_radar"].mean(),

        "std_W_lidar": df["W_lidar"].std(),
        "std_W_ultrasonic": df["W_ultrasonic"].std(),
        "std_W_radar": df["W_radar"].std(),

        "avg_fusion_entropy": df["fusion_entropy"].mean(),
        "avg_effective_sensor_count": df["effective_sensor_count"].mean(),
        "avg_dominance_ratio": finite_dominance.mean(),
        "avg_fusion_confidence": df["fusion_confidence"].mean(),

        "fusion_mae": df["fusion_error_abs"].mean(),
        "fusion_rmse": np.sqrt((df["fusion_error"]**2).mean()),
        "equal_fusion_mae": df["equal_fusion_error_abs"].mean(),
        "equal_fusion_rmse": np.sqrt((df["equal_fusion_error"]**2).mean()),
        "oracle_mae": df["oracle_error_abs"].mean(),

        "dom_lidar_count": int((df["dominant_sensor"] == "lidar").sum()),
        "dom_ultrasonic_count": int((df["dominant_sensor"] == "ultrasonic").sum()),
        "dom_radar_count": int((df["dominant_sensor"] == "radar").sum()),
    }

    summary_df = pd.DataFrame([summary]).round(6)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    # Correlation Matrix
    corr_cols = [
        "water_depth", "water_context_depth_proxy", "ntu", "snr", "clutter_probability",
        "lidar_error", "ultrasonic_error", "radar_error",
        "lidar_error_abs", "ultrasonic_error_abs", "radar_error_abs",
        "R_lidar", "R_ultrasonic", "R_radar",
        "W_lidar", "W_ultrasonic", "W_radar",
        "surface_echo", "bottom_echo", "bottom_confidence", "echo_ambiguity",
        "fusion_entropy", "effective_sensor_count", "dominance_ratio", "fusion_confidence", 
        "scene_complexity", "severity_level", "measurement_spread", "measurement_consistency", 
        "reliability_spread", "reliability_entropy", "fusion_error_abs", "equal_fusion_error_abs"]

    existing_corr_cols = [column for column in corr_cols if column in df.columns]
    corr_pearson = df[existing_corr_cols].corr(numeric_only=True, method="pearson")
    corr_pearson.to_csv(CORR_PEARSON_FILE)
    corr_spearman = df[existing_corr_cols].corr(numeric_only=True, method="spearman")
    corr_spearman.to_csv(CORR_SPEARMAN_FILE)

    # Quantiles
    qcols = [
        "R_lidar", "R_ultrasonic", "R_radar", "R_imu", "R_water",
        "W_lidar", "W_ultrasonic", "W_radar",
        "fusion_entropy", "effective_sensor_count", "dominance_ratio", "fusion_confidence",
        "scene_complexity", "measurement_spread", "measurement_consistency",
        "reliability_spread", "reliability_entropy", "water_context_depth_proxy",
        "fusion_error_abs", "equal_fusion_error_abs", "equal_fusion_depth"]

    qcols = [column for column in qcols if column in df.columns]
    quantiles = df[qcols].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).round(6)
    quantiles.to_csv(QUANTILES_FILE)

    # Metadata Export
    metadata = {
        "dataset_name": "FloodTwin-HIL Dataset V2",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "2.0",
        "scenario_count": len(df),

        "truth_usage_policy": {
            "water_depth": "Latent ground truth; evaluation only",
            "severity": "Scenario metadata; not used for reliability inference",
            "lidar_error": "Evaluation label derived from ground truth",
            "ultrasonic_error": "Evaluation label derived from ground truth",
            "radar_error": "Evaluation label derived from ground truth",
        },

        "observable_provenance": {
            "water_state": "Sensor-derived logic",
            "water_contact": "Sensor-derived logic",
            "surface_echo": "Ultrasonic-observable parameter",
            "bottom_echo": "Ultrasonic-observable parameter",
            "clutter_probability": "Radar-observable parameter",
            "snr": "Radar-observable parameter",
            "ntu": "Turbidity sensor observable parameter",
        },

        "scientific_provenance": {
            "water_context_depth_proxy": (
                "A rule-based observable water-context proxy bounded to 0-20 cm. "
                "It represents a context severity score and is NOT a ground-truth water depth estimator"),
            "measurement_consistency": (
                "Inter-sensor measurement consistency based on relative dispersion. "
                "High consistency does not automatically guarantee high accuracy if sensors fail in a correlated manner"),
            "scene_complexity": (
                "A bounded phenomenological construct whose coefficients represent predefined expert priors rather than fitted parameters"),
            "missing_data_handling": (
                "Missing observable inputs trigger explicit availability flags "
                "and are replaced by predefined neutral engineering priors for "
                "continued computation. These imputed values are not treated as "
                "direct observations and should be interpreted as uncertainty-bearing defaults."),
            "maximum_spread_prior": (
                f"Measurement spread is normalized using a fixed physical prior (NOMINAL_MAX_SPREAD_CM = {NOMINAL_MAX_SPREAD_CM}) "
                "rather than a transductive dataset-wide percentile"),
            "adaptive_fusion_estimate": (
                "An auxiliary linear-weighted depth estimate is calculated to provide baseline validation for the fusion weights relative to an equal-weighted fusion and the oracle best sensor"),
            "sensor_availability_flow": (
                "Unavailable sensors are strictly propagated: sensor missing -> Reliability = 0 -> Weight = 0 -> measurement contributes nothing to the fusion estimate"),
        }
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Console Report
    print()
    print("Reliability-Aware Digital Twin Dataset")
    print()
    print(f"Samples : {len(df)}")
    print(f"Dataset Saved : {DATASET_V2}")
    print(f"Summary Saved : {SUMMARY_FILE}")
    print(f"Metadata Saved : {METADATA_FILE}")
    print(f"Pearson Corr Saved : {CORR_PEARSON_FILE}")
    print(f"Spearman Corr Saved : {CORR_SPEARMAN_FILE}")
    print(f"Quantiles Saved : {QUANTILES_FILE}")
    print()

    print("Reliability Means:")
    print(f" LiDAR      : {df['R_lidar'].mean():.4f}")
    print(f" Ultrasonic : {df['R_ultrasonic'].mean():.4f}")
    print(f" Radar      : {df['R_radar'].mean():.4f}")

    print()
    print("Observable Context:")
    print(f" Mean Water-Depth Context Proxy : {df['water_context_depth_proxy'].mean():.4f} cm")
    print(f" Water Context Proxy MAE        : {summary['water_context_proxy_mae']:.4f} cm")
    print(f" Mean Scene Complexity          : {df['scene_complexity'].mean():.4f}")
    print(f" Mean Measurement Consistency   : {df['measurement_consistency'].mean():.4f}")

    print()
    print("Fusion Accuracy Baseline (Linear Combinations):")
    print(f" Equal Fusion MAE    : {summary['equal_fusion_mae']:.4f} cm")
    print(f" Adaptive Fusion MAE : {summary['fusion_mae']:.4f} cm")
    print(f" Oracle Baseline MAE : {summary['oracle_mae']:.4f} cm")

    print()
    print("Reliability-Error Monotonicity (Pearson):")
    print(f" LiDAR R vs |Error|      : {df['R_lidar'].corr(df['lidar_error_abs']):.4f}")
    print(f" Ultrasonic R vs |Error| : {df['R_ultrasonic'].corr(df['ultrasonic_error_abs']):.4f}")
    print(f" Radar R vs |Error|      : {df['R_radar'].corr(df['radar_error_abs']):.4f}")
    print()

if __name__ == "__main__":
    generate_reliability_dataset()