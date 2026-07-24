import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reliability_engine.lidar_reliability import LidarReliabilityModel
from reliability_engine.ultrasonic_reliability import UltrasonicReliabilityModel
from reliability_engine.radar_reliability import RadarReliabilityModel

try:
    from reliability_engine.reliability_fusion import compute_adaptive_weights, compute_diagnostics
except ImportError:

    def _sanitize_reliability(value):
        value = np.asarray(value, dtype=float)
        value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(value, 0.0, 1.0)

    def compute_adaptive_weights(r_lidar, r_ultra, r_radar):
        priors = np.array([1.05, 1.00, 1.08], dtype=float)

        reliabilities = np.array([
            _sanitize_reliability(r_lidar),
            _sanitize_reliability(r_ultra),
            _sanitize_reliability(r_radar)], dtype=float)

        trust = priors * np.power(reliabilities, 1.20)
        trust = np.maximum(trust, 0.05)
        trust_sum = float(np.sum(trust))

        if not np.isfinite(trust_sum) or trust_sum <= 1e-12:
            return (1/3, 1/3, 1/3)

        weights = trust / trust_sum
        return tuple(float(w) for w in weights)

    def _fusion_entropy(weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 1e-12, 1.0)
        weights = weights / np.sum(weights)
        return float(-np.sum(weights * np.log(weights)))

    def _effective_sensor_count(weights):
        weights = np.asarray(weights, dtype=float)
        denom = float(np.sum(np.square(weights)))
        if denom <= 1e-12:
            return 3.0
        return float(1.0 / denom)

    def compute_diagnostics(r_lidar, r_ultra, r_radar, scene_complexity, sensor_agreement):
        weights = compute_adaptive_weights(r_lidar, r_ultra, r_radar)
        weights_arr = np.asarray(weights, dtype=float)
        sensor_names = ["lidar", "ultrasonic", "radar"]

        dominant_idx = int(np.argmax(weights_arr))
        dominant_sensor = sensor_names[dominant_idx]

        other_mean = float(np.mean(np.delete(weights_arr, dominant_idx))) if len(weights_arr) > 1 else 1e-12
        dominance_ratio = float(np.max(weights_arr) / (other_mean + 1e-12))

        return {
            "weights": tuple(float(w) for w in weights_arr),
            "entropy": _fusion_entropy(weights_arr),
            "effective_sensor_count": _effective_sensor_count(weights_arr),
            "dominant_sensor": dominant_sensor,
            "dominance_ratio": dominance_ratio,
            "confidence": float(np.clip(1.0 - (_fusion_entropy(weights_arr) / np.log(len(weights_arr))), 0.0, 1.0)),
        }

DATASET_V1 = BASE_DIR / "datasets" / "dataset_v1.csv"
DATASET_V2 = BASE_DIR / "datasets" / "dataset_v2.csv"
RESULTS_DIR = BASE_DIR / "results" / "reliability_engine"
SUMMARY_FILE = RESULTS_DIR / "reliability_summary.csv"
CORR_FILE = RESULTS_DIR / "reliability_correlations.csv"
QUANTILES_FILE = RESULTS_DIR / "reliability_quantiles.csv"

RISK_LEVELS = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "medium": 2,
    "high": 3,
    "severe": 3,
    "extreme": 4,
}

def _severity_to_level(value):
    return int(RISK_LEVELS.get(str(value).strip().lower(), 2))

def _safe_series(df, column, default=0.0):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)

def _safe_radar_rcs(df):
    if "rcs" in df.columns:
        return pd.to_numeric(df["rcs"], errors="coerce")
    return None

def _entropy_from_weights(weights):
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 1e-12, 1.0)
    total = float(np.sum(weights))
    if total <= 1e-12: return 0.0
    weights = weights / total
    return float(-np.sum(weights * np.log(weights)))

def _agreement_index(values):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    spread = float(np.std(values))
    denom = float(np.mean(np.abs(values)) + 1e-6)
    agreement = 1.0 - (spread / (denom + spread + 1e-6))
    return float(np.clip(agreement, 0.0, 1.0))

def _clip01(series):
    return pd.to_numeric(series, errors="coerce").clip(0.0, 1.0)

def _bounded_refinement(base, candidate, strength):
    base = np.asarray(base, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return base + strength * (candidate - base)

def _build_scene_complexity(
    water_depth, ntu, clutter, severity_score, measurement_spread, max_spread):
    water_depth = np.asarray(water_depth, dtype=float)
    ntu = np.asarray(ntu, dtype=float)
    clutter = np.asarray(clutter, dtype=float)

    measurement_spread = np.asarray(measurement_spread, dtype=float)
    severity_score = np.asarray(severity_score, dtype=float)

    water_norm = np.clip(water_depth / 20.0, 0, 1)
    water_norm = np.power(water_norm, 1.30)

    ntu_norm = np.clip(ntu / 500.0, 0, 1)
    clutter_norm = np.clip(clutter / 10.0, 0, 1)
    clutter_norm = np.power(clutter_norm, 1.30)

    spread_norm = np.clip(measurement_spread / max_spread, 0, 1)
    severity_norm = np.clip(severity_score, 0.0, 1.0)

    scene = (
        0.28 * water_norm
        + 0.24 * ntu_norm
        + 0.18 * clutter_norm
        + 0.17 * spread_norm
        + 0.13 * severity_norm)
    return np.clip(scene, 0.0, 1.0)

def _calibrate_reliabilities(
    r_lidar_base,
    r_ultrasonic_base,
    r_radar_base,
    water_depth,
    ntu, clutter,
    sensor_agreement,
    severity_score,
    measurement_spread,
    max_spread):

    r_lidar_base = np.asarray(r_lidar_base, dtype=float)
    r_ultrasonic_base = np.asarray(r_ultrasonic_base, dtype=float)
    r_radar_base = np.asarray(r_radar_base, dtype=float)

    water_depth = np.asarray(water_depth, dtype=float)
    ntu = np.asarray(ntu, dtype=float)
    clutter = np.asarray(clutter, dtype=float)
    measurement_spread = np.asarray(measurement_spread, dtype=float)
    sensor_agreement = np.asarray(sensor_agreement, dtype=float)
    severity_score = np.asarray(severity_score, dtype=float)

    water_norm = np.clip(water_depth / 20.0, 0, 1)
    spread_norm = np.clip(measurement_spread / max_spread, 0, 1)

    scene_complexity = _build_scene_complexity(
        water_depth, ntu, clutter, severity_score, measurement_spread, max_spread)
    easy_scene = 1.0 - scene_complexity

    lidar_boost = (0.88
        + 0.27 * easy_scene * sensor_agreement
        + 0.10 * (1.0 - water_norm)
        - 0.03 * scene_complexity)

    ultrasonic_boost = (0.90
        + 0.38 * easy_scene * sensor_agreement
        + 0.14 * (1.0 - water_norm)
        - 0.05 * scene_complexity)

    radar_boost = (0.92
        + 0.06 * scene_complexity
        - 0.15 * easy_scene * sensor_agreement
        - 0.08 * spread_norm)

    agreement_pull = 1.0 + 0.05 * (sensor_agreement - 0.5)

    r_lidar_candidate = np.clip(r_lidar_base * lidar_boost * agreement_pull, 0.06, 1.0)
    r_ultrasonic_candidate = np.clip(r_ultrasonic_base * ultrasonic_boost * agreement_pull, 0.22, 0.985)
    r_radar_candidate = np.clip(
        r_radar_base * radar_boost * (1.0 - 0.02 * easy_scene) * (0.99 + 0.02 * sensor_agreement), 0.10, 1.0)

    r_lidar_final = np.clip(_bounded_refinement(r_lidar_base, r_lidar_candidate, 0.70), 0.06, 1.0)
    r_ultrasonic_final = np.clip(_bounded_refinement(r_ultrasonic_base, r_ultrasonic_candidate, 0.65), 0.22, 0.985)
    r_radar_final = np.clip(_bounded_refinement(r_radar_base, r_radar_candidate, 0.55), 0.10, 1.0)
    return r_lidar_final, r_ultrasonic_final, r_radar_final, scene_complexity

def generate_reliability_dataset():
    df = pd.read_csv(DATASET_V1).copy()
    required_cols = [ "scenario_id", "water_depth", "ntu", "severity", "lidar", "ultrasonic", "radar", "snr" ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset_v1: {missing}")

    lidar_model = LidarReliabilityModel()
    ultrasonic_model = UltrasonicReliabilityModel()
    radar_model = RadarReliabilityModel()

    water_depth = pd.to_numeric(df["water_depth"], errors="coerce").fillna(0.0).clip(lower=0.0)
    ntu = pd.to_numeric(df["ntu"], errors="coerce").fillna(0.0).clip(lower=0.0)
    snr = pd.to_numeric(df["snr"], errors="coerce").fillna(0.0)
    lidar_abs_error = _safe_series(df, "lidar_error", default=0.0).abs()
    ultrasonic_abs_error = _safe_series(df, "ultrasonic_error", default=0.0).abs()
    clutter = _safe_series(df, "clutter_probability", default=np.nan)

    if clutter.isna().all():
        clutter = (0.05 * water_depth).clip(lower=0.0)
    else:
        clutter = clutter.fillna(0.05 * water_depth).clip(lower=0.0)

    measurement_spread = pd.DataFrame({
            "lidar": pd.to_numeric(df["lidar"], errors="coerce"),
            "ultrasonic": pd.to_numeric(df["ultrasonic"], errors="coerce"),
            "radar": pd.to_numeric(df["radar"], errors="coerce"),
        }).std(axis=1).fillna(0.0)

    severity_level = df["severity"].map(_severity_to_level).astype(int)
    severity_score = (severity_level / 4.0).clip(0.0, 1.0)

    sensor_agreement = pd.Series([
        _agreement_index([l, u, r]) for l, u, r in zip(df["lidar"], df["ultrasonic"], df["radar"])],
        index=df.index, dtype=float)

    max_spread = float(np.percentile(measurement_spread, 95))
    max_spread = max(max_spread, 1e-6)

    lidar_noise_sigma = 0.36 + 0.0012 * ntu + 0.0065 * water_depth + 0.008 * lidar_abs_error

    r_lidar_base = pd.Series(
        lidar_model.compute_reliability(ntu=ntu, water_depth=water_depth, noise_sigma=lidar_noise_sigma),
        index=df.index, dtype=float)

    r_ultrasonic_base = pd.Series(
        ultrasonic_model.compute_reliability(
            water_depth=water_depth,
            surface_echo_prob=_clip01(df["surface_echo"]) if "surface_echo" in df.columns else np.zeros(len(df)),
            bottom_echo_prob=_clip01(df["bottom_echo"]) if "bottom_echo" in df.columns else np.ones(len(df))),
        index=df.index, dtype=float)

    radar_rcs = _safe_radar_rcs(df)
    clutter_for_radar = clutter * (1.0 + 0.06 * (measurement_spread / max_spread))

    if radar_rcs is None:
        r_radar_base = pd.Series(
            radar_model.compute_reliability(snr=snr, clutter=clutter_for_radar, water_depth=water_depth),
            index=df.index, dtype=float)
    else:
        radar_rcs = pd.to_numeric(radar_rcs, errors="coerce").fillna(0.0).clip(lower=0.0)
        r_radar_base = pd.Series(
            radar_model.compute_reliability(
                snr=snr,
                clutter=clutter_for_radar,
                water_depth=water_depth,
                rcs=radar_rcs),
            index=df.index, dtype=float)

    r_lidar, r_ultrasonic, r_radar, scene_complexity = _calibrate_reliabilities(
        r_lidar_base.to_numpy(),
        r_ultrasonic_base.to_numpy(),
        r_radar_base.to_numpy(),
        water_depth.to_numpy(),
        ntu.to_numpy(),
        clutter.to_numpy(),
        sensor_agreement.to_numpy(),
        severity_score.to_numpy(),
        measurement_spread.to_numpy(),
        max_spread)

    df["R_lidar_base"] = r_lidar_base.to_numpy(dtype=float)
    df["R_ultrasonic_base"] = r_ultrasonic_base.to_numpy(dtype=float)
    df["R_radar_base"] = r_radar_base.to_numpy(dtype=float)

    df["R_lidar"] = np.asarray(r_lidar, dtype=float)
    df["R_ultrasonic"] = np.asarray(r_ultrasonic, dtype=float)
    df["R_radar"] = np.asarray(r_radar, dtype=float)

    if "R_imu" not in df.columns:
        df["R_imu"] = 0.96
    if "R_water" not in df.columns:
        df["R_water"] = 0.95

    df["severity_level"] = severity_level.astype(int)
    df["severity_score"] = severity_score.astype(float)
    df["scene_complexity"] = scene_complexity.astype(float)
    df["measurement_spread"] = measurement_spread.astype(float)
    df["sensor_agreement"] = sensor_agreement.astype(float)

    df["reliability_spread_base"] = (
        df[["R_lidar_base", "R_ultrasonic_base", "R_radar_base"]].max(axis=1)
        - df[["R_lidar_base", "R_ultrasonic_base", "R_radar_base"]].min(axis=1)).clip(0.0, 1.0)

    df["reliability_spread"] = (
        df[["R_lidar", "R_ultrasonic", "R_radar"]].max(axis=1)
        - df[["R_lidar", "R_ultrasonic", "R_radar"]].min(axis=1)).clip(0.0, 1.0)

    df["reliability_entropy_base"] = [
        _entropy_from_weights(row)
        for row in df[["R_lidar_base", "R_ultrasonic_base", "R_radar_base"]].to_numpy(dtype=float)]

    df["reliability_entropy"] = [
        _entropy_from_weights(row)
        for row in df[["R_lidar", "R_ultrasonic", "R_radar"]].to_numpy(dtype=float)]

    base_weights = []
    base_diagnostics = []
    final_weights = []
    final_diagnostics = []

    for rl_b, ru_b, rr_b, rl, ru, rr, scene in zip(
        df["R_lidar_base"],
        df["R_ultrasonic_base"],
        df["R_radar_base"],
        df["R_lidar"],
        df["R_ultrasonic"],
        df["R_radar"],
        df["scene_complexity"]):

        wb = compute_adaptive_weights(float(rl_b), float(ru_b), float(rr_b), scene_complexity=float(scene))
        db = compute_diagnostics(float(rl_b), float(ru_b), float(rr_b), scene_complexity=float(scene))
        wf = compute_adaptive_weights(float(rl), float(ru), float(rr), scene_complexity=float(scene))
        dfinal = compute_diagnostics(float(rl), float(ru), float(rr), scene_complexity=float(scene))

        base_weights.append(wb)
        base_diagnostics.append(db)
        final_weights.append(wf)
        final_diagnostics.append(dfinal)

    df["W_lidar_base"] = [float(w[0]) for w in base_weights]
    df["W_ultrasonic_base"] = [float(w[1]) for w in base_weights]
    df["W_radar_base"] = [float(w[2]) for w in base_weights]
    df["fusion_entropy_base"] = [float(d["entropy"]) for d in base_diagnostics]
    df["effective_sensor_count_base"] = [float(d["effective_sensor_count"]) for d in base_diagnostics]
    df["dominant_sensor_base"] = [d["dominant_sensor"] for d in base_diagnostics]
    df["dominance_ratio_base"] = [float(d["dominance_ratio"]) for d in base_diagnostics]
    df["fusion_confidence_base"] = [float(d["confidence"]) for d in base_diagnostics]

    df["W_lidar"] = [float(w[0]) for w in final_weights]
    df["W_ultrasonic"] = [float(w[1]) for w in final_weights]
    df["W_radar"] = [float(w[2]) for w in final_weights]
    df["fusion_entropy"] = [float(d["entropy"]) for d in final_diagnostics]
    df["effective_sensor_count"] = [float(d["effective_sensor_count"]) for d in final_diagnostics]
    df["dominant_sensor"] = [d["dominant_sensor"] for d in final_diagnostics]
    df["dominance_ratio"] = [float(d["dominance_ratio"]) for d in final_diagnostics]
    df["fusion_confidence"] = [float(d["confidence"]) for d in final_diagnostics]

    df["dominant_sensor_base"] = df["dominant_sensor_base"].astype("string")
    df["dominant_sensor"] = df["dominant_sensor"].astype("string")

    DATASET_V2.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_V2, index=False)

    summary = {
        "samples": len(df),
        "avg_R_lidar_base": df["R_lidar_base"].mean(),
        "avg_R_ultrasonic_base": df["R_ultrasonic_base"].mean(),
        "avg_R_radar_base": df["R_radar_base"].mean(),
        "avg_R_lidar": df["R_lidar"].mean(),
        "avg_R_ultrasonic": df["R_ultrasonic"].mean(),
        "avg_R_radar": df["R_radar"].mean(),
        "avg_R_imu": df["R_imu"].mean(),
        "avg_R_water": df["R_water"].mean(),
        "std_R_lidar_base": df["R_lidar_base"].std(),
        "std_R_ultrasonic_base": df["R_ultrasonic_base"].std(),
        "std_R_radar_base": df["R_radar_base"].std(),
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
        "avg_sensor_agreement": df["sensor_agreement"].mean(),
        "avg_reliability_spread_base": df["reliability_spread_base"].mean(),
        "avg_reliability_spread": df["reliability_spread"].mean(),
        "avg_reliability_entropy_base": df["reliability_entropy_base"].mean(),
        "avg_reliability_entropy": df["reliability_entropy"].mean(),
        "avg_W_lidar_base": df["W_lidar_base"].mean(),
        "avg_W_ultrasonic_base": df["W_ultrasonic_base"].mean(),
        "avg_W_radar_base": df["W_radar_base"].mean(),
        "avg_W_lidar": df["W_lidar"].mean(),
        "avg_W_ultrasonic": df["W_ultrasonic"].mean(),
        "avg_W_radar": df["W_radar"].mean(),
        "std_W_lidar_base": df["W_lidar_base"].std(),
        "std_W_ultrasonic_base": df["W_ultrasonic_base"].std(),
        "std_W_radar_base": df["W_radar_base"].std(),
        "std_W_lidar": df["W_lidar"].std(),
        "std_W_ultrasonic": df["W_ultrasonic"].std(),
        "std_W_radar": df["W_radar"].std(),
        "avg_fusion_entropy_base": df["fusion_entropy_base"].mean(),
        "avg_effective_sensor_count_base": df["effective_sensor_count_base"].mean(),
        "avg_dominance_ratio_base": df["dominance_ratio_base"].mean(),
        "avg_fusion_confidence_base": df["fusion_confidence_base"].mean(),
        "avg_fusion_entropy": df["fusion_entropy"].mean(),
        "avg_effective_sensor_count": df["effective_sensor_count"].mean(),
        "avg_dominance_ratio": df["dominance_ratio"].mean(),
        "avg_fusion_confidence": df["fusion_confidence"].mean(),
        "dom_lidar_count_base": int((df["dominant_sensor_base"] == "lidar").sum()),
        "dom_ultrasonic_count_base": int((df["dominant_sensor_base"] == "ultrasonic").sum()),
        "dom_radar_count_base": int((df["dominant_sensor_base"] == "radar").sum()),
        "dom_lidar_count": int((df["dominant_sensor"] == "lidar").sum()),
        "dom_ultrasonic_count": int((df["dominant_sensor"] == "ultrasonic").sum()),
        "dom_radar_count": int((df["dominant_sensor"] == "radar").sum()),
    }

    summary_df = pd.DataFrame([summary]).round(6)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    corr_cols = [
        "water_depth",
        "ntu",
        "snr",
        "clutter_probability",
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
        "surface_echo",
        "bottom_echo",
        "fusion_entropy_base",
        "effective_sensor_count_base",
        "dominance_ratio_base",
        "fusion_confidence_base",
        "fusion_entropy",
        "effective_sensor_count",
        "dominance_ratio",
        "fusion_confidence",
        "scene_complexity",
        "severity_level",
        "measurement_spread",
        "sensor_agreement",
        "reliability_spread_base",
        "reliability_spread",
        "reliability_entropy_base",
        "reliability_entropy" ]

    existing_corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[existing_corr_cols].corr(numeric_only=True, method="pearson")
    corr.to_csv(CORR_FILE)

    qcols = [
        "R_lidar_base",
        "R_ultrasonic_base",
        "R_radar_base",
        "R_lidar",
        "R_ultrasonic",
        "R_radar",
        "R_imu",
        "R_water",
        "W_lidar",
        "W_ultrasonic",
        "W_radar",
        "fusion_entropy_base",
        "effective_sensor_count_base",
        "dominance_ratio_base",
        "fusion_confidence_base",
        "fusion_entropy",
        "effective_sensor_count",
        "dominance_ratio",
        "fusion_confidence",
        "scene_complexity",
        "measurement_spread",
        "sensor_agreement",
        "reliability_spread_base",
        "reliability_spread",
        "reliability_entropy_base",
        "reliability_entropy" ]

    qcols = [c for c in qcols if c in df.columns]
    quantiles = df[qcols].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).round(6)
    quantiles.to_csv(QUANTILES_FILE)

    print()
    print("FloodTwin-HIL Reliability Dataset")
    print()
    print(f"Samples : {len(df)}")
    print(f"Dataset Saved : {DATASET_V2}")
    print(f"Summary Saved : {SUMMARY_FILE}")
    print(f"Correlations Saved : {CORR_FILE}")
    print(f"Quantiles Saved : {QUANTILES_FILE}")
    print()

if __name__ == "__main__":
    generate_reliability_dataset()