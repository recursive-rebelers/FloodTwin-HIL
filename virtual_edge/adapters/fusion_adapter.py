from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from fusion_engine.adaptive_fusion import AdaptiveFusionEngine
from fusion_engine.bayesian_fusion import BayesianFusionCore
from fusion_engine.uncertainty import UncertaintyQuantification

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)

def _safe_dict(value: Any, keys: Iterable[str], default: float = 1.0) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {k: float(default) for k in keys}
    return {k: _safe_float(value.get(k, default), default) for k in keys}

def _coerce_active_mask(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=bool).reshape(-1)
        if arr.shape != (3,):
            return None
        return arr
    except Exception:
        return None

def _as_finite_measurement(value: Any, fallback: float) -> float:
    x = _safe_float(value, np.nan)
    if np.isfinite(x):
        return float(x)
    return float(fallback)

@dataclass
class FusionAdapter:

    depth_min_cm: float = 0.0
    depth_max_cm: float = 30.0
    resolution: int = 300

    def __post_init__(self) -> None:
        self.core = BayesianFusionCore(
            depth_min=self.depth_min_cm,
            depth_max=self.depth_max_cm,
            resolution=self.resolution)
        self.engine = AdaptiveFusionEngine(core=self.core)
        self.uq = UncertaintyQuantification()

    def fuse(self, packet: Dict[str, Any], reliability: Dict[str, Any]) -> Dict[str, Any]:
        packet = dict(packet or {})
        reliability = dict(reliability or {})

        # ---------------------------
        # Measurements from the Packet
        # ---------------------------
        z_lidar_raw = packet.get("lidar", np.nan)
        z_ultra_raw = packet.get("ultrasonic", np.nan)
        z_radar_raw = packet.get("radar", np.nan)

        z_lidar = _as_finite_measurement(z_lidar_raw, np.nan)
        z_ultra = _as_finite_measurement(z_ultra_raw, np.nan)
        z_radar = _as_finite_measurement(z_radar_raw, np.nan)

        measurement_vector = np.asarray([z_lidar, z_ultra, z_radar], dtype=float)

        # ---------------------------
        # Reliability Inputs
        # ---------------------------
        r_lidar = _safe_float(reliability.get("r_lidar", 0.8), 0.8)
        r_ultra = _safe_float(reliability.get("r_ultrasonic", 0.8), 0.8)
        r_radar = _safe_float(reliability.get("r_radar", 0.8), 0.8)

        reliability_vector = np.asarray([r_lidar, r_ultra, r_radar], dtype=float)

        scene_complexity = _safe_float(packet.get("scene_complexity",
            reliability.get("scene_complexity", 0.0)), 0.0)
        water_depth = _safe_float(packet.get("water_context_depth_proxy",
            reliability.get("water_depth", 0.0)), 0.0)
        ntu = _safe_float(packet.get("ntu_observed",
            reliability.get("ntu", 0.0)), 0.0)

        # Prefer upstream-computed context if available
        measurement_spread = _safe_float(reliability.get("measurement_spread", np.nan), np.nan)
        sensor_agreement = _safe_float(reliability.get("sensor_agreement", np.nan), np.nan)
        reliability_spread = _safe_float(reliability.get("reliability_spread", np.nan), np.nan)
        effective_sensor_count = _safe_float(reliability.get("effective_sensor_count", np.nan), np.nan)
        dominance_ratio = _safe_float(reliability.get("dominance_ratio", np.nan), np.nan)
        adaptive_confidence_seed = _safe_float(
            reliability.get("fusion_confidence", reliability.get("adaptive_confidence", 0.5)), 0.5)

        if not np.isfinite(measurement_spread):
            measurement_spread = float(np.std(measurement_vector))

        if not np.isfinite(sensor_agreement):
            depth_span = max(self.depth_max_cm - self.depth_min_cm, 1e-6)
            sensor_agreement = float(np.clip(1.0 - (measurement_spread / depth_span), 0.0, 1.0))

        if not np.isfinite(reliability_spread):
            reliability_spread = float(np.std(reliability_vector))

        if not np.isfinite(effective_sensor_count):
            effective_sensor_count = float(3.0 / (1.0 + 3.0 * reliability_spread))

        if not np.isfinite(dominance_ratio):
            mean_rel = float(np.mean(reliability_vector) + 1e-12)
            dominance_ratio = float(np.max(reliability_vector) / mean_rel)

        # ---------------------------
        # Health / Availability / Active Mask
        # ---------------------------
        sensor_health = _safe_dict(reliability.get("sensor_health", packet.get("sensor_health")),
            keys=("lidar", "ultrasonic", "radar"), default=1.0)
        sensor_availability = _safe_dict(reliability.get("sensor_availability", packet.get("sensor_availability")),
            keys=("lidar", "ultrasonic", "radar"), default=1.0)

        active_mask = _coerce_active_mask(reliability.get("active_mask", packet.get("active_mask")))
        if active_mask is None:
            active_mask = np.array([
                np.isfinite(z_lidar) and sensor_health["lidar"] > 0.0 and sensor_availability["lidar"] > 0.0,
                np.isfinite(z_ultra) and sensor_health["ultrasonic"] > 0.0 and sensor_availability["ultrasonic"] > 0.0,
                np.isfinite(z_radar) and sensor_health["radar"] > 0.0 and sensor_availability["radar"] > 0.0], dtype=bool)

        # ---------------------------
        # Fuse API From Fusion Eengine
        # ---------------------------
        fuse_result = self.engine.fuse(
            z_lidar=z_lidar, z_ultra=z_ultra, z_radar=z_radar,
            r_lidar=r_lidar, r_ultra=r_ultra, r_radar=r_radar,
            scene_complexity=scene_complexity,
            sensor_agreement=sensor_agreement,
            reliability_spread=reliability_spread,
            effective_sensor_count=effective_sensor_count,
            dominance_ratio=dominance_ratio,
            confidence=adaptive_confidence_seed,
            measurement_spread=measurement_spread,
            water_depth=water_depth, ntu=ntu,
            active_mask=active_mask.tolist(),
            sensor_health={
                "lidar": sensor_health["lidar"],
                "ultrasonic": sensor_health["ultrasonic"],
                "radar": sensor_health["radar"]})

        posterior = fuse_result.get("posterior", None)
        if posterior is None:
            raise RuntimeError("Adaptive fusion did not return a posterior distribution")

        posterior = np.asarray(posterior, dtype=float)
        posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)
        if posterior.shape != self.core.depths.shape:
            raise RuntimeError("Posterior shape does not match Bayesian depth grid")

        mass_sum = float(np.sum(posterior) * self.core.delta_x)
        if not np.isfinite(mass_sum) or mass_sum <= 1e-12:
            raise RuntimeError("Posterior distribution is invalid or degenerate")

        posterior_summary = self.core.summary(posterior, mass_level=0.95)
        uncertainty = self.uq.compute(self.core.depths, posterior, self.core.delta_x, mass_level=0.95)

        weight_vector = np.asarray([
            float(fuse_result.get("weights", {}).get("lidar", reliability.get("weights", {}).get("lidar", 1.0 / 3.0))),
            float(fuse_result.get("weights", {}).get("ultrasonic", reliability.get("weights", {}).get("ultrasonic", 1.0 / 3.0))),
            float(fuse_result.get("weights", {}).get("radar", reliability.get("weights", {}).get("radar", 1.0 / 3.0)))], dtype=float)

        weight_sum = float(np.sum(weight_vector))
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise RuntimeError(
                "Fusion engine returned invalid weights\n"
                f"Expected sum=1.0, got {weight_sum:.6f}")

        out = dict(fuse_result)
        out.update({
            "core": self.core,
            "engine": self.engine,
            "measurement_vector": measurement_vector.tolist(),
            "reliability_vector": reliability_vector.tolist(),
            "weight_vector": weight_vector.tolist(),
            "weight_sum": weight_sum,
            "active_mask": active_mask.tolist(),
            "sensor_health": dict(sensor_health),
            "sensor_availability": dict(sensor_availability),
            "measurement_spread": float(measurement_spread),
            "sensor_agreement": float(sensor_agreement),
            "reliability_spread": float(reliability_spread),
            "effective_sensor_count_est": float(effective_sensor_count),
            "dominance_ratio_est": float(dominance_ratio),
            "adaptive_confidence_seed": float(adaptive_confidence_seed),
            "posterior_summary": posterior_summary,
            "uncertainty": uncertainty,
            "posterior_mass_sum": float(mass_sum),
            "expected_depth": float(posterior_summary["expected_depth"]),
            "map_depth": float(posterior_summary["map_depth"]),
            "posterior_confidence": float(posterior_summary["confidence"]),
            "ci_lower": float(posterior_summary["ci_lower"]),
            "ci_upper": float(posterior_summary["ci_upper"]),
            "ci_width_95": float(posterior_summary.get(
                "ci_width_95", posterior_summary["ci_upper"] - posterior_summary["ci_lower"])),
            "posterior_peak": float(posterior_summary["posterior_peak"]),
            "posterior_entropy": float(posterior_summary["entropy"]),
            "posterior_variance": float(posterior_summary["variance"]),
            "posterior_std": float(posterior_summary["std"]),
            "uncertainty_confidence": float(uncertainty["confidence"]),
            "uncertainty_map_depth": float(uncertainty["map_depth"]),
            "uncertainty_expected_depth": float(uncertainty["expected_depth"]),
            "uncertainty_ci_lower": float(uncertainty["ci_lower"]),
            "uncertainty_ci_upper": float(uncertainty["ci_upper"]),
            "metadata": {
                "fusion_engine": "Adaptive Bayesian Fusion",
                "depth_min_cm": self.depth_min_cm,
                "depth_max_cm": self.depth_max_cm,
                "resolution": self.resolution,
            }})

        return out