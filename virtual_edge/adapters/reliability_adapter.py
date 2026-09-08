from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reliability_engine.lidar_reliability import LidarReliabilityModel
from reliability_engine.radar_reliability import RadarReliabilityModel
from reliability_engine.reliability_fusion import AdaptiveReliabilityWeighting
from reliability_engine.ultrasonic_reliability import UltrasonicReliabilityModel

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
    out: Dict[str, float] = {}
    for key in keys:
        out[key] = _safe_float(value.get(key, default), default)
    return out

def _depth_metrics(depth_values: np.ndarray, depth_span_cm: float = 30.0) -> Dict[str, float]:
    values = np.asarray(depth_values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return {
            "measurement_spread": 0.0,
            "sensor_agreement": 0.5,
            "consensus_depth": 0.0,
            "valid_measurement_count": 0.0,
        }

    spread = float(np.std(finite)) if finite.size > 1 else 0.0
    consensus = float(np.median(finite))
    agreement = float(np.clip(1.0 - (spread / max(depth_span_cm, 1e-6)), 0.0, 1.0))

    return {
        "measurement_spread": spread,
        "sensor_agreement": agreement,
        "consensus_depth": consensus,
        "valid_measurement_count": float(finite.size),
    }

def _reliability_spread(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return float(np.std(arr))

@dataclass
class ReliabilityAdapter:

    seed: int = 42
    depth_span_cm: float = 30.0

    def __post_init__(self) -> None:
        self.lidar = LidarReliabilityModel()
        self.radar = RadarReliabilityModel()
        self.ultrasonic = UltrasonicReliabilityModel()
        self.fusion = AdaptiveReliabilityWeighting()

    def compute(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        packet = dict(packet or {})

        # ---------------- Inputs from the Base Sensors ----------------
        ntu = _safe_float(packet.get("ntu_observed", packet.get("ntu", 0.0)), 0.0)
        water_depth = _safe_float(packet.get("water_context_depth_proxy", 0.0), 0.0)
        lidar_noise_sigma = _safe_float(packet.get("lidar_noise_sigma", 1.0))

        radar_snr = _safe_float(packet.get("radar_snr", 20.0))
        radar_clutter = _safe_float(packet.get("radar_clutter", 0.0))
        radar_rcs = _safe_float(packet.get("radar_rcs", 1.0))

        ultrasonic_surface = _safe_float(packet.get("ultrasonic_surface_echo", 0.5))
        ultrasonic_bottom = _safe_float(packet.get("ultrasonic_bottom_echo", 0.5))
        ultrasonic_measurement_type = str(packet.get("ultrasonic_measurement_type", "Unknown"))
        ultrasonic_bottom_confidence = _safe_float(packet.get("ultrasonic_bottom_confidence", 0.5))
        ultrasonic_ambiguity = _safe_float(packet.get("ultrasonic_echo_ambiguity", 0.5))

        scene_complexity = _safe_float(packet.get("scene_complexity", 0.0))

        lidar_depth = _safe_float(packet.get("lidar", np.nan), np.nan)
        ultrasonic_depth = _safe_float(packet.get("ultrasonic", np.nan), np.nan)
        radar_depth = _safe_float(packet.get("radar", np.nan), np.nan)

        imu_prior = _safe_float(packet.get("imu_reliability_prior", 0.95))
        turbidity_prior = _safe_float(packet.get("turbidity_reliability_prior", 0.95))
        water_prior = _safe_float(packet.get("water_contact_reliability_prior", 0.95))

        sensor_health = _safe_dict(packet.get("sensor_health"),
            keys=("lidar", "ultrasonic", "radar", "imu", "turbidity", "water_contact"), default=1.0)
        sensor_availability = _safe_dict(packet.get("sensor_availability"),
            keys=("lidar", "ultrasonic", "radar", "imu", "turbidity", "water_contact"), default=1.0)

        # ---------------- Active Sensor Mask ----------------
        active_mask = np.array([
            np.isfinite(lidar_depth) and sensor_health["lidar"] > 0.0 and sensor_availability["lidar"] > 0.0,
            np.isfinite(ultrasonic_depth) and sensor_health["ultrasonic"] > 0.0 and sensor_availability["ultrasonic"] > 0.0,
            np.isfinite(radar_depth) and sensor_health["radar"] > 0.0 and sensor_availability["radar"] > 0.0], dtype=bool)

        # ---------------- Reliability Computation ----------------
        r_lidar = float(self.lidar.compute_reliability(ntu, water_depth, lidar_noise_sigma))
        r_radar = float(self.radar.compute_reliability(radar_snr, radar_clutter, water_depth, rcs=radar_rcs))
        r_ultrasonic = float(self.ultrasonic.compute_reliability(
            water_depth=water_depth,
            surface_echo_prob=ultrasonic_surface,
            bottom_echo_prob=ultrasonic_bottom))

        r_imu = float(np.clip(imu_prior, 0.0, 1.0))
        r_turbidity = float(np.clip(turbidity_prior, 0.0, 1.0))
        r_water = float(np.clip(water_prior, 0.0, 1.0))

        reliability_vector = np.array([r_lidar, r_ultrasonic, r_radar], dtype=float)
        reliability_spread = _reliability_spread(reliability_vector)

        # ---------------- Measurement Consistency Metrics ----------------
        depths = np.array([lidar_depth, ultrasonic_depth, radar_depth], dtype=float)
        depths = depths[active_mask]
        depth_metrics = _depth_metrics(depths, depth_span_cm=self.depth_span_cm)

        # ---------------- Adaptive Fusion ----------------
        weights = self.fusion.compute_adaptive_weights(
            r_lidar=r_lidar, r_ultra=r_ultrasonic, r_radar=r_radar,
            scene_complexity=scene_complexity)

        diagnostics = self.fusion.fusion_diagnostics(weights,
            reliabilities=[r_lidar, r_ultrasonic, r_radar],
            scene_complexity=scene_complexity)

        weight_vector = np.array(weights, dtype=float)
        weight_sum = float(np.sum(weight_vector))

        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Adaptive fusion weights must sum to 1\n"
                f"Current sum={weight_sum:.6f}")

        ultrasonic_echo_confidence = float(
            self.ultrasonic.echo_confidence(
                surface_echo_prob=ultrasonic_surface,
                bottom_echo_prob=ultrasonic_bottom))

        return {
            "r_lidar": r_lidar,
            "r_radar": r_radar,
            "r_ultrasonic": r_ultrasonic,
            "r_imu": r_imu,
            "r_turbidity": r_turbidity,
            "r_water": r_water,
            
            "reliability_vector": reliability_vector.tolist(),
            "measurement_vector": depths.tolist(),
            "weight_vector": weight_vector.tolist(),
            "weight_sum": weight_sum,
            
            "weights": {
                "lidar": float(weight_vector[0]),
                "ultrasonic": float(weight_vector[1]),
                "radar": float(weight_vector[2]),
            },

            "fusion_confidence": float(diagnostics["confidence"]),
            "dominant_sensor": diagnostics["dominant_sensor"],
            "dominance_ratio": float(diagnostics["dominance_ratio"]),
            "weight_entropy": float(diagnostics["entropy"]),
            "effective_sensor_count": float(diagnostics["effective_sensor_count"]),
            "weight_variance": float(diagnostics["weight_variance"]),
            
            "scene_complexity": scene_complexity,
            "sensor_agreement": float(depth_metrics["sensor_agreement"]),
            "measurement_spread": float(depth_metrics["measurement_spread"]),
            "consensus_depth": float(depth_metrics["consensus_depth"]),
            "valid_measurement_count": float(depth_metrics["valid_measurement_count"]),
            "reliability_spread": float(reliability_spread),
            "active_mask": active_mask.tolist(),
            "sensor_health": dict(sensor_health),
            "sensor_availability": dict(sensor_availability),
            
            "ntu": ntu,
            "water_depth": water_depth,
            "lidar_noise_sigma": lidar_noise_sigma,
            "radar_snr": radar_snr,
            "radar_clutter": radar_clutter,
            "radar_rcs": radar_rcs,
            "ultrasonic_surface_echo": ultrasonic_surface,
            "ultrasonic_bottom_echo": ultrasonic_bottom,
            "ultrasonic_bottom_confidence": ultrasonic_bottom_confidence,
            "ultrasonic_echo_ambiguity": ultrasonic_ambiguity,
            "ultrasonic_measurement_type": ultrasonic_measurement_type,
            "ultrasonic_echo_confidence": ultrasonic_echo_confidence,
            
            "imu_reliability_prior": imu_prior,
            "turbidity_reliability_prior": turbidity_prior,
            "water_contact_reliability_prior": water_prior,
            
            "metadata": {
                "engine": "ReliabilityAdapter",
                "seed": self.seed,
                "depth_span_cm": self.depth_span_cm,
            },
        }