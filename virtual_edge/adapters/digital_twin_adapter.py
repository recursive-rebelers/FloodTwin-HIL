from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from digital_twin.imu_model import IMUModel
from digital_twin.lidar_model import LidarModel
from digital_twin.radar_model import RadarModel
from digital_twin.turbidity_model import TurbidityModel
from digital_twin.ultrasonic_model import UltrasonicModel
from digital_twin.water_contact_model import FSIR01Model

def _effective_lidar_noise_sigma(base_noise_sigma: float, water_depth: float, ntu: float) -> float:
    water_attenuation = float(np.exp(0.055 * max(float(water_depth), 0.0)))
    turbidity_scattering = float(np.exp(0.003 * max(float(ntu), 0.0)))
    return float(base_noise_sigma) * water_attenuation * turbidity_scattering

def _scene_complexity(water_depth: float, ntu: float,
    vehicle_speed: float, weather: str = "", lighting: str = "") -> float:
    water = np.clip(float(water_depth) / 20.0, 0.0, 1.0)
    turb = np.clip(float(ntu) / 500.0, 0.0, 1.0)
    speed = np.clip(float(vehicle_speed) / 60.0, 0.0, 1.0)

    weather_penalty = 1.0 if str(weather).strip().lower() in {"rainy", "storm", "foggy"} else 0.0
    lighting_penalty = 1.0 if str(lighting).strip().lower() in {"night", "low", "poor"} else 0.0
    score = (0.35 * water + 0.30 * turb + 0.15 * speed + 0.10 * weather_penalty + 0.10 * lighting_penalty)
    return float(np.clip(score, 0.0, 1.0))

@dataclass
class DigitalTwinAdapter:

    seed: int = 42
    include_metadata: bool = True

    def __post_init__(self) -> None:
        self.lidar = LidarModel(seed=self.seed)
        self.radar = RadarModel(seed=self.seed)
        self.ultrasonic = UltrasonicModel(seed=self.seed)
        self.turbidity = TurbidityModel(seed=self.seed)
        self.water_contact = FSIR01Model()
        self.sensor_metadata = self._build_sensor_metadata() if self.include_metadata else {}

        self.sensor_health = {
            "lidar": 1.0,
            "radar": 1.0,
            "ultrasonic": 1.0,
            "imu": 1.0,
            "turbidity": 1.0,
            "water_contact": 1.0,
        }

        self.sensor_availability = {
            "lidar": 1.0,
            "radar": 1.0,
            "ultrasonic": 1.0,
            "imu": 1.0,
            "turbidity": 1.0,
            "water_contact": 1.0,
        }

    def _build_sensor_metadata(self) -> Dict[str, Any]:
        water_contact_meta = {
            "sensor": "FSIR01",
            "model": "Binary water-contact detector",
            "states": ["Dry", "Shallow Puddle", "Flooded Pothole"],
        }

        return {
            "lidar": self.lidar.metadata(),
            "radar": self.radar.metadata(),
            "ultrasonic": self.ultrasonic.metadata(),
            "turbidity": self.turbidity.metadata(),

            "imu": {
                "sensor": "MPU6050",
                "model": "Physics-based road vibration / pitch / roll estimator",
                "source": "IMUModel",
            }, "water_contact": water_contact_meta,
        }

    def simulate(self, scenario: Dict[str, Any], frame_idx: int = 0) -> Dict[str, Any]:
        scenario = dict(scenario or {})
        road_type = str(scenario.get("road_type", "Asphalt"))
        road_environment = str(scenario.get("road_environment", "Urban"))
        weather = str(scenario.get("weather", "Clear"))
        lighting = str(scenario.get("lighting", "Day"))
        vehicle_speed = float(scenario.get("vehicle_speed", 0.0))

        # Validate depth fields if both are provided
        if "true_depth" in scenario and "pothole_depth" in scenario:
            scenario_true_depth = float(scenario["true_depth"])
            scenario_pothole_depth = float(scenario["pothole_depth"])
            tolerance = 1e-6

            if abs(scenario_true_depth - scenario_pothole_depth) > tolerance:
                raise ValueError(
                    "Scenario contains inconsistent depth values: "
                    f"true_depth={scenario_true_depth} cm, "
                    f"pothole_depth={scenario_pothole_depth} cm")

        true_depth = float(scenario.get("pothole_depth", scenario.get("true_depth", 0.0)))
        water_depth = float(scenario.get("water_depth", 0.0))
        true_ntu = float(scenario.get("true_ntu", scenario.get("ntu", 0.0)))

        # ---------------- LiDAR ----------------
        lidar_distance = float(self.lidar.generate(true_depth, water_depth, true_ntu))
        lidar_noise_sigma = float(_effective_lidar_noise_sigma(self.lidar.base_noise_sigma, water_depth, true_ntu))
        lidar_reliability = float(self.lidar.reliability(water_depth, true_ntu))
        lidar_degradation = float(self.lidar.degradation_factor(water_depth, true_ntu))

        # ---------------- Radar ----------------
        radar_packet = self.radar.generate(true_depth, water_depth, true_ntu)
        radar_distance = float(radar_packet["distance"])
        radar_snr = float(radar_packet["snr"])
        radar_rcs = float(radar_packet["rcs"])
        radar_clutter = float(radar_packet["clutter_probability"])
        radar_noise_sigma = float(self.radar.base_noise_sigma + water_depth * 0.01 + true_ntu / 5000.0)
        radar_reliability = float(self.radar.reliability(water_depth, true_ntu))
        radar_degradation = float(np.clip(1.0 - radar_reliability, 0.0, 1.0))

        # ---------------- Ultrasonic ----------------
        ultrasonic_packet = self.ultrasonic.generate(true_depth, water_depth)
        ultrasonic_distance = float(ultrasonic_packet["distance"])
        ultrasonic_surface_echo = float(ultrasonic_packet["surface_echo"])
        ultrasonic_bottom_echo = float(ultrasonic_packet["bottom_echo"])
        ultrasonic_bottom_confidence = float(ultrasonic_packet["bottom_confidence"])
        ultrasonic_echo_ambiguity = float(ultrasonic_packet["echo_ambiguity"])
        ultrasonic_measurement_type = str(ultrasonic_packet["measurement_type"])
        ultrasonic_surface_sigma = float(ultrasonic_packet["surface_sigma"])
        ultrasonic_bottom_sigma = float(ultrasonic_packet["bottom_sigma"])

        ultrasonic_reliability = float(
            self.ultrasonic.reliability(
                water_depth=water_depth,
                measurement_type=ultrasonic_measurement_type,
                surface_echo=ultrasonic_surface_echo,
                bottom_echo=ultrasonic_bottom_echo,
                bottom_confidence=ultrasonic_bottom_confidence,
                echo_ambiguity=ultrasonic_echo_ambiguity))
        ultrasonic_degradation = float(np.clip(1.0 - ultrasonic_reliability, 0.0, 1.0))

        # ---------------- IMU ----------------
        imu_packet = IMUModel(
            road_type=road_type,
            road_environment=road_environment,
            vehicle_speed=vehicle_speed,
            pothole_depth=true_depth,
            water_depth=water_depth,
            base_pitch=0.0, base_roll=0.0)

        imu_reliability = float(imu_packet["reliability"])
        imu_pitch = float(imu_packet["pitch"])
        imu_roll = float(imu_packet["roll"])
        imu_acceleration = float(imu_packet["acceleration"])
        imu_degradation = float(np.clip(1.0 - imu_reliability, 0.0, 1.0))

        # ---------------- Turbidity ----------------
        turbidity_value = float(self.turbidity.generate(true_ntu))
        turbidity_category = self.turbidity.classify(turbidity_value)
        ntu_observed = float(turbidity_value)
        ntu_available = bool(np.isfinite(ntu_observed)
        and self.sensor_availability["turbidity"] > 0.0)
        turbidity_reliability = float(self.turbidity.reliability(true_ntu))
        turbidity_degradation = float(self.turbidity.degradation_factor(true_ntu))

        # ---------------- Water Contact ----------------
        water_packet = self.water_contact.detect(water_depth)
        water_state = str(water_packet["state"])
        water_present = bool(water_packet["water_present"])
        water_confidence = float(water_packet["confidence"])
        water_reliability = float(water_packet["reliability"])
        water_degradation = float(np.clip(1.0 - water_reliability, 0.0, 1.0))

        # ---------------- Observable Water Context ----------------
        water_context = float(water_present)
        water_context_depth_proxy = float(np.clip((
            ultrasonic_surface_echo - 0.2) / 0.02, 0.0, 20.0))

        water_context_available = bool(np.isfinite(water_context_depth_proxy)
            and self.sensor_availability["water_contact"] > 0.0
            and self.sensor_availability["ultrasonic"] > 0.0)

        # ---------------- Scene Complexity ----------------
        scene_complexity = _scene_complexity(
            water_depth=water_context_depth_proxy,
            ntu=ntu_observed,
            vehicle_speed=vehicle_speed,
            weather=weather,
            lighting=lighting)

        return {
            "frame_idx": int(frame_idx),
            "scenario_id": scenario.get("scenario_id", f"SC{frame_idx + 1:03d}"),
            "road_type": road_type,
            "road_environment": road_environment,
            "weather": weather,
            "lighting": lighting,
            "vehicle_speed": vehicle_speed,
            "pothole_depth": true_depth,
            "water_depth": water_depth,
            "ntu": true_ntu,
            "true_depth": true_depth,
            "true_ntu": true_ntu,
            "scene_complexity": scene_complexity,

            "ntu_observed": ntu_observed,
            "ntu_available": int(ntu_available),

            "water_context": water_context,
            "water_context_depth_proxy": water_context_depth_proxy,
            "water_context_available": int(water_context_available),
            
            "lidar": lidar_distance,
            "radar": radar_distance,
            "ultrasonic": ultrasonic_distance,
            "imu_acceleration": imu_acceleration,
            "imu_pitch": imu_pitch,
            "imu_roll": imu_roll,
            "turbidity": turbidity_value,
            "water_contact": int(water_present),
            
            "lidar_reliability_prior": lidar_reliability,
            "radar_reliability_prior": radar_reliability,
            "ultrasonic_reliability_prior": ultrasonic_reliability,
            "imu_reliability_prior": imu_reliability,
            "turbidity_reliability_prior": turbidity_reliability,
            "water_contact_reliability_prior": water_reliability,
            "sensor_health": dict(self.sensor_health),
            "sensor_availability": dict(self.sensor_availability),
            
            "lidar_noise_sigma": lidar_noise_sigma,
            "lidar_degradation_factor": lidar_degradation,
            "radar_noise_sigma": radar_noise_sigma,
            "radar_snr": radar_snr,
            "radar_rcs": radar_rcs,
            "radar_clutter": radar_clutter,
            "radar_degradation_factor": radar_degradation,
            "ultrasonic_surface_echo": ultrasonic_surface_echo,
            "ultrasonic_bottom_echo": ultrasonic_bottom_echo,
            "ultrasonic_bottom_confidence": ultrasonic_bottom_confidence,
            "ultrasonic_echo_ambiguity": ultrasonic_echo_ambiguity,
            "ultrasonic_measurement_type": ultrasonic_measurement_type,
            "ultrasonic_surface_sigma": ultrasonic_surface_sigma,
            "ultrasonic_bottom_sigma": ultrasonic_bottom_sigma,
            "ultrasonic_degradation_factor": ultrasonic_degradation,
            "imu_degradation_factor": imu_degradation,
            "turbidity_category": turbidity_category,
            "turbidity_degradation_factor": turbidity_degradation,
            "water_contact_state": water_state,
            "water_contact_confidence": water_confidence,
            "water_contact_degradation_factor": water_degradation,
            "sensor_metadata": dict(self.sensor_metadata) if self.include_metadata else {},
        }