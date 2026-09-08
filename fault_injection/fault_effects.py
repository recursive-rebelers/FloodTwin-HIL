import math
from typing import Dict, Optional, Sequence, Tuple

class FaultEffects:
    
    SEVERITY_ALIASES = {
        "mild": "low",
        "moderate": "medium",
        "severe": "high",
        "normal": "none"
    }

    SEVERITY_SCALE = {
        "none": 0.00,
        "low": 0.10,
        "medium": 0.30,
        "high": 0.50,
        "extreme": 0.80
    }

    SENSOR_SENSITIVITY = {
        "lidar": 1.00,
        "ultrasonic": 1.05,
        "radar": 0.90,
        "imu": 1.15,
        "water": 0.85
    }

    FAULT_PROFILE = {

        "dropout": {
            "reliability_alpha": 0.00,
            "sigma_beta": 0.00
        },

        "noise": {
            "reliability_alpha": 1.15,
            "sigma_beta": 2.40
        },

        "delay": {
            "reliability_alpha": 0.50,
            "sigma_beta": 1.05
        },

        "sync": {
            "reliability_alpha": 0.65,
            "sigma_beta": 1.20
        },

        "drift": {
            "reliability_alpha": 0.08,
            "sigma_beta": 0.12
        },

        "compound": {
            "reliability_alpha": 1.00,
            "sigma_beta": 1.00
        }
    }

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed

    def _normalize_fault_type(self, fault_type: str) -> str:
        if fault_type is None:
            raise ValueError("fault_type cannot be None")
        return str(fault_type).strip().lower()

    def _normalize_severity(self, severity) -> Tuple[str, float]:
        if severity is None:
            return "none", 0.0
        key = str(severity).strip().lower()
        key = self.SEVERITY_ALIASES.get(key, key)

        if key not in self.SEVERITY_SCALE:
            raise ValueError(
                f"Severity '{severity}' is not supported\n"
                f"Supported values: {list(self.SEVERITY_SCALE.keys()) + list(self.SEVERITY_ALIASES.keys())}")
        return key, self.SEVERITY_SCALE[key]

    def get_severity_value(self, severity) -> float:
        _, value = self._normalize_severity(severity)
        return value

    def get_sensor_sensitivity(self, sensor: str) -> float:
        key = str(sensor).lower().strip()
        return self.SENSOR_SENSITIVITY.get(key, 1.0)

    def _clip_reliability(self, value: float) -> float:
        return float(max(0.02, min(1.0, value)))

    def _clip_availability(self, value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _clip_sigma(self, value: float) -> float:
        return float(max(1e-6, value))

    def apply_fault(self, sensor: str, fault_type: str,
        severity: str = "low",
        reliability: float = 1.0,
        sigma: float = 1.0,
        availability: float = 1.0,
        frames: Optional[int] = None,
        drift_rate: Optional[float] = None,
        drift_bias: Optional[float] = None) -> Dict[str, float]:
    
        sensor_key = str(sensor).lower().strip()
        fault_key = self._normalize_fault_type(fault_type)
        severity_key, severity_value = self._normalize_severity(severity)
        sensor_scale = self.get_sensor_sensitivity(sensor_key)

        updated_reliability = self._clip_reliability(reliability)
        updated_sigma = self._clip_sigma(sigma)
        updated_availability = self._clip_availability(availability)

        if fault_key not in self.FAULT_PROFILE:
            raise ValueError(f"fault_type must be one of {list(self.FAULT_PROFILE.keys())}")
        profile = self.FAULT_PROFILE[fault_key]

        if fault_key == "dropout":
            loss = severity_value
            updated_availability = self._clip_availability(updated_availability * (1.0 - loss))

            return {
                "sensor": sensor_key,
                "fault_type": fault_key,
                "severity": severity_key,
                "reliability": updated_reliability,
                "sigma": updated_sigma,
                "availability": updated_availability
            }

        if fault_key in {"noise", "delay", "sync"}:
            alpha = profile["reliability_alpha"]
            beta = profile["sigma_beta"]
            effect_strength = severity_value * sensor_scale

            reliability_factor = math.exp(-alpha * effect_strength)
            sigma_factor = 1.0 + 0.60 * beta * effect_strength

            updated_reliability = self._clip_reliability(updated_reliability * reliability_factor)
            updated_sigma = self._clip_sigma(updated_sigma * sigma_factor)

            return {
                "sensor": sensor_key,
                "fault_type": fault_key,
                "severity": severity_key,
                "reliability": updated_reliability,
                "sigma": updated_sigma,
                "availability": updated_availability
            }

        if fault_key == "drift":
            if drift_bias is not None:
                drift_magnitude = abs(float(drift_bias))
            elif drift_rate is not None and frames is not None:
                drift_magnitude = abs(float(drift_rate)) * float(frames)
            elif frames is not None:
                drift_magnitude = severity_value * float(frames) * 0.01
            else:
                drift_magnitude = severity_value * 10.0

            alpha = profile["reliability_alpha"]
            beta = profile["sigma_beta"]
            effect_strength = drift_magnitude * sensor_scale

            reliability_factor = math.exp(-alpha * effect_strength)
            sigma_factor = 1.0 + 0.60 * beta * effect_strength

            updated_reliability = self._clip_reliability(updated_reliability * reliability_factor)
            updated_sigma = self._clip_sigma(updated_sigma * sigma_factor)

            return {
                "sensor": sensor_key,
                "fault_type": fault_key,
                "severity": severity_key,
                "reliability": updated_reliability,
                "sigma": updated_sigma,
                "availability": updated_availability,
                "drift_magnitude": float(drift_magnitude)
            }

        return {
            "sensor": sensor_key,
            "fault_type": fault_key,
            "severity": severity_key,
            "reliability": updated_reliability,
            "sigma": updated_sigma,
            "availability": updated_availability
        }

    def apply_fault_sequence(self, sensor: str,
        fault_sequence: Sequence[Dict[str, object]],
        reliability: float = 1.0,
        sigma: float = 1.0,
        availability: float = 1.0) -> Dict[str, float]:
    
        state = {
            "sensor": str(sensor),
            "reliability": float(reliability),
            "sigma": float(sigma),
            "availability": float(availability)
        }

        for fault in fault_sequence:
            fault_type = fault.get("fault_type")
            if fault_type is None:
                raise ValueError("Each fault entry must include 'fault_type'")

            state = self.apply_fault(
                sensor=sensor, fault_type=fault_type,
                severity=fault.get("severity", "none"),
                reliability=state["reliability"],
                sigma=state["sigma"],
                availability=state["availability"],
                frames=fault.get("frames"),
                drift_rate=fault.get("drift_rate"),
                drift_bias=fault.get("drift_bias"))
        return state

    def summary_for_experiment(self, sensor: str, fault_type: str, severity: str,
        reliability: float, sigma: float, **kwargs) -> Dict[str, float]:    
        updated = self.apply_fault(
            sensor=sensor,
            fault_type=fault_type,
            severity=severity,
            reliability=reliability,
            sigma=sigma, **kwargs)

        return {
            "sensor": updated["sensor"],
            "fault_type": updated["fault_type"],
            "severity": updated["severity"],
            "reliability_before": float(reliability),
            "reliability_after": updated["reliability"],
            "sigma_before": float(sigma),
            "sigma_after": updated["sigma"],
            "availability": updated["availability"]
        }

    def metadata(self) -> Dict[str, object]:
        return {
            "fault_type": "fault_effect_mapping",
            "model": "Fault-aware reliability and uncertainty propagation",
            "severity_scale": self.SEVERITY_SCALE,
            "severity_aliases": self.SEVERITY_ALIASES,
            "sensor_sensitivity": self.SENSOR_SENSITIVITY,
            "profiles": self.FAULT_PROFILE
        }