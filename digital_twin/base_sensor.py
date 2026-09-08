from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

class BaseSensor(ABC):

    """Abstract Base Contract For All Sensor Models"""
    def __init__(self, name: str, config: Optional[dict] = None, seed: int = 42) -> None:
        self.name = str(name)
        self.config = dict(config) if config else {}
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def generate(self, scenario: dict) -> Any:
        # Generate the sensor observation for a scenario
        raise NotImplementedError

    @abstractmethod
    def add_noise(self, measurement: Any, scenario: dict) -> Any:
        # Apply environmental noise or degradation
        raise NotImplementedError

    @abstractmethod
    def inject_fault(self, measurement: Any, fault_type: Optional[str] = None) -> Any:
        # Inject a fault such as dropout, drift, or saturation
        raise NotImplementedError

    @abstractmethod
    def get_reliability(self, scenario: dict) -> float:
        # Return the sensor trust score in the range [0.0, 1.0]
        raise NotImplementedError

    def calibrate(self) -> None:
        # Optional hook for sensor-specific calibration
        return None

    def validate(self, measurement: Any) -> bool:
        # Check whether the measurement is finite and usable
        if measurement is None:
            return False

        if isinstance(measurement, dict):
            return all(self.validate(value) for value in measurement.values())

        if isinstance(measurement, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(measurement, dtype=float)
                return bool(np.all(np.isfinite(arr)))
            except (TypeError, ValueError):
                return False

        try:
            return bool(np.isfinite(measurement))
        except (TypeError, ValueError):
            return False

    def reset(self) -> None:
        # Restore the original random state
        self.rng = np.random.default_rng(self.seed)

    def get_metadata(self) -> dict[str, Any]:
        # Return a small, structured description of the sensor instance
        return {
            "sensor": self.name,
            "seed": self.seed,
            "config": self.config,
        }