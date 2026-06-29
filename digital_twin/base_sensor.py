from abc import ABC
from abc import abstractmethod
import numpy as np


class BaseSensor(ABC):

    def __init__(self, name, config=None, seed=42):
        self.name = name
        self.config = config or {}
        self.seed = seed
        np.random.seed(seed)

    
    @abstractmethod
    def generate(self, scenario):

        """ Generate sensor observation

        Parameters
        ----------
        scenario : dict

        Returns
        -------
        measurement """

        pass


    @abstractmethod
    def add_noise(self, measurement, scenario):

        """ Apply environmental degradation. Examples: turbidity, scattering, attenuation, thermal noise """

        pass


    @abstractmethod
    def inject_fault(self, measurement, fault_type=None):

        """ Fault Injection Engine. Supports: dropout, delay, drift, saturation, blockage """

        pass


    @abstractmethod
    def get_reliability(self, scenario):

        """ Compute trustworthiness. Output: 0.0 to 1.0 """

        pass


    def calibrate(self):

        """ Sensor calibration. Optional override """

        return None


    def validate(self, measurement):
        if measurement is None:
            return False
        if isinstance(measurement, dict):
            return all(self.validate(v) for v in measurement.values())
        if isinstance(measurement, (list, tuple, np.ndarray)):
            try:
                return np.all(np.isfinite(np.asarray(measurement, dtype=float)))
            except (TypeError, ValueError):
                return False
        try:
            return bool(np.isfinite(measurement))
        except (TypeError, ValueError):
            return False


    def reset(self):

        """ Reset internal state. Useful for drift models """

        pass


    def get_metadata(self):

        return {
            "sensor": self.name,
            "seed": self.seed,
            "config": self.config
        }