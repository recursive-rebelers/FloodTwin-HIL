import numpy as np
import random

class UltrasonicModel:

    def __init__(self, base_noise_sigma=0.15, seed=42):
        self.base_noise_sigma = base_noise_sigma
        random.seed(seed)
        np.random.seed(seed)

    def generate(self, true_depth, water_depth):

        if water_depth <= 0:
            measurement = true_depth + np.random.normal(0, self.base_noise_sigma)

            return {
                "distance": round(measurement, 2),
                "surface_echo": 0.0,
                "bottom_echo": 1.0,
                "measurement_type": "Bottom"
            }

        surface_echo = min(0.6, 0.2 + water_depth * 0.02)
        bottom_echo = 1 - surface_echo

        if random.random() < surface_echo:
            distance = (true_depth - water_depth + np.random.normal(0, self.base_noise_sigma * 2))
            measurement_type = "Surface"

        else:
            distance = (true_depth + np.random.normal(0, self.base_noise_sigma))
            measurement_type = "Bottom"

        return {
            "distance": round(max(0, distance), 2),
            "surface_echo": round(surface_echo, 3),
            "bottom_echo": round(bottom_echo, 3),
            "measurement_type": measurement_type
        }

    def reliability(self, water_depth):

        surface_echo = min(0.6, 0.2 + water_depth * 0.02)
        reliability = max(0.25, 1 - 1.5 * surface_echo)
        return round(reliability, 3)

    def metadata(self):

        return {
            "sensor": "JSN-SR04T",
            "noise_sigma": self.base_noise_sigma
        }