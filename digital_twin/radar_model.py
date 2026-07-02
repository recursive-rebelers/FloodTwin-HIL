import numpy as np
import random

class RadarModel:

    def __init__(self, base_noise_sigma=0.1, seed=42):

        self.base_noise_sigma = base_noise_sigma
        random.seed(seed)
        np.random.seed(seed)

    def generate(self, true_depth, water_depth, ntu):

        noise_sigma = self.base_noise_sigma + water_depth * 0.01 + ntu / 5000
        clutter_probability = min(0.30, 0.05 + water_depth * 0.02 + ntu / 5000)
        radar_distance = true_depth + np.random.normal(0, noise_sigma)

        if random.random() < clutter_probability:
            radar_distance += np.random.normal(0.4, 0.15)

        snr = (30 - water_depth * 1.2 + np.random.normal(0, 1.5))
        snr = np.clip(snr, 5, 35)
        rcs = max(0.15, np.exp(-water_depth / 25))

        return {
            "distance": round(radar_distance, 2),
            "snr": round(snr, 2),
            "rcs": rcs,
            "clutter_probability": round(clutter_probability, 3)
        }

    def reliability(self, water_depth, ntu=0):

        water_term = np.exp(-water_depth / 22)
        turbidity_term = np.exp(-ntu / 2500)
        reliability = (water_term * turbidity_term)
        return round(min(1.0, max(0.35, reliability)), 3)

    def metadata(self):

        return {
            "sensor": "HLK-LD2411S",
            "noise_sigma": self.base_noise_sigma
        }