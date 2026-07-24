import numpy as np

class TurbidityModel:

    def __init__(self, noise_sigma=5, seed=42):
        self.noise_sigma = noise_sigma
        np.random.seed(seed)

    def generate(self, true_ntu):
        measured_ntu = (true_ntu + np.random.normal(0, self.noise_sigma))
        measured_ntu = np.clip(measured_ntu, 0, 500)
        return round(measured_ntu, 2)

    def classify(self, ntu):
        if ntu <= 50:
            return "Clear"
        elif ntu <= 150:
            return "Moderate"
        elif ntu <= 300:
            return "High"
        return "Extreme"

    def degradation_factor(self, ntu):
        return round(np.exp(-ntu / 500), 3)

    def reliability(self, ntu):
        return round(max(0.1, np.exp(-ntu / 450)), 3)

    def metadata(self):
        return {
            "sensor": "SEN0189",
            "noise_sigma": self.noise_sigma,
            "categories": ["Clear", "Moderate", "High", "Extreme"]
        }