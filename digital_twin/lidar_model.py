import numpy as np

class LidarModel:

    def __init__(self, base_noise_sigma=0.25, refraction_factor=0.12, seed=42):

        self.base_noise_sigma = base_noise_sigma
        self.refraction_factor = refraction_factor
        np.random.seed(seed)

    def generate(self, true_depth, water_depth, ntu):

        noise_sigma = self.base_noise_sigma

        # Water attenuation
        water_penalty = 1 + 0.05 * water_depth

        # Turbidity attenuation
        turbidity_penalty = 1 + (ntu / 500) * 1.5

        # Mild nonlinear refraction
        refraction_bias = (self.refraction_factor * np.sqrt(max(water_depth, 0)))
        noise_sigma *= (water_penalty * turbidity_penalty)
        noise = np.random.normal(0, noise_sigma)
        measurement = (true_depth + refraction_bias + noise)

        return round(max(0, measurement), 2)

    def reliability(self, water_depth, ntu):

        R = (np.exp(-water_depth/15) * np.exp(-ntu/500))
        return round(max(0.1, R), 3)

    def degradation_factor(self, water_depth, ntu):

        degradation = (1 - self.reliability(water_depth, ntu))
        return round(degradation, 3)

    def metadata(self):

        return {
            "sensor": "VL53L1X",
            "base_noise_sigma": self.base_noise_sigma,
            "refraction_factor": self.refraction_factor
        }