import numpy as np

class LidarModel:

    def __init__(self, base_noise_sigma=0.30, refraction_factor=0.12, seed=42):
        self.base_noise_sigma = base_noise_sigma
        self.refraction_factor = refraction_factor
        np.random.seed(seed)

    def generate(self, true_depth, water_depth, ntu):        
        water_attenuation = np.exp(0.055 * max(water_depth, 0))
        turbidity_scattering = np.exp(0.003 * max(ntu, 0))

        noise_sigma = (self.base_noise_sigma * water_attenuation * turbidity_scattering)
        refraction_bias = (self.refraction_factor * np.sqrt(max(water_depth, 0.0)) * (1.0 - np.exp(-water_depth / 12.0)))

        noise = np.random.normal(0.0, noise_sigma)
        measurement = true_depth + refraction_bias + noise
        return round(max(0.0, measurement), 2)

    def reliability(self, water_depth, ntu):
        reliability = np.exp(-(water_depth / 16.0) - (ntu / 500.0))
        return round(max(0.10, reliability), 3)

    def degradation_factor(self, water_depth, ntu):
        degradation = 1.0 - self.reliability(water_depth, ntu)
        return round(degradation, 3)

    def signal_strength(self, water_depth, ntu):
        strength = np.exp(-(0.05 * water_depth) - (0.002 * ntu))
        return round(float(np.clip(strength, 0.0, 1.0)), 3)

    def metadata(self):
        return {
            "sensor": "VL53L1X",
            "base_noise_sigma": self.base_noise_sigma,
            "refraction_factor": self.refraction_factor,

            "physical_model": [
                "Beer-Lambert Water Attenuation",
                "Optical Turbidity Scattering",
                "Saturating Water Refraction",
                "Gaussian Receiver Noise",
            ]
        }