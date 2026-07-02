import numpy as np

class LidarReliabilityModel:

    def __init__(self, max_ntu=500, attenuation_coeff=0.05):
        self.max_ntu = max_ntu
        self.attenuation_coeff = attenuation_coeff

    def compute_reliability(self, ntu, water_depth, noise_sigma):
        ntu = np.asarray(ntu)
        water_depth = np.asarray(water_depth)
        noise_sigma = np.asarray(noise_sigma)

        ntu_penalty = np.exp(-0.6 * (ntu / self.max_ntu))
        water_penalty = np.exp(-self.attenuation_coeff * water_depth)
        noise_penalty = np.exp(-0.2 * noise_sigma)
        reliability = (ntu_penalty * water_penalty * noise_penalty)

        return np.clip(reliability, 0, 1)

    def classify(self, reliability):
        if reliability >= 0.8:
            return "High"

        elif reliability >= 0.6:
            return "Moderate"

        elif reliability >= 0.4:
            return "Low"

        return "Very Low"

    def metadata(self):
        return {
            "sensor": "VL53L1X",
            "max_ntu": self.max_ntu,
            "attenuation": self.attenuation_coeff
        }