import numpy as np

class RadarReliabilityModel:

    def __init__(self, max_clutter=10.0, nominal_snr=20.0):
        self.max_clutter = max_clutter
        self.nominal_snr = nominal_snr

    def compute_reliability(self, snr, clutter, water_depth):
        snr = np.asarray(snr)
        clutter = np.asarray(clutter)
        water_depth = np.asarray(water_depth)

        snr_factor = 1.0 / (1.0 + np.exp(-(snr - (self.nominal_snr * 0.6)) / 2.0))
        water_penalty = np.exp(-0.012 * water_depth)

        clutter_penalty = np.exp(-0.015 * clutter)
        clutter_factor = 1.0 / (1.0 + (clutter / self.max_clutter) ** 2)

        reliability = (snr_factor * clutter_factor * clutter_penalty * water_penalty)

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
            "sensor": "HLK-LD2411S",
            "max_clutter": self.max_clutter,
            "nominal_snr": self.nominal_snr
        }