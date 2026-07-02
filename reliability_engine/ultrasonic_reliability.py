import numpy as np

class UltrasonicReliabilityModel:

    def __init__(self, blind_zone=2.0):
        self.blind_zone = blind_zone

    def compute_reliability(self, water_depth, surface_echo_prob, bottom_echo_prob):
        water_depth = np.asarray(water_depth)
        surface_echo_prob = np.asarray(surface_echo_prob)
        bottom_echo_prob = np.asarray(bottom_echo_prob)

        echo_ratio = bottom_echo_prob

        depth_penalty = np.where(
            (water_depth > 0) & (water_depth < self.blind_zone), 
            np.exp(-0.5 *(self.blind_zone - water_depth)), 1.0)

        uncertainty_penalty = np.exp(-0.5 * surface_echo_prob)
        reliability = (echo_ratio * depth_penalty * uncertainty_penalty)

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
            "sensor": "JSN-SR04T",
            "blind_zone": self.blind_zone
        }