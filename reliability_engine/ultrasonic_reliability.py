import numpy as np

class UltrasonicReliabilityModel:
    
    def __init__(self,
        blind_zone=2.0,
        water_attenuation_coeff=0.022,
        ambiguity_coeff=0.45,
        min_reliability=0.25):

        self.blind_zone = float(blind_zone)
        self.water_attenuation_coeff = float(water_attenuation_coeff)
        self.ambiguity_coeff = float(ambiguity_coeff)
        self.min_reliability = float(min_reliability)

    def _as_array(self, value):
        return np.asarray(value, dtype=float)

    def compute_reliability(self, water_depth, surface_echo_prob, bottom_echo_prob):
        water_depth = np.maximum(self._as_array(water_depth), 0.0)
        surface_echo_prob = np.clip(self._as_array(surface_echo_prob), 0.0, 1.0)
        bottom_echo_prob = np.clip(self._as_array(bottom_echo_prob), 0.0, 1.0)
        total_echo = surface_echo_prob + bottom_echo_prob + 1e-9

        # Desired bottom echo dominance
        echo_ratio = bottom_echo_prob / total_echo
        echo_ratio = np.sqrt(echo_ratio)

        # Smooth blind-zone response
        depth_penalty = np.where(
            water_depth < 1e-6, 1.0,
            np.tanh(0.8 * water_depth / self.blind_zone))

        # Ambiguity is high when surface and bottom echoes are similar
        echo_clarity = np.abs(bottom_echo_prob - surface_echo_prob) / total_echo
        ambiguity = 1.0 - np.clip(echo_clarity, 0.0, 1.0)
        ambiguity_penalty = np.exp(-self.ambiguity_coeff * ambiguity)

        # Mild attenuation through water
        water_penalty = np.exp(-self.water_attenuation_coeff * water_depth)
        reliability = echo_ratio * depth_penalty * ambiguity_penalty * water_penalty
        return np.clip(reliability, self.min_reliability, 1.0)

    def echo_confidence(self, surface_echo_prob, bottom_echo_prob):
        surface_echo_prob = np.clip(self._as_array(surface_echo_prob), 0.0, 1.0)
        bottom_echo_prob = np.clip(self._as_array(bottom_echo_prob), 0.0, 1.0)
        total = surface_echo_prob + bottom_echo_prob + 1e-9
        return np.clip(bottom_echo_prob / total, 0.0, 1.0)

    def classify(self, reliability):
        reliability = np.asarray(reliability, dtype=float)
        if reliability.ndim == 0:
            value = float(reliability)
            if value >= 0.8:
                return "High"
            elif value >= 0.6:
                return "Moderate"
            elif value >= 0.4:
                return "Low"
            return "Very Low"

        labels = np.empty(reliability.shape, dtype=object)
        labels[reliability >= 0.8] = "High"
        labels[(reliability >= 0.6) & (reliability < 0.8)] = "Moderate"
        labels[(reliability >= 0.4) & (reliability < 0.6)] = "Low"
        labels[reliability < 0.4] = "Very Low"
        return labels

    def metadata(self):
        return {
            "sensor": "JSN-SR04T",
            "model": "Physics-aware Ultrasonic Reliability",
            "blind_zone": self.blind_zone,
            "water_attenuation_coeff": self.water_attenuation_coeff,
            "ambiguity_coeff": self.ambiguity_coeff,
            "min_reliability": self.min_reliability,
            
            "components": [
                "Normalized Echo Ratio",
                "Smooth Blind-Zone Response",
                "Echo Ambiguity Penalty",
                "Water Attenuation",
            ]
        }