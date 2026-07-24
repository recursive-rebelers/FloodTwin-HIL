import numpy as np

class LidarReliabilityModel:
    
    def __init__(self,
        max_ntu=500.0,
        attenuation_coeff=0.05,
        noise_ref_sigma=1.0,
        turbidity_coeff=0.60,
        coupling_coeff=0.12,
        reliability_floor=0.10):
        
        self.max_ntu = float(max_ntu)
        self.attenuation_coeff = float(attenuation_coeff)
        self.noise_ref_sigma = float(noise_ref_sigma)
        self.turbidity_coeff = float(turbidity_coeff)
        self.coupling_coeff = float(coupling_coeff)
        self.reliability_floor = float(reliability_floor)

    def _as_array(self, value):
        return np.asarray(value, dtype=float)

    def compute_reliability(self, ntu, water_depth, noise_sigma):
        ntu = np.maximum(self._as_array(ntu), 0.0)
        water_depth = np.maximum(self._as_array(water_depth), 0.0)
        noise_sigma = np.maximum(self._as_array(noise_sigma), 0.0)

        # Beer-Lambert style optical loss through turbid water
        optical_loss = (self.turbidity_coeff * (ntu / self.max_ntu)
            + self.attenuation_coeff * water_depth)
        transmission = np.exp(-optical_loss)

        # Normalized receiver-noise penalty
        effective_noise = (noise_sigma * (1.0 + 0.45 * (1.0 - transmission)))
        noise_penalty = np.exp(-0.18 * np.power(effective_noise / self.noise_ref_sigma, 1.15))

        # Interaction penalty for simultaneous turbidity + path length
        interaction = np.exp(-1.15 * self.coupling_coeff * (ntu / self.max_ntu) * water_depth / 20.0)

        reliability = transmission * noise_penalty * interaction
        return np.clip(reliability, self.reliability_floor, 1.0)

    def signal_strength(self, ntu, water_depth):        
        ntu = np.maximum(self._as_array(ntu), 0.0)
        water_depth = np.maximum(self._as_array(water_depth), 0.0)

        optical_loss = (self.turbidity_coeff * (ntu / self.max_ntu)
            + self.attenuation_coeff * water_depth)

        strength = np.exp(-optical_loss)
        return np.clip(strength, 0.0, 1.0)

    def classify(self, reliability):
        reliability = np.asarray(reliability, dtype=float)

        if reliability.ndim == 0:
            value = float(reliability)
            if value >= 0.80:
                return "High"
            elif value >= 0.60:
                return "Moderate"
            elif value >= 0.35:
                return "Low"
            return "Very Low"

        labels = np.empty(reliability.shape, dtype=object)
        labels[reliability >= 0.80] = "High"
        labels[(reliability >= 0.60) & (reliability < 0.80)] = "Moderate"
        labels[(reliability >= 0.35) & (reliability < 0.60)] = "Low"
        labels[reliability < 0.35] = "Very Low"
        return labels

    def metadata(self):
        return {
            "sensor": "VL53L1X",
            "model": "Physics-aware Optical Reliability",
            "max_ntu": self.max_ntu,
            "attenuation_coeff": self.attenuation_coeff,
            "noise_ref_sigma": self.noise_ref_sigma,
            "turbidity_coeff": self.turbidity_coeff,
            "coupling_coeff": self.coupling_coeff,
            "reliability_floor": self.reliability_floor,
            
            "components": [
                "Beer-Lambert Optical Transmission",
                "Receiver Noise Penalty",
                "Turbidity-Depth Interaction",
                "Reliability Saturation",
            ]
        }