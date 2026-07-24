import numpy as np

class RadarReliabilityModel:
    
    def __init__(self,
        max_clutter=10.0,
        nominal_snr=20.0,
        reliability_floor=0.12,
        interaction_snr_coupling=0.45):

        self.max_clutter = float(max_clutter)
        self.nominal_snr = float(nominal_snr)
        self.reliability_floor = float(reliability_floor)
        self.interaction_snr_coupling = float(interaction_snr_coupling)

    def _sigmoid(self, x):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def compute_reliability(self, snr, clutter, water_depth, rcs=None):
        snr = np.asarray(snr, dtype=float)
        clutter = np.asarray(clutter, dtype=float)
        water_depth = np.asarray(water_depth, dtype=float)

        snr = np.maximum(snr, 0.0)
        clutter = np.maximum(clutter, 0.0)
        water_depth = np.maximum(water_depth, 0.0)

        # 1) SNR saturation: higher SNR => stronger confidence
        snr_center = 0.48 * self.nominal_snr
        snr_scale = max(0.22 * self.nominal_snr, 1e-6)
        snr_factor = self._sigmoid((snr - snr_center) / snr_scale)

        # 2) Clutter suppression: reliability drops faster once clutter approaches the upper operating region
        clutter_center = 0.62 * self.max_clutter
        clutter_scale = max(0.16 * self.max_clutter, 1e-6)
        clutter_factor = self._sigmoid((clutter_center - clutter) / clutter_scale)

        # 3) Water attenuation: radar is comparatively robust to shallow flood water, so the penalty is mild
        water_attenuation_coeff = (0.013 + 0.004 * (1.0 - snr_factor) + 0.003 * (clutter / self.max_clutter))
        water_penalty = np.exp(-water_attenuation_coeff * water_depth)

        # 4) Interaction penalty: stronger when clutter acts on weak returns
        coupled_scattering = (0.0038 * np.power(clutter, 1.25) * water_depth / (self.max_clutter + 1e-6))
        interaction_boost = 1.0 + self.interaction_snr_coupling * (1.0 - snr_factor)
        interaction_penalty = np.exp(-(coupled_scattering * interaction_boost))

        # 5) Optional RCS confidence: if available, weak reflections reduce trust slightly without dominating the model
        if rcs is None: rcs_factor = 1.0
        else:
            rcs = np.asarray(rcs, dtype=float)
            rcs = np.maximum(rcs, 0.0)
            rcs_factor = 0.85 + 0.15 * (rcs / (rcs + 0.35))

        reliability = (snr_factor
            * clutter_factor
            * water_penalty
            * interaction_penalty
            * rcs_factor)

        return np.clip(reliability, self.reliability_floor, 1.0)

    def classify(self, reliability):
        reliability = np.asarray(reliability, dtype=float)

        labels = np.full(reliability.shape, "Very Low", dtype=object)
        labels = np.where(reliability >= 0.4, "Low", labels)
        labels = np.where(reliability >= 0.6, "Moderate", labels)
        labels = np.where(reliability >= 0.8, "High", labels)

        if labels.shape == ():
            return str(labels.item())
        return labels

    def metadata(self):
        return {
            "sensor": "HLK-LD2411S",
            "max_clutter": self.max_clutter,
            "nominal_snr": self.nominal_snr,
            "reliability_floor": self.reliability_floor,
            "interaction_snr_coupling": self.interaction_snr_coupling,
            
            "physical_model": [
                "SNR Saturation",
                "Clutter Suppression",
                "Water Attenuation",
                "Multipath Water-Clutter Interaction",
                "Optional RCS Confidence",
                "Reliability Floor",
            ],
        }