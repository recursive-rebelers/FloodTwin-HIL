import numpy as np

class LidarReliabilityModel:

    def __init__(
        self, max_ntu=500.0, attenuation_coeff=0.05,
        noise_ref_sigma=1.0, turbidity_coeff=0.60,
        coupling_coeff=0.12, reliability_floor=0.10):

        # ------------------------------------------------------------------
        # Parameter validation
        # ------------------------------------------------------------------
        if max_ntu <= 0: raise ValueError("max_ntu must be greater than 0")
        if attenuation_coeff < 0: raise ValueError("attenuation_coeff must be non-negative")
        if noise_ref_sigma <= 0: raise ValueError("noise_ref_sigma must be greater than 0")
        if turbidity_coeff < 0: raise ValueError("turbidity_coeff must be non-negative")
        if coupling_coeff < 0: raise ValueError("coupling_coeff must be non-negative")
        if not 0.0 <= reliability_floor <= 1.0:
            raise ValueError("reliability_floor must lie within [0, 1]")

        self.max_ntu = float(max_ntu)
        self.attenuation_coeff = float(attenuation_coeff)
        self.noise_ref_sigma = float(noise_ref_sigma)
        self.turbidity_coeff = float(turbidity_coeff)
        self.coupling_coeff = float(coupling_coeff)
        self.reliability_floor = float(reliability_floor)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _as_array(value):
        return np.asarray(value, dtype=float)

    @staticmethod
    def _validate_finite(name, value):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or infinite values")

    # ------------------------------------------------------------------
    # Core reliability estimation
    # ------------------------------------------------------------------
    def compute_reliability(self, ntu, water_depth, noise_sigma):
        ntu = self._as_array(ntu)
        water_depth = self._as_array(water_depth)
        noise_sigma = self._as_array(noise_sigma)

        self._validate_finite("ntu", ntu)
        self._validate_finite("water_depth", water_depth)
        self._validate_finite("noise_sigma", noise_sigma)

        ntu = np.maximum(ntu, 0.0)
        water_depth = np.maximum(water_depth, 0.0)
        noise_sigma = np.maximum(noise_sigma, 0.0)

        # 1. Normalized turbidity
        turbidity_ratio = np.clip(ntu / self.max_ntu, 0.0, 1.0)

        # 2. Phenomenological optical attenuation
        optical_loss = (self.turbidity_coeff * turbidity_ratio + self.attenuation_coeff * water_depth)
        transmission = np.exp(-optical_loss)

        # 3. Receiver / measurement-noise penalty
        effective_noise = (noise_sigma * (1.0 + 0.45 * (1.0 - transmission)))
        normalized_noise = effective_noise / self.noise_ref_sigma
        noise_penalty = np.exp(-0.18 * np.power(normalized_noise, 1.15))

        # 4. Turbidity × water-path interaction
        interaction = np.exp(-1.15 * self.coupling_coeff * turbidity_ratio * water_depth / 20.0)

        # 5. Aggregate contextual reliability
        reliability = transmission * noise_penalty * interaction

        # 6. Reliability regularization
        reliability = np.clip(reliability, self.reliability_floor, 1.0)
        if reliability.ndim == 0: return float(reliability)
        return reliability

    # ------------------------------------------------------------------
    # Optical signal-strength diagnostic
    # ------------------------------------------------------------------
    def signal_strength(self, ntu, water_depth):
        ntu = self._as_array(ntu)
        water_depth = self._as_array(water_depth)

        self._validate_finite("ntu", ntu)
        self._validate_finite("water_depth", water_depth)

        ntu = np.maximum(ntu, 0.0)
        water_depth = np.maximum(water_depth, 0.0)

        turbidity_ratio = np.clip(ntu / self.max_ntu, 0.0, 1.0)
        optical_loss = (self.turbidity_coeff * turbidity_ratio + self.attenuation_coeff * water_depth)

        strength = np.exp(-optical_loss)
        strength = np.clip(strength, 0.0, 1.0)

        if strength.ndim == 0: return float(strength)
        return strength

    # ------------------------------------------------------------------
    # Qualitative reliability classification
    # ------------------------------------------------------------------
    def classify(self, reliability):
        reliability = self._as_array(reliability)
        self._validate_finite("reliability", reliability)

        scalar_input = reliability.ndim == 0
        labels = np.empty(reliability.shape, dtype=object)

        labels[reliability >= 0.80] = "High"
        labels[(reliability >= 0.60) & (reliability < 0.80)] = "Moderate"
        labels[(reliability >= 0.35) & (reliability < 0.60)] = "Low"
        labels[reliability < 0.35] = "Very Low"

        if scalar_input: return str(labels.item())
        return labels

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------
    def metadata(self):
        return {
            "sensor": "VL53L1X",
            "model": "Phenomenological Optical Reliability",
            "model_type": "Inference-time contextual trust estimator",

            "max_ntu": self.max_ntu,
            "attenuation_coeff": self.attenuation_coeff,
            "noise_ref_sigma": self.noise_ref_sigma,
            "turbidity_coeff": self.turbidity_coeff,
            "coupling_coeff": self.coupling_coeff,
            "reliability_floor": self.reliability_floor,

            "components": [
                "Phenomenological Optical Transmission",
                "Receiver Noise Penalty",
                "Turbidity-Depth Interaction",
                "Reliability Floor",
            ],

            "inputs": {
                "ntu": "Measured/estimated turbidity available at inference time",
                "water_depth":
                    "Inference-time estimated/observable water-depth context; "
                    "must not be simulator ground truth",
                "noise_sigma": "LiDAR measurement uncertainty estimate",
            },

            "forbidden_inputs": [
                "true_depth", "true_water_depth", "true_ntu",
                "other simulator-only latent variables",
            ],

            "assumptions": [
                "Optical degradation is represented using engineering-defined "
                "phenomenological relationships",
                "Turbidity and water-depth effects are normalized within the "
                "synthetic Digital Twin operating range",
                "The noise term represents measurement uncertainty rather than a "
                "manufacturer-specified sensor characteristic",
                "Reliability is a contextual fusion trust score and not a "
                "physical probability of correct measurement",
                "The water_depth input must be obtained from inference-time "
                "observable/estimated information",
            ],

            "scientific_limitations": [
                "The coefficients are phenomenological and require sensitivity "
                "analysis and ablation",
                "The model is not a manufacturer-calibrated VL53L1X reliability model",
                "The reliability floor is a numerical trust regularization mechanism, "
                "not a claim of minimum physical sensor reliability",
            ],
        }