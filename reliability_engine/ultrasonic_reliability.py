import numpy as np

class UltrasonicReliabilityModel:

    def __init__(
        self, blind_zone=2.0, water_attenuation_coeff=0.028,
        ambiguity_coeff=0.45, min_reliability=0.25):

        # ------------------------------------------------------------------
        # Parameter validation
        # ------------------------------------------------------------------
        if blind_zone <= 0: raise ValueError("blind_zone must be greater than 0")
        if water_attenuation_coeff < 0:
            raise ValueError("water_attenuation_coeff must be non-negative")
        if ambiguity_coeff < 0: raise ValueError("ambiguity_coeff must be non-negative")
        if not 0.0 <= min_reliability <= 1.0:
            raise ValueError("min_reliability must lie within [0, 1]")

        self.blind_zone = float(blind_zone)
        self.water_attenuation_coeff = float(water_attenuation_coeff)
        self.ambiguity_coeff = float(ambiguity_coeff)
        self.min_reliability = float(min_reliability)

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
    # Core reliability model
    # ------------------------------------------------------------------
    def compute_reliability(self, water_depth, surface_echo_prob, bottom_echo_prob):
        water_depth = self._as_array(water_depth)
        surface_echo_prob = self._as_array(surface_echo_prob)
        bottom_echo_prob = self._as_array(bottom_echo_prob)

        self._validate_finite("water_depth", water_depth)
        self._validate_finite("surface_echo_prob", surface_echo_prob)
        self._validate_finite("bottom_echo_prob", bottom_echo_prob)

        # ------------------------------------------------------------------
        # Domain constraints
        # ------------------------------------------------------------------
        water_depth = np.maximum(water_depth, 0.0)
        surface_echo_prob = np.clip(surface_echo_prob, 0.0, 1.0)
        bottom_echo_prob = np.clip(bottom_echo_prob, 0.0, 1.0)

        # Small numerical stabilizer
        total_echo = surface_echo_prob + bottom_echo_prob + 1e-9

        # 1. Bottom-echo dominance
        bottom_dominance = bottom_echo_prob / total_echo
        bottom_strength = np.sqrt(bottom_echo_prob)
        echo_quality = (0.55 * np.sqrt(bottom_dominance) + 0.45 * bottom_strength)

        # 2. Smooth blind-zone response
        depth_response = (0.88 + 0.12 * np.tanh(water_depth / self.blind_zone))

        # 3. Surface/bottom echo ambiguity
        echo_clarity = np.abs(bottom_echo_prob - surface_echo_prob) / total_echo
        echo_clarity = np.clip(echo_clarity, 0.0, 1.0)

        ambiguity = 1.0 - echo_clarity
        ambiguity_penalty = np.exp(-self.ambiguity_coeff * ambiguity)

        # 4. Phenomenological water attenuation
        water_penalty = np.exp(-self.water_attenuation_coeff * water_depth)

        # 5. Aggregate contextual reliability
        reliability = (echo_quality * depth_response * ambiguity_penalty * water_penalty)

        # 6. Trust-score regularization
        reliability = np.clip(reliability, self.min_reliability, 1.0)
        if reliability.ndim == 0: return float(reliability)
        return reliability

    # ------------------------------------------------------------------
    # Echo confidence diagnostic
    # ------------------------------------------------------------------
    def echo_confidence(self, surface_echo_prob, bottom_echo_prob):
        surface_echo_prob = self._as_array(surface_echo_prob)
        bottom_echo_prob = self._as_array(bottom_echo_prob)

        self._validate_finite("surface_echo_prob", surface_echo_prob)
        self._validate_finite("bottom_echo_prob", bottom_echo_prob)

        surface_echo_prob = np.clip(surface_echo_prob, 0.0, 1.0)
        bottom_echo_prob = np.clip(bottom_echo_prob, 0.0, 1.0)

        total = surface_echo_prob + bottom_echo_prob + 1e-9
        confidence = bottom_echo_prob / total
        confidence = np.clip(confidence, 0.0, 1.0)

        if confidence.ndim == 0: return float(confidence)
        return confidence

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
        labels[(reliability >= 0.40) & (reliability < 0.60)] = "Low"
        labels[reliability < 0.40] = "Very Low"

        if scalar_input: return str(labels.item())
        return labels

    # ------------------------------------------------------------------
    # Scientific metadata
    # ------------------------------------------------------------------
    def metadata(self):
        return {
            "sensor": "JSN-SR04T",
            "model": "Phenomenological Ultrasonic Reliability",
            "model_type": "Inference-time contextual trust estimator",

            "blind_zone": self.blind_zone,
            "water_attenuation_coeff": self.water_attenuation_coeff,
            "ambiguity_coeff": self.ambiguity_coeff,
            "min_reliability": self.min_reliability,

            "components": [
                "Composite Bottom-Echo Quality",
                "Smooth Blind-Zone Response",
                "Echo Ambiguity Penalty",
                "Phenomenological Water-Depth Attenuation",
            ],

            "inputs": {
                "water_depth":
                    "Inference-time estimated/observable water-depth context; "
                    "must not be simulator ground truth",
                "surface_echo_prob":
                    "Estimated/simulated surface-echo signal available at inference time",
                "bottom_echo_prob":
                    "Estimated/simulated bottom-echo signal available at inference time",
            },

            "forbidden_inputs": [
                "true_depth", "true_water_depth", "true_ntu",
                "other simulator-only latent variables",
            ],

            "assumptions": [
                "Surface and bottom echo quantities are synthetic Digital Twin "
                "observables or inference-time signal-derived estimates",
                "Water-depth attenuation is represented using an engineering-defined "
                "phenomenological relationship",
                "The blind-zone parameter represents an effective model transition "
                "rather than a manufacturer-calibrated water-depth limit",
                "Reliability is a contextual fusion trust score, not a probability "
                "of measurement correctness",
                "The water_depth input must originate exclusively from inference-time "
                "observable/estimated context",
            ],

            "scientific_limitations": [
                "The model coefficients are phenomenological and require sensitivity "
                "and ablation analysis",
                "The model is not a manufacturer-calibrated JSN-SR04T reliability "
                "specification",
                "Echo probabilities should not be interpreted as statistically calibrated "
                "probabilities unless independently validated",
                "The minimum reliability is a trust-score regularization mechanism rather "
                "than a claim of minimum physical sensor reliability",
            ],
        }