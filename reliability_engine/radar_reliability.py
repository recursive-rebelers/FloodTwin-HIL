import numpy as np

class RadarReliabilityModel:

    def __init__(
        self, max_clutter=0.30, nominal_snr=20.0,
        reliability_floor=0.12, interaction_snr_coupling=0.45):

        # ------------------------------------------------------------------
        # Parameter validation
        # ------------------------------------------------------------------
        if max_clutter <= 0: raise ValueError("max_clutter must be greater than 0")
        if nominal_snr <= 0: raise ValueError("nominal_snr must be greater than 0")
        if not 0.0 <= reliability_floor <= 1.0:
            raise ValueError("reliability_floor must lie within [0, 1]")
        if interaction_snr_coupling < 0:
            raise ValueError("interaction_snr_coupling must be non-negative")

        self.max_clutter = float(max_clutter)
        self.nominal_snr = float(nominal_snr)
        self.reliability_floor = float(reliability_floor)
        self.interaction_snr_coupling = float(interaction_snr_coupling)

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
    # Numerically stable sigmoid
    # ------------------------------------------------------------------
    def _sigmoid(self, x):
        x = self._as_array(x)
        self._validate_finite("sigmoid input", x)
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    # Core reliability model
    # ------------------------------------------------------------------
    def compute_reliability(self, snr, clutter, water_depth, rcs=None):
        snr = self._as_array(snr)
        clutter = self._as_array(clutter)
        water_depth = self._as_array(water_depth)

        self._validate_finite("snr", snr)
        self._validate_finite("clutter", clutter)
        self._validate_finite("water_depth", water_depth)

        # ------------------------------------------------------------------
        # Domain constraints
        # ------------------------------------------------------------------
        snr = np.maximum(snr, 0.0)
        clutter = np.maximum(clutter, 0.0)
        water_depth = np.maximum(water_depth, 0.0)

        # 1. SNR-based confidence
        snr_center = 0.48 * self.nominal_snr
        snr_scale = max(0.22 * self.nominal_snr, 1e-6)
        snr_factor = self._sigmoid((snr - snr_center) / snr_scale)

        # 2. Clutter suppression
        clutter_ratio = np.clip(clutter / max(self.max_clutter, 1e-6), 0.0, 1.0)
        clutter_factor = 1.0 - 0.20 * clutter_ratio

        # 3. Water/path attenuation
        water_attenuation_coeff = (0.013 + 0.004 * (1.0 - snr_factor) + 0.003 * clutter_ratio)
        water_penalty = np.exp(-water_attenuation_coeff * water_depth)

        # 4. Water × clutter × weak-return interaction
        coupled_scattering = (0.0038 * np.power(clutter, 1.25) * water_depth
            / max(self.max_clutter, 1e-6))

        interaction_boost = (1.0 + self.interaction_snr_coupling * (1.0 - snr_factor))
        interaction_penalty = np.exp(-coupled_scattering * interaction_boost)

        # 5. Optional normalized return-strength confidence
        if rcs is None: rcs_factor = 1.0
        else:
            rcs = self._as_array(rcs)
            self._validate_finite("rcs", rcs)
            rcs = np.maximum(rcs, 0.0)
            rcs_factor = (0.85 + 0.15 * (rcs / (rcs + 0.35)))

        # 6. Aggregate contextual reliability
        reliability = (snr_factor * clutter_factor * water_penalty * interaction_penalty * rcs_factor)

        # 7. Trust-score regularization
        reliability = np.clip(reliability, self.reliability_floor, 1.0)
        if reliability.ndim == 0: return float(reliability)
        return reliability

    # ------------------------------------------------------------------
    # Qualitative reliability classification
    # ------------------------------------------------------------------
    def classify(self, reliability):
        reliability = self._as_array(reliability)
        self._validate_finite("reliability", reliability)

        labels = np.full(reliability.shape, "Very Low", dtype=object)
        labels[reliability >= 0.40] = "Low"
        labels[reliability >= 0.60] = "Moderate"
        labels[reliability >= 0.80] = "High"

        if labels.ndim == 0: return str(labels.item())
        return labels

    # ------------------------------------------------------------------
    # Scientific metadata
    # ------------------------------------------------------------------
    def metadata(self):
        return {
            "sensor": "HLK-LD2411S",
            "model": "Phenomenological Radar Reliability",
            "model_type": "Inference-time contextual trust estimator",

            "max_clutter": self.max_clutter,
            "nominal_snr": self.nominal_snr,
            "reliability_floor": self.reliability_floor,
            "interaction_snr_coupling": self.interaction_snr_coupling,

            "components": [
                "SNR-Based Confidence",
                "Clutter Suppression",
                "Water-Depth Degradation",
                "Water-Clutter Interaction",
                "Normalized Return-Strength Confidence",
                "Reliability Floor",
            ],

            "inputs": {
                "snr": "Measured/estimated signal-to-noise ratio available at inference time",
                "clutter": "Measured/estimated normalized clutter indicator available at inference time",
                "water_depth":
                    "Inference-time estimated/observable water-depth context; "
                    "must not be simulator ground truth",
                "rcs":
                    "Optional normalized return-strength simulation indicator; "
                    "not automatically interpreted as physical radar cross section",
            },

            "forbidden_inputs": [
                "true_depth", "true_water_depth", "true_ntu",
                "other simulator-only latent variables",
            ],

            "assumptions": [
                "SNR and clutter are represented using measurable or estimated "
                "Digital Twin observations",
                "Water-depth and clutter effects are represented using "
                "engineering-defined phenomenological relationships",
                "The radar water-path penalty is intentionally milder than the "
                "corresponding optical LiDAR degradation within the modeled "
                "operating range",
                "Return strength is a normalized simulation indicator rather than "
                "an automatically calibrated physical radar cross section",
                "Reliability is a contextual fusion trust score rather than a "
                "probability of correct measurement",
                "The water_depth input must be obtained exclusively from "
                "inference-time observable/estimated context",
            ],

            "scientific_limitations": [
                "All coefficients are phenomenological and require sensitivity "
                "and ablation analysis",
                "The model is not a manufacturer-calibrated HLK-LD2411S "
                "reliability specification",
                "The optional rcs input should be described as normalized return "
                "strength unless an independent physical calibration is available",
                "The reliability floor is a trust-score regularization mechanism "
                "rather than a claim of minimum physical radar reliability",
            ],
        }