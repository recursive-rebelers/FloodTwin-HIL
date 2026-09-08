import numpy as np

class AdaptiveReliabilityWeighting:

    def __init__(
        self, sensor_prior=None, trust_exponent=1.20,
        weight_floor=0.025, balance_strength=0.05,
        disagreement_gain=0.18, blend_cap=0.28):

        # ------------------------------------------------------------------
        # Parameter validation
        # ------------------------------------------------------------------
        if trust_exponent <= 0: raise ValueError("trust_exponent must be greater than 0")
        if not 0.0 <= weight_floor < 1.0: raise ValueError("weight_floor must lie within [0, 1)")
        if balance_strength < 0: raise ValueError("balance_strength must be non-negative")
        if disagreement_gain < 0: raise ValueError("disagreement_gain must be non-negative")
        if not 0.0 <= blend_cap <= 1.0: raise ValueError("blend_cap must lie within [0, 1]")

        # ------------------------------------------------------------------
        # Neutral default priors
        # ------------------------------------------------------------------
        if sensor_prior is None:
            sensor_prior = {"lidar": 1.00, "ultrasonic": 1.00, "radar": 1.00}

        if not isinstance(sensor_prior, dict): raise TypeError("sensor_prior must be a dictionary")
        required_sensors = ("lidar", "ultrasonic", "radar")

        for sensor in required_sensors:
            value = sensor_prior.get(sensor, 1.0)
            if not np.isfinite(value): raise ValueError(f"sensor_prior['{sensor}'] must be finite")
            if value <= 0: raise ValueError(f"sensor_prior['{sensor}'] must be greater than 0")

        self.sensor_prior = dict(sensor_prior)
        self.trust_exponent = float(trust_exponent)
        self.weight_floor = float(weight_floor)
        self.balance_strength = float(balance_strength)
        self.disagreement_gain = float(disagreement_gain)
        self.blend_cap = float(blend_cap)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_reliability(value):
        value = np.asarray(value, dtype=float)
        value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(value, 0.0, 1.0)

    def _normalize_priors(self):
        priors = np.array([
            float(self.sensor_prior.get("lidar", 1.0)),
            float(self.sensor_prior.get("ultrasonic", 1.0)),
            float(self.sensor_prior.get("radar", 1.0)),
        ], dtype=float)

        priors = np.nan_to_num(priors, nan=1.0, posinf=1.0, neginf=1.0)
        priors = np.clip(priors, 1e-6, None)
        mean_prior = np.mean(priors)

        if mean_prior <= 1e-12: return np.ones(3, dtype=float)
        return priors / mean_prior

    # ------------------------------------------------------------------
    # Weight-floor handling
    # ------------------------------------------------------------------
    def _apply_weight_floor(self, weights, active_mask=None):
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        if active_mask is None:
            active_mask = weights > 0.0
        else:
            active_mask = np.asarray(active_mask, dtype=bool)

        weights = np.where(active_mask, np.maximum(weights, self.weight_floor), 0.0)
        total = np.sum(weights)

        if total <= 1e-12: return np.zeros(len(weights), dtype=float)
        return weights / total

    # ------------------------------------------------------------------
    # Consensus regularization
    # ------------------------------------------------------------------
    def _consensus_regularize(self, raw_weights, reliabilities, scene_complexity=0):
        raw_weights = np.asarray(raw_weights, dtype=float)
        reliabilities = self._sanitize_reliability(reliabilities)
        raw_weights = np.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)
        scene_complexity = float(np.clip(scene_complexity, 0.0, 1.0))

        # Sensor is active only when its reliability is strictly positive
        active_mask = reliabilities > 0.0
        active_count = int(np.sum(active_mask))

        if active_count == 0: return np.zeros(len(raw_weights), dtype=float)

        # Preserve zero contribution from unavailable sensors
        raw_weights = np.where(active_mask, raw_weights, 0.0)
        total = np.sum(raw_weights)

        if total <= 1e-12:
            raw_weights = np.zeros_like(raw_weights)
            raw_weights[active_mask] = 1.0 / active_count
        else:
            raw_weights = raw_weights / total

        # Uniform distribution only across active sensors
        uniform = np.zeros_like(raw_weights)
        uniform[active_mask] = 1.0 / active_count
        active_reliabilities = reliabilities[active_mask]

        mean_r = float(np.mean(active_reliabilities))
        std_r = float(np.std(active_reliabilities))
        rel_spread = std_r / (mean_r + 1e-6)

        # Reliability disagreement increases regularization
        effective_disagreement = rel_spread * (1.0 + 0.5 * scene_complexity)

        blend = (self.balance_strength
            + self.disagreement_gain * (1.0 - np.exp(-effective_disagreement)))

        blend *= 1.0 + 0.25 * scene_complexity
        blend = float(np.clip(blend, 0.0, self.blend_cap))
        blended = (1.0 - blend) * raw_weights + blend * uniform

        # Reapply active-only floor
        return self._apply_weight_floor(blended, active_mask)

    # ------------------------------------------------------------------
    # Adaptive weight computation
    # ------------------------------------------------------------------
    def compute_adaptive_weights(self, r_lidar, r_ultra, r_radar, scene_complexity=0):
        r_lidar = float(self._sanitize_reliability(r_lidar))
        r_ultra = float(self._sanitize_reliability(r_ultra))
        r_radar = float(self._sanitize_reliability(r_radar))

        scene_complexity = float(np.clip(scene_complexity, 0.0, 1.0))

        reliabilities = np.array([r_lidar, r_ultra, r_radar], dtype=float)
        active_mask = reliabilities > 0.0
        active_count = int(np.sum(active_mask))

        if active_count == 0: return (0.0, 0.0, 0.0)
        priors = self._normalize_priors()

        # Reliability-to-trust transformation
        trust = priors * np.power(reliabilities, self.trust_exponent)
        trust = np.where(active_mask, trust, 0.0)

        # Scenario-complexity adjustment
        lidar_scene = np.exp(-0.12 * scene_complexity)
        ultra_scene = np.exp(-0.11 * scene_complexity)
        radar_scene = np.exp(-0.05 * scene_complexity)

        scene_factors = np.array([lidar_scene, ultra_scene, radar_scene], dtype=float)
        trust *= scene_factors
        trust = np.where(active_mask, trust, 0.0)
        total = np.sum(trust)

        if not np.isfinite(total) or total <= 1e-12:
            raw_weights = np.zeros(3, dtype=float)
            raw_weights[active_mask] = 1.0 / active_count
        else:
            raw_weights = trust / total

        # Disagreement-aware regularization
        weights = self._consensus_regularize(raw_weights, reliabilities, scene_complexity)
        return tuple(float(w) for w in weights)

    # ------------------------------------------------------------------
    # Weight entropy
    # ------------------------------------------------------------------
    def fusion_entropy(self, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = np.sum(weights)

        if total <= 1e-12: return 0.0
        weights = weights / total
        positive = weights > 1e-12
        return float(-np.sum(weights[positive] * np.log(weights[positive])))

    # ------------------------------------------------------------------
    # Effective sensor count
    # ------------------------------------------------------------------
    def effective_sensor_count(self, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = np.sum(weights)

        if total <= 1e-12: return 0.0
        weights = weights / total
        denom = np.sum(np.square(weights))
        if denom <= 1e-12: return 0.0
        return float(1.0 / denom)

    # ------------------------------------------------------------------
    # Fusion confidence diagnostic
    # ------------------------------------------------------------------
    def fusion_confidence(self, weights, reliabilities=None, scene_complexity=0.0):
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = np.sum(weights)

        if total <= 1e-12:
            weights = np.zeros_like(weights)
        else:
            weights = weights / total

        # Weight-balance component
        effective_count = self.effective_sensor_count(weights)
        sensor_count = len(weights)

        if sensor_count <= 0:
            balance = 0.0
        else:
            balance = np.clip(effective_count / sensor_count, 0.0, 1.0)

        # Reliability component
        if reliabilities is None:
            reliability_score = 0.0
        else:
            reliabilities = self._sanitize_reliability(reliabilities)
            active = reliabilities > 0.0

            if np.any(active):
                reliability_score = float(np.mean(reliabilities[active]))
            else:
                reliability_score = 0.0

        scene_complexity = float(np.clip(scene_complexity, 0.0, 1.0))
        confidence = (0.40 * balance + 0.40 * reliability_score + 0.20 * (1.0 - scene_complexity))
        return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def fusion_diagnostics(self, weights, reliabilities=None, scene_complexity=0.0):
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = np.sum(weights)

        if total > 1e-12:
            weights = weights / total
        else:
            weights = np.zeros_like(weights)

        sensor_names = ["lidar", "ultrasonic", "radar"]

        if len(weights) != len(sensor_names):
            raise ValueError("Expected exactly three sensor weights: lidar, ultrasonic, radar")

        active_mask = weights > 1e-12
        active_count = int(np.sum(active_mask))

        # ------------------------------------------------------------------
        # No-active-sensor diagnostic
        # ------------------------------------------------------------------
        if active_count == 0:
            dominant_sensor = "none"
            dominance_ratio = 0.0
        else:
            dominant_idx = int(np.argmax(weights))
            dominant_sensor = sensor_names[dominant_idx]
            other_weights = np.delete(weights, dominant_idx)
            other_active = other_weights > 1e-12

            if np.any(other_active):
                mean_other = float(np.mean(other_weights[other_active]))
                dominance_ratio = float(weights[dominant_idx] / (mean_other + 1e-12))
            else:
                dominance_ratio = float("inf")

        return {
            "weights": tuple(float(w) for w in weights),
            "entropy": self.fusion_entropy(weights),
            "effective_sensor_count": self.effective_sensor_count(weights),
            "dominant_sensor": dominant_sensor,
            "dominance_ratio": dominance_ratio,
            "confidence": self.fusion_confidence(weights, reliabilities, scene_complexity),
            "weight_variance": float(np.var(weights)),
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def metadata(self):
        return {
            "strategy": "Adaptive Reliability-Based Sensor Fusion",
            "method_role": "Reliability-aware sensor weighting; not Bayesian measurement fusion",

            "priors": self.sensor_prior,
            "trust_exponent": self.trust_exponent,
            "weight_floor": self.weight_floor,
            "balance_strength": self.balance_strength,
            "disagreement_gain": self.disagreement_gain,
            "blend_cap": self.blend_cap,

            "diagnostics": [
                "Weight Entropy",
                "Effective Sensor Count",
                "Dominant Sensor",
                "Dominance Ratio",
                "Fusion Confidence Score",
                "Weight Variance",
            ],

            "components": [
                "Reliability-Based Trust Weighting",
                "Sensor Prior Normalization",
                "Scenario-Complexity Adjustment",
                "Disagreement-Aware Consensus Regularization",
                "Active-Sensor Weight Constraint",
            ],

            "assumptions": [
                "Sensor reliabilities are contextual fusion trust scores within "
                "the range [0, 1]",
                "A reliability value of zero denotes an unavailable sensor",
                "Sensor priors represent engineering-defined relative preferences "
                "and are not empirically calibrated probabilities",
                "The trust exponent provides nonlinear emphasis to higher-reliability "
                "sensors",
                "Scenario-complexity adjustments use engineering-defined "
                "sensor-specific coefficients",
                "Consensus regularization limits excessive dominance by a single "
                "active sensor",
                "The minimum weight constraint applies only to active sensors "
                "and never resurrects an unavailable sensor",
                "Fusion confidence is a normalized engineering diagnostic and "
                "is not a calibrated probability of estimation correctness",
                "Fusion entropy and effective sensor count describe weight "
                "distribution rather than estimation accuracy",
            ],

            "scientific_limitations": [
                "The weighting coefficients are phenomenological and require "
                "sensitivity and ablation analysis",
                "Reliability-weighted fusion should not be described as "
                "conventional Bayesian fusion",
                "The trust exponent, scene factors, and consensus regularization "
                "require empirical justification through controlled experiments",
                "Confidence is a diagnostic score and should not be interpreted "
                "as statistical calibration",
            ],
        }

# Backward-compatible module-level interface
fusion_engine = AdaptiveReliabilityWeighting()

def compute_adaptive_weights(r_lidar, r_ultra, r_radar, scene_complexity=0):
    return fusion_engine.compute_adaptive_weights(r_lidar, r_ultra, r_radar, scene_complexity)

def compute_diagnostics(r_lidar, r_ultra, r_radar, scene_complexity=0):
    weights = compute_adaptive_weights(r_lidar, r_ultra, r_radar, scene_complexity)
    return fusion_engine.fusion_diagnostics(weights, [r_lidar, r_ultra, r_radar], scene_complexity)