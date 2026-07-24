import numpy as np

class ReliabilityFusion:
    
    def __init__(self,
        sensor_prior=None,
        trust_exponent=1.20,
        saturation_beta=1.10,
        weight_floor=0.025,
        balance_strength=0.05,
        disagreement_gain=0.18,
        blend_cap=0.28):

        # Neutral default priors in use until custom priors are provided
        self.sensor_prior = sensor_prior or { "lidar": 1.00, "ultrasonic": 1.00, "radar": 1.00 }
        self.trust_exponent = float(trust_exponent)
        self.saturation_beta = float(saturation_beta)
        self.weight_floor = float(weight_floor)

        # Conservative regularization terms
        self.balance_strength = float(balance_strength)
        self.disagreement_gain = float(disagreement_gain)
        self.blend_cap = float(blend_cap)

    def _sanitize_reliability(self, value):
        value = np.asarray(value, dtype=float)
        value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(value, 0.0, 1.0)

    def _normalize_priors(self):
        priors = np.array([
            float(self.sensor_prior.get("lidar", 1.0)),
            float(self.sensor_prior.get("ultrasonic", 1.0)),
            float(self.sensor_prior.get("radar", 1.0))], dtype=float)

        priors = np.nan_to_num(priors, nan=1.0, posinf=1.0, neginf=1.0)
        priors = np.clip(priors, 1e-6, None)
        priors = priors / np.mean(priors)
        return priors

    def _saturate_reliability(self, reliability):
        reliability = self._sanitize_reliability(reliability)
        sat = np.tanh(self.saturation_beta * reliability)
        return np.clip(sat, 0.0, 1.0)

    def _apply_weight_floor(self, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(weights) <= 1e-12:
            return np.array([1/3, 1/3, 1/3], dtype=float)

        weights = np.maximum(weights, self.weight_floor)
        weights_sum = np.sum(weights)
        if weights_sum <= 1e-12:
            return np.array([1/3, 1/3, 1/3], dtype=float)
        return weights / weights_sum

    def _consensus_regularize(self, raw_weights, reliabilities, scene_complexity=0):        
        raw_weights = np.asarray(raw_weights, dtype=float)
        reliabilities = np.asarray(reliabilities, dtype=float)

        raw_weights = np.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)
        reliabilities = np.nan_to_num(reliabilities, nan=0.0, posinf=1.0, neginf=0.0)
        total = np.sum(raw_weights)        
        if total <= 1e-12:
            return np.array([1/3, 1/3, 1/3], dtype=float)

        raw_weights = raw_weights / total
        uniform = np.full_like(raw_weights, 1.0 / len(raw_weights), dtype=float)

        mean_r = float(np.mean(reliabilities))
        std_r = float(np.std(reliabilities))
        rel_spread = std_r / (mean_r + 1e-6)

        # Small blend in easy scenes, stronger blend when one sensor is far above the others
        effective_disagreement = rel_spread * (1 + 0.5 * scene_complexity)
        blend = (self.balance_strength + self.disagreement_gain * (1 - np.exp(-effective_disagreement)))
        blend *= (1 + 0.25 * scene_complexity)
        blend = float(np.clip(blend,0,self.blend_cap))
        blended = (1.0 - blend) * raw_weights + blend * uniform
        return self._apply_weight_floor(blended)

    def compute_adaptive_weights(self, r_lidar, r_ultra, r_radar, scene_complexity=0):
        r_lidar = self._sanitize_reliability(r_lidar)
        r_ultra = self._sanitize_reliability(r_ultra)
        r_radar = self._sanitize_reliability(r_radar)

        priors = self._normalize_priors()

        sat_lidar = self._saturate_reliability(r_lidar)
        sat_ultra = self._saturate_reliability(r_ultra)
        sat_radar = self._saturate_reliability(r_radar)

        trust_lidar = priors[0] * np.power(sat_lidar, self.trust_exponent)
        trust_ultra = priors[1] * np.power(sat_ultra, self.trust_exponent)
        trust_radar = priors[2] * np.power(sat_radar, self.trust_exponent)

        lidar_scene = np.exp(-0.12*scene_complexity)
        ultra_scene = np.exp(-0.11*scene_complexity)
        radar_scene = np.exp(-0.05*scene_complexity)

        trust_lidar *= lidar_scene
        trust_ultra *= ultra_scene
        trust_radar *= radar_scene

        total = trust_lidar + trust_ultra + trust_radar

        if not np.isfinite(total) or total <= 1e-12:
            weights = np.array([1/3, 1/3, 1/3], dtype=float)
        else:
            weights = np.array([trust_lidar / total, trust_ultra / total, trust_radar / total], dtype=float)

        weights = self._consensus_regularize(weights,
            np.array([r_lidar, r_ultra, r_radar], dtype=float), scene_complexity)
        return tuple(float(w) for w in weights)

    def fusion_entropy(self, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 1e-12, 1.0)
        weights = weights / np.sum(weights)
        return float(-np.sum(weights * np.log(weights)))

    def effective_sensor_count(self, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.sum(np.square(weights))
        if denom <= 1e-12:
            return 3.0
        return float(1.0 / denom)

    def fusion_confidence(self, weights, reliabilities=None, scene_complexity=0.0):
        weights = np.asarray(weights, dtype = float)
        total = np.sum(weights)
        if total <= 1e-12:
            weights = np.array([1/3, 1/3, 1/3], dtype=float)
        else: weights /= total
        balance = self.effective_sensor_count(weights) / len(weights)
        
        if reliabilities is None:
            reliability_score = 1.0
        else:
            reliability_score = float(np.mean(reliabilities))

        confidence = (0.40 * balance + 0.40 * reliability_score + 0.20 * (1 - scene_complexity))
        return float(np.clip(confidence,0,1))

    def fusion_diagnostics(self, weights, reliabilities=None, scene_complexity=0.0):
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights / np.sum(weights) if np.sum(weights) > 1e-12 else np.array([1/3, 1/3, 1/3], dtype=float)

        dominant_idx = int(np.argmax(weights))
        sensor_names = ["lidar", "ultrasonic", "radar"]
        dominant_sensor = sensor_names[dominant_idx]
        dominance_ratio = float(np.max(weights) / (np.mean(np.delete(weights, dominant_idx)) + 1e-12))

        return {
            "weights": tuple(float(w) for w in weights),
            "entropy": self.fusion_entropy(weights),
            "effective_sensor_count": self.effective_sensor_count(weights),
            "dominant_sensor": dominant_sensor,
            "dominance_ratio": dominance_ratio,
            "confidence": self.fusion_confidence(weights, reliabilities, scene_complexity),
            "weight_variance": float(np.var(weights)),
        }

    def metadata(self):
        return {
            "strategy": "Adaptive Reliability Fusion",
            "priors": self.sensor_prior,
            "trust_exponent": self.trust_exponent,
            "saturation_beta": self.saturation_beta,
            "weight_floor": self.weight_floor,
            "balance_strength": self.balance_strength,
            "disagreement_gain": self.disagreement_gain,
            "blend_cap": self.blend_cap,

            "diagnostics": [
                "Weight Entropy",
                "Effective Sensor Count",
                "Dominant Sensor",
                "Dominance Ratio",
                "Fusion Confidence",
                "Weight Variance",
            ],
        }

fusion_engine = ReliabilityFusion()

def compute_adaptive_weights(r_lidar, r_ultra, r_radar, scene_complexity=0):
    return fusion_engine.compute_adaptive_weights(r_lidar, r_ultra, r_radar, scene_complexity)

def compute_confidence(r_lidar, r_ultra, r_radar, scene_complexity=0):
    weights = compute_adaptive_weights(r_lidar, r_ultra, r_radar, scene_complexity)
    return fusion_engine.fusion_confidence(weights, [r_lidar, r_ultra, r_radar], scene_complexity)

def compute_diagnostics(r_lidar, r_ultra, r_radar, scene_complexity=0):
    weights = compute_adaptive_weights(r_lidar, r_ultra, r_radar, scene_complexity)
    return fusion_engine.fusion_diagnostics(weights, [r_lidar, r_ultra, r_radar], scene_complexity)