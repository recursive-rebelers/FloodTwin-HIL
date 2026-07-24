import numpy as np
from pathlib import Path
import sys
from typing import Optional, Dict, Sequence, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion_engine.bayesian_fusion import BayesianFusionCore

class AdaptiveFusionEngine:

    def __init__(self,
        core: BayesianFusionCore,
        gamma: float = 1.75,
        weight_blend: float = 0.60,
        sigma_scale: float = 1.00,
        sigma_offset: float = 0.08,
        sigma_min: float = 0.45,
        sigma_max: float = 2.85,
        sensor_prior: Optional[Dict[str, float]] = None,
        context_blend_boost: float = 0.28,
        context_sigma_boost: float = 0.45,
        weight_floor: float = 0.01,
        adaptation_threshold: float = 0.42,
        transition_width: float = 0.14,
        quality_threshold: float = 0.24,
        quality_margin: float = 0.05,
        sensor_health: Optional[Dict[str, float]] = None,
        health_floor: float = 0.08) -> None:

        self.core = core
        self.gamma = float(gamma)
        self.weight_blend = float(weight_blend)
        self.sigma_scale = float(sigma_scale)
        self.sigma_offset = float(sigma_offset)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sensor_prior = sensor_prior or {"lidar": 1.00, "ultrasonic": 1.00, "radar": 1.00}
        self.context_blend_boost = float(context_blend_boost)
        self.context_sigma_boost = float(context_sigma_boost)
        self.weight_floor = float(weight_floor)
        self.adaptation_threshold = float(adaptation_threshold)
        self.transition_width = float(transition_width)
        self.quality_threshold = float(quality_threshold)
        self.quality_margin = float(quality_margin)
        self.sensor_health = sensor_health or {"lidar": 1.00, "ultrasonic": 1.00, "radar": 1.00}
        self.health_floor = float(health_floor)

    def _sanitize_scalar(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(value):
            return float(default)
        return float(value)

    def _clip01(self, value: Any, default: float = 0.0) -> float:
        return float(np.clip(self._sanitize_scalar(value, default), 0.0, 1.0))

    def _normalize_priors(self) -> np.ndarray:
        priors = np.array([
            float(self.sensor_prior.get("lidar", 1.0)),
            float(self.sensor_prior.get("ultrasonic", 1.0)),
            float(self.sensor_prior.get("radar", 1.0))], dtype=float)

        priors = np.nan_to_num(priors, nan=1.0, posinf=1.0, neginf=1.0)
        priors = np.clip(priors, 1e-6, None)
        return priors / np.mean(priors)

    def _normalize_health(self, active_mask: Optional[Sequence[bool]] = None) -> np.ndarray:
        health = np.array([
            float(self.sensor_health.get("lidar", 1.0)),
            float(self.sensor_health.get("ultrasonic", 1.0)),
            float(self.sensor_health.get("radar", 1.0))], dtype=float)

        health = np.nan_to_num(health, nan=1.0, posinf=1.0, neginf=1.0)
        health = np.clip(health, 0.0, 1.0)

        if active_mask is not None:
            mask = np.asarray(active_mask, dtype=bool)
            if mask.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")
            health = np.where(mask, health, 0.0)
        return health

    def _apply_weight_floor(self, weights: Sequence[float]) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(weights) <= 1e-12:
            return np.array([1/3, 1/3, 1/3], dtype=float)

        weights = np.maximum(weights, self.weight_floor)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 1e-12:
            return np.array([1/3, 1/3, 1/3], dtype=float)
        return weights / total

    def _apply_active_mask(self, weights: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).copy()
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape != (3,):
            raise ValueError("active_mask must have shape (3,)")

        weights = np.where(active_mask, weights, 0.0)
        active_count = int(np.sum(active_mask))
        if active_count <= 0:
            return np.array([1/3, 1/3, 1/3], dtype=float)

        active_weights = weights[active_mask]
        active_weights = np.maximum(active_weights, self.weight_floor)
        total = float(np.sum(active_weights))
        if not np.isfinite(total) or total <= 1e-12:
            active_weights = np.full(active_count, 1.0 / active_count, dtype=float)
        else:
            active_weights = active_weights / total

        weights[:] = 0.0
        weights[active_mask] = active_weights
        return weights

    def _effective_sensor_count(self, weights: Sequence[float]) -> float:
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(weights))
        if total <= 1e-12:
            return 3.0

        weights = weights / total
        denom = float(np.sum(np.square(weights)))
        if denom <= 1e-12:
            return 3.0
        return float(1.0 / denom)

    def _fusion_entropy(self, weights: Sequence[float]) -> float:
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 1e-12, 1.0)
        weights = weights / np.sum(weights)
        return float(-np.sum(weights * np.log(weights)))

    def _dominance_ratio(self, weights: Sequence[float]) -> float:
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(weights) <= 1e-12:
            return 1.0

        weights = weights / np.sum(weights)
        dominant_idx = int(np.argmax(weights))
        dominant_val = float(np.max(weights))
        others = np.delete(weights, dominant_idx)
        if len(others) == 0:
            return 1.0
        return float(dominant_val / (float(np.mean(others)) + 1e-12))

    def _context_difficulty(self,
        scene_complexity: float = 0.0,
        sensor_agreement: Optional[float] = None,
        reliability_spread: Optional[float] = None,
        effective_sensor_count: Optional[float] = None,
        dominance_ratio: Optional[float] = None,
        confidence: Optional[float] = None,
        measurement_spread: Optional[float] = None,
        water_depth: Optional[float] = None,
        ntu: Optional[float] = None) -> float:

        scene = self._clip01(scene_complexity, 0.0)
        agreement = self._clip01(sensor_agreement, 0.5)
        spread = self._clip01(reliability_spread, 0.0)
        conf = self._clip01(confidence, 0.5)
        water = self._clip01(0.0 if water_depth is None else self._sanitize_scalar(water_depth, 0.0) / 20.0, 0.0)
        turb = self._clip01(0.0 if ntu is None else self._sanitize_scalar(ntu, 0.0) / 500.0, 0.0)

        if effective_sensor_count is None:
            eff_score = 1.0
        else:
            eff = self._sanitize_scalar(effective_sensor_count, 3.0)
            eff_score = float(np.clip(eff / 3.0, 0.0, 1.0))

        if dominance_ratio is None:
            dom_score = 0.0
        else:
            dom = self._sanitize_scalar(dominance_ratio, 1.0)
            dom_score = float(np.clip((dom - 1.0) / 2.0, 0.0, 1.0))

        if measurement_spread is None:
            meas_score = 0.0
        else:
            meas = self._sanitize_scalar(measurement_spread, 0.0)
            meas_score = float(np.clip(meas / 5.0, 0.0, 1.0))

        difficulty = (0.28 * scene
            + 0.18 * (1.0 - agreement)
            + 0.12 * spread
            + 0.10 * (1.0 - eff_score)
            + 0.08 * dom_score
            + 0.08 * meas_score
            + 0.08 * water
            + 0.08 * turb
            + 0.08 * (1.0 - conf))

        return float(np.clip(difficulty, 0.0, 1.0))

    def _adaptation_gate(self, difficulty: float) -> float:
        difficulty = self._clip01(difficulty, 0.0)
        x = (difficulty - self.adaptation_threshold) / max(self.transition_width, 1e-6)
        gate = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(gate, 0.0, 1.0))

    def _measurement_consistency(self, z_lidar: float, z_ultra: float, z_radar: float) -> np.ndarray:
        measurements = np.asarray([z_lidar, z_ultra, z_radar], dtype=float)
        if not np.isfinite(np.nanmedian(measurements)):
            return np.array([1/3, 1/3, 1/3], dtype=float)

        measurements = np.nan_to_num(measurements, nan=np.nanmedian(measurements))
        consensus = float(np.median(measurements))
        deviations = np.abs(measurements - consensus)
        mad = float(np.median(deviations))
        scale = max(1.4826 * mad, 0.35)

        consistency = np.exp(-np.square(deviations / scale))
        return np.clip(consistency, 0.05, 1.0)

    def _compute_sensor_quality(self, reliability: float, sigma: float, consistency: float, 
        health: float = 1.0, active: bool = True) -> float:
        if not active: return 0.0
        reliability = self._clip01(reliability, 0.0)
        consistency = self._clip01(consistency, 0.5)
        health = self._clip01(health, 1.0)
        sigma_score = 1.0 - np.clip((sigma - self.sigma_min) / (self.sigma_max - self.sigma_min), 0.0, 1.0)
        quality = (0.42 * reliability + 0.22 * consistency + 0.16 * sigma_score + 0.20 * health)
        return float(np.clip(quality, 0.0, 1.0))

    def _adaptive_quality_threshold(self, difficulty: float, agreement: float, consistency: float, reliability_spread: float) -> float:
        threshold = (self.quality_threshold
            + 0.12 * difficulty
            + 0.08 * (1.0 - agreement)
            + 0.08 * (1.0 - consistency)
            + 0.05 * reliability_spread)
        return float(np.clip(threshold, 0.18, 0.45))

    def _normalize_weights(self,
        r_lidar: float, r_ultra: float, r_radar: float,
        z_lidar: Optional[float] = None,
        z_ultra: Optional[float] = None,
        z_radar: Optional[float] = None,
        scene_complexity: float = 0.0,
        sensor_agreement: Optional[float] = None,
        reliability_spread: Optional[float] = None,
        effective_sensor_count: Optional[float] = None,
        dominance_ratio: Optional[float] = None,
        confidence: Optional[float] = None,
        measurement_spread: Optional[float] = None,
        water_depth: Optional[float] = None,
        ntu: Optional[float] = None,
        difficulty: Optional[float] = None,
        active_mask: Optional[Sequence[bool]] = None,
        sensor_health: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float, float]:

        reliabilities = np.array([r_lidar, r_ultra, r_radar], dtype=float)
        reliabilities = np.clip(np.nan_to_num(reliabilities, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if active_mask is None:
            active_mask_arr = np.array([True, True, True], dtype=bool)
        else:
            active_mask_arr = np.asarray(active_mask, dtype=bool)
            if active_mask_arr.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")

        health_arr = np.array(sensor_health if sensor_health is not None else [1.0, 1.0, 1.0], dtype=float)
        health_arr = np.nan_to_num(health_arr, nan=1.0, posinf=1.0, neginf=1.0)
        health_arr = np.clip(health_arr, 0.0, 1.0)
        health_arr = np.where(active_mask_arr, health_arr, 0.0)

        reliabilities = np.where(active_mask_arr, reliabilities, 0.0)
        priors = self._normalize_priors()

        trust = np.array([
            self._sensor_trust_strength("lidar", reliabilities[0],
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth,
                ntu=ntu, health=health_arr[0]),

            self._sensor_trust_strength("ultrasonic", reliabilities[1],
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth,
                ntu=ntu, health=health_arr[1]),

            self._sensor_trust_strength("radar", reliabilities[2],
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth,
                ntu=ntu, health=health_arr[2])], dtype=float)

        trust *= priors
        trust *= np.where(active_mask_arr, 1.0, 0.0)
        trust *= (self.health_floor + (1.0 - self.health_floor) * health_arr)

        if z_lidar is not None and z_ultra is not None and z_radar is not None:
            consistency = self._measurement_consistency(z_lidar, z_ultra, z_radar)
            trust *= np.power(consistency, 0.65)

        trust = np.where(active_mask_arr, np.maximum(trust, self.weight_floor), 0.0)
        total = float(np.sum(trust))

        if not np.isfinite(total) or total <= 1e-12:
            raw_weights = np.array([1/3, 1/3, 1/3], dtype=float)
        else:
            raw_weights = trust / total

        if difficulty is None:
            difficulty = self._context_difficulty(
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth, ntu=ntu)

        agreement = self._clip01(sensor_agreement, 0.5)
        balance_score = self._effective_sensor_count(raw_weights) / 3.0
        gate = self._adaptation_gate(difficulty)

        retain_raw = self.weight_blend + gate * (
            0.12 * self.context_blend_boost * difficulty
            + 0.05 * (1.0 - balance_score)
            - 0.03 * (agreement - 0.5))

        retain_raw = float(np.clip(retain_raw, 0.22, 0.78))
        uniform = np.full(3, 1.0 / 3.0, dtype=float)
        weights = retain_raw * raw_weights + (1.0 - retain_raw) * uniform
        weights = self._apply_weight_floor(weights)
        weights = self._apply_active_mask(weights, active_mask_arr)
        return tuple(float(w) for w in weights)

    def _infer_sigmas(self,
        r_lidar: float, r_ultra: float, r_radar: float,
        z_lidar: Optional[float] = None,
        z_ultra: Optional[float] = None,
        z_radar: Optional[float] = None,
        scene_complexity: float = 0.0,
        sensor_agreement: Optional[float] = None,
        reliability_spread: Optional[float] = None,
        effective_sensor_count: Optional[float] = None,
        dominance_ratio: Optional[float] = None,
        confidence: Optional[float] = None,
        measurement_spread: Optional[float] = None,
        water_depth: Optional[float] = None,
        ntu: Optional[float] = None,
        difficulty: Optional[float] = None,
        measurement_consistency: Optional[np.ndarray] = None,
        active_mask: Optional[Sequence[bool]] = None,
        sensor_health: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float, float]:

        reliabilities = np.array([r_lidar, r_ultra, r_radar], dtype=float)
        reliabilities = np.clip(reliabilities, 1e-3, 1.0)

        if active_mask is None:
            active_mask_arr = np.array([True, True, True], dtype=bool)
        else:
            active_mask_arr = np.asarray(active_mask, dtype=bool)
            if active_mask_arr.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")

        health_arr = np.array(sensor_health if sensor_health is not None else [1.0, 1.0, 1.0], dtype=float)
        health_arr = np.nan_to_num(health_arr, nan=1.0, posinf=1.0, neginf=1.0)
        health_arr = np.clip(health_arr, 0.0, 1.0)
        health_arr = np.where(active_mask_arr, health_arr, 0.0)
        base_sigmas = self.sigma_scale / (np.sqrt(reliabilities) + self.sigma_offset)

        if difficulty is None:
            difficulty = self._context_difficulty(
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth, ntu=ntu)

        agreement = self._clip01(sensor_agreement, 0.5)
        spread = self._clip01(reliability_spread, 0.0)
        meas_norm = 0.0 if measurement_spread is None else float(
            np.clip(self._sanitize_scalar(measurement_spread, 0.0) / 5.0, 0.0, 1.0))

        water = self._clip01(0.0 if water_depth is None else self._sanitize_scalar(water_depth, 0.0) / 20.0, 0.0)
        turb = self._clip01(0.0 if ntu is None else self._sanitize_scalar(ntu, 0.0) / 500.0, 0.0)

        gate = self._adaptation_gate(difficulty)
        context_multiplier = 1.0 + gate * (self.context_sigma_boost * difficulty + 0.05 * spread)
        sigmas = np.array(base_sigmas, dtype=float) * context_multiplier

        if measurement_consistency is None and (z_lidar is not None and z_ultra is not None and z_radar is not None):
            measurement_consistency = self._measurement_consistency(z_lidar, z_ultra, z_radar)

        if measurement_consistency is not None:
            sigma_penalty = 1.0 + 0.22 * (1.0 - np.asarray(measurement_consistency, dtype=float))
            sigmas *= sigma_penalty

        sigmas *= np.where(active_mask_arr, 1.0, self.sigma_max)
        sigmas[0] *= 1.0 + gate * (
            0.08 * difficulty + 0.08 * meas_norm + 0.06 * turb + 0.05 * water + 0.03 * (1.0 - agreement))
        sigmas[1] *= 1.0 + gate * (
            0.08 * difficulty + 0.10 * meas_norm + 0.10 * water + 0.03 * (1.0 - agreement))
        sigmas[2] *= 1.0 + gate * (
            0.07 * difficulty + 0.12 * meas_norm + 0.06 * water + 0.04 * turb + 0.03 * (1.0 - agreement))

        sigmas = np.clip(sigmas, self.sigma_min, self.sigma_max)
        sigmas = np.where(active_mask_arr, sigmas, self.sigma_max)
        return tuple(float(s) for s in sigmas)

    def _sensor_trust_strength(self,
        sensor_name: str,
        reliability: float,
        scene_complexity: float = 0.0,
        sensor_agreement: Optional[float] = None,
        reliability_spread: Optional[float] = None,
        effective_sensor_count: Optional[float] = None,
        dominance_ratio: Optional[float] = None,
        confidence: Optional[float] = None,
        measurement_spread: Optional[float] = None,
        water_depth: Optional[float] = None,
        ntu: Optional[float] = None,
        health: float = 1.0) -> float:

        reliability = float(np.clip(reliability, 0.0, 1.0))
        agreement = self._clip01(sensor_agreement, 0.5)
        spread = self._clip01(reliability_spread, 0.0)
        scene = self._clip01(scene_complexity, 0.0)
        conf = self._clip01(confidence, 0.5)
        water = self._clip01(0.0 if water_depth is None else self._sanitize_scalar(water_depth, 0.0) / 20.0, 0.0)
        turb = self._clip01(0.0 if ntu is None else self._sanitize_scalar(ntu, 0.0) / 500.0, 0.0)
        health = self._clip01(health, 1.0)

        if effective_sensor_count is None:
            eff_bonus = 1.0
        else:
            eff = self._sanitize_scalar(effective_sensor_count, 3.0)
            eff_bonus = float(np.clip(eff / 3.0, 0.6, 1.0))

        if dominance_ratio is None:
            dom_penalty = 0.0
        else:
            dom = self._sanitize_scalar(dominance_ratio, 1.0)
            dom_penalty = float(np.clip((dom - 1.0) / 3.0, 0.0, 0.45))

        meas = 0.0 if measurement_spread is None else float(
            np.clip(self._sanitize_scalar(measurement_spread, 0.0) / 5.0, 0.0, 1.0))
        sharpened = np.power(reliability, self.gamma)

        if sensor_name == "lidar":
            context = (1.0
                + 0.10 * agreement
                - 0.12 * scene
                - 0.08 * turb
                - 0.06 * water
                - 0.03 * meas
                - 0.04 * spread
                - 1.00 * dom_penalty)

        elif sensor_name == "ultrasonic":
            context = (1.0
                + 0.08 * agreement
                - 0.10 * scene
                - 0.10 * water
                - 0.05 * meas
                - 0.04 * spread
                - 0.50 * dom_penalty)

        elif sensor_name == "radar":
            context = (1.0
                - 0.01 * agreement
                + 0.09 * scene
                - 0.01 * turb
                - 0.01 * water
                - 0.04 * meas
                + 0.04 * spread
                - 0.25 * dom_penalty)

        context = float(np.clip(context, 0.55, 1.55))
        trust = (0.92 + 0.08 * eff_bonus) * sharpened * context * (0.98 + 0.02 * conf)
        return float(np.clip(trust, 1e-6, 10.0))

    def _fusion_confidence(self, posterior, weights, scene_complexity: float = 0.0) -> float:
        posterior_mass = self.core.posterior_mass(posterior)
        entropy = self.core.entropy(posterior)
        max_entropy = float(np.log(len(posterior_mass)))
        entropy_score = 0.0 if max_entropy <= 1e-12 else float(np.clip(1.0 - (entropy / max_entropy), 0.0, 1.0))

        variance = self.core.posterior_variance(posterior)
        span = max(self.core.depth_max - self.core.depth_min, 1e-6)
        variance_scale = max(0.12 * span * span, 1e-6)
        variance_score = float(np.clip(np.exp(-variance / variance_scale), 0.0, 1.0))

        peak_score = float(np.clip(np.max(posterior_mass) / 0.05, 0.0, 1.0))
        ci_lower, ci_upper = self.core.credible_interval(posterior, mass_level=0.95)
        interval_score = float(np.clip(1.0 - ((ci_upper - ci_lower) / span), 0.0, 1.0))

        balance_score = self._effective_sensor_count(weights) / 3.0
        scene_score = 1.0 - self._clip01(scene_complexity, 0.0)

        confidence = (
            0.30 * entropy_score
            + 0.18 * variance_score
            + 0.20 * peak_score
            + 0.12 * interval_score
            + 0.20 * balance_score * scene_score)

        return float(np.clip(confidence, 0.0, 1.0))

    def fuse(self,
        z_lidar, z_ultra, z_radar,
        r_lidar, r_ultra, r_radar,
        sigmas=None,
        scene_complexity: float = 0.0,
        sensor_agreement: Optional[float] = None,
        reliability_spread: Optional[float] = None,
        effective_sensor_count: Optional[float] = None,
        dominance_ratio: Optional[float] = None,
        confidence: Optional[float] = None,
        measurement_spread: Optional[float] = None,
        water_depth: Optional[float] = None,
        ntu: Optional[float] = None,
        active_mask: Optional[Sequence[bool]] = None,
        sensor_health: Optional[Dict[str, float]] = None):

        difficulty = self._context_difficulty(
            scene_complexity=scene_complexity,
            sensor_agreement=sensor_agreement,
            reliability_spread=reliability_spread,
            effective_sensor_count=effective_sensor_count,
            dominance_ratio=dominance_ratio,
            confidence=confidence,
            measurement_spread=measurement_spread,
            water_depth=water_depth, ntu=ntu)

        if z_lidar is not None and z_ultra is not None and z_radar is not None:
            measurement_consistency = self._measurement_consistency(z_lidar, z_ultra, z_radar)
        else:
            measurement_consistency = None

        if active_mask is None:
            active_mask_arr = np.array([True, True, True], dtype=bool)
        else:
            active_mask_arr = np.asarray(active_mask, dtype=bool)
            if active_mask_arr.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")

        sensor_health_arr = np.array([
            float((sensor_health or self.sensor_health).get("lidar", 1.0)),
            float((sensor_health or self.sensor_health).get("ultrasonic", 1.0)),
            float((sensor_health or self.sensor_health).get("radar", 1.0))], dtype=float)

        sensor_health_arr = np.nan_to_num(sensor_health_arr, nan=1.0, posinf=1.0, neginf=1.0)
        sensor_health_arr = np.clip(sensor_health_arr, 0.0, 1.0)
        sensor_health_arr = np.where(active_mask_arr, sensor_health_arr, 0.0)

        w_lidar, w_ultra, w_radar = self._normalize_weights(
            r_lidar, r_ultra, r_radar,
            z_lidar=z_lidar, z_ultra=z_ultra, z_radar=z_radar,
            scene_complexity=scene_complexity,
            sensor_agreement=sensor_agreement,
            reliability_spread=reliability_spread,
            effective_sensor_count=effective_sensor_count,
            dominance_ratio=dominance_ratio,
            confidence=confidence,
            measurement_spread=measurement_spread,
            water_depth=water_depth,
            ntu=ntu,
            difficulty=difficulty,
            active_mask=active_mask_arr,
            sensor_health=sensor_health_arr)

        if sigmas is None:
            sigmas = self._infer_sigmas(
                r_lidar, r_ultra, r_radar,
                z_lidar=z_lidar, z_ultra=z_ultra, z_radar=z_radar,
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth,
                ntu=ntu, difficulty=difficulty,
                measurement_consistency=measurement_consistency,
                active_mask=active_mask_arr,
                sensor_health=sensor_health_arr)
        else:
            sigmas = tuple(float(np.clip(float(s), self.sigma_min, self.sigma_max)) for s in sigmas)

        raw_weights = np.array([w_lidar, w_ultra, w_radar], dtype=float).copy()
        consistency_values = measurement_consistency if measurement_consistency is not None else np.ones(3)

        quality_scores = np.array([
            self._compute_sensor_quality(
                r_lidar, sigmas[0],
                consistency_values[0],
                health=sensor_health_arr[0],
                active=active_mask_arr[0]),
                
            self._compute_sensor_quality(
                r_ultra, sigmas[1],
                consistency_values[1],
                health=sensor_health_arr[1],
                active=active_mask_arr[1]),
                
            self._compute_sensor_quality(
                r_radar, sigmas[2],
                consistency_values[2],
                health=sensor_health_arr[2],
                active=active_mask_arr[2])], dtype=float)

        agreement = 0.5 if sensor_agreement is None else sensor_agreement
        consistency_mean = 1.0 if measurement_consistency is None else float(np.mean(measurement_consistency))
        spread = 0.0 if reliability_spread is None else reliability_spread

        adaptive_threshold = self._adaptive_quality_threshold(
            difficulty=difficulty,
            agreement=agreement,
            consistency=consistency_mean,
            reliability_spread=spread)

        weights = raw_weights.copy()
        quality_gate_scores = quality_scores * (self.health_floor + (1.0 - self.health_floor) * sensor_health_arr)

        for i in range(3):
            if not active_mask_arr[i]:
                weights[i] = 0.0
                continue
            q = quality_gate_scores[i]

            if q < adaptive_threshold:
                weights[i] *= self.weight_floor
            elif q < adaptive_threshold + self.quality_margin:
                scale = (q-adaptive_threshold) / self.quality_margin
                weights[i] *= self.weight_floor + 0.95 * scale ** 2

        pre_quality_weights = raw_weights.copy()
        post_quality_weights = weights.copy()
        final_mask = weights >= self.weight_floor

        if not np.any(final_mask):
            active_candidates = np.where(active_mask_arr)[0]
            if len(active_candidates) > 0:
                best_idx = int(active_candidates[np.argmax(quality_gate_scores[active_candidates])])
                final_mask = np.zeros(3, dtype=bool)
                final_mask[best_idx] = True
                weights = np.zeros(3, dtype=float)
                weights[best_idx] = 1.0
            else:
                final_mask = np.array([True, True, True], dtype=bool)
                weights = np.array([1/3, 1/3, 1/3], dtype=float)

        if np.sum(weights) > 1e-12:
            weights /= np.sum(weights)
        else:
            weights[:] = 1.0 / 3.0

        w_lidar, w_ultra, w_radar = weights
        final_weights = weights.copy()

        l_lidar = self.core.compute_likelihood(z_lidar, sigmas[0]) if final_mask[0] else None
        l_ultra = self.core.compute_likelihood(z_ultra, sigmas[1]) if final_mask[1] else None
        l_radar = self.core.compute_likelihood(z_radar, sigmas[2]) if final_mask[2] else None

        likelihoods = [l for l in (l_lidar, l_ultra, l_radar) if l is not None]
        likelihood_weights = [float(w) for w, ok in zip((w_lidar, w_ultra, w_radar), final_mask) if ok]

        if len(likelihoods) == 0:
            likelihoods = [self.core.compute_likelihood(0.0, 1.0)]
            likelihood_weights = [1.0]

        pooled_likelihood = self.core.combine_likelihoods(likelihoods, likelihood_weights)
        posterior = self.core.compute_posterior(pooled_likelihood)

        depth_map = self.core.map_estimate(posterior)
        depth_mean = self.core.expected_depth(posterior)
        variance = self.core.posterior_variance(posterior)
        entropy = self.core.entropy(posterior)

        posterior_mass = self.core.posterior_mass(posterior)
        map_idx = int(np.argmax(posterior_mass))
        posterior_peak = float(np.max(posterior_mass))
        ci_lower, ci_upper = self.core.credible_interval(posterior, mass_level=0.95)

        diagnostics = {
            "effective_sensor_count": float(self._effective_sensor_count((w_lidar, w_ultra, w_radar))),
            "dominance_ratio": float(self._dominance_ratio((w_lidar, w_ultra, w_radar))),
            "weight_entropy": float(self._fusion_entropy((w_lidar, w_ultra, w_radar))),
            "confidence": float(self._fusion_confidence(posterior, (w_lidar, w_ultra, w_radar), scene_complexity)),
            "posterior_peak": posterior_peak,
            "map_index": map_idx,
            "credible_interval_95": (float(ci_lower), float(ci_upper)),
            "scene_complexity": float(self._clip01(scene_complexity, 0.0)),
            "sensor_agreement": None if sensor_agreement is None else float(self._clip01(sensor_agreement, 0.5)),
            "reliability_spread": None if reliability_spread is None else float(self._clip01(reliability_spread, 0.0)),
            "measurement_spread": 
            None if measurement_spread is None else float(self._sanitize_scalar(measurement_spread, 0.0)),
            "context_difficulty": float(difficulty),
            "adaptation_gate": float(self._adaptation_gate(difficulty)),
            "measurement_consistency_mean": 
            float(np.mean(measurement_consistency)) if measurement_consistency is not None else None,
            "active_sensor_count": int(np.sum(final_mask)),

            "sensor_health": {
                "lidar": float(sensor_health_arr[0]),
                "ultrasonic": float(sensor_health_arr[1]),
                "radar": float(sensor_health_arr[2]),
            },

            "quality_gate": {
                "quality_threshold": adaptive_threshold,
                "quality_margin": self.quality_margin,
                "quality_scores": quality_scores.tolist(),
                "quality_gate_scores": quality_gate_scores.tolist(),
                "quality_active_mask": final_mask.tolist(),
            },

            "sensor_quality": {
                "lidar": float(quality_scores[0]),
                "ultrasonic": float(quality_scores[1]),
                "radar": float(quality_scores[2]),
            },

            "pre_quality_weights": {
                "lidar": float(pre_quality_weights[0]),
                "ultrasonic": float(pre_quality_weights[1]),
                "radar": float(pre_quality_weights[2]),
            },

            "post_quality_weights": {
                "lidar": float(post_quality_weights[0]),
                "ultrasonic": float(post_quality_weights[1]),
                "radar": float(post_quality_weights[2]),
            },

            "final_weights": {
                "lidar": float(final_weights[0]),
                "ultrasonic": float(final_weights[1]),
                "radar": float(final_weights[2]),
            },
        }

        return {
            "posterior": posterior,
            "depth_map": float(depth_map),
            "depth_mean": float(depth_mean),
            "variance": float(variance),
            "entropy": float(entropy),
            "confidence": float(diagnostics["confidence"]),

            "weights": {
                "lidar": float(w_lidar),
                "ultrasonic": float(w_ultra),
                "radar": float(w_radar),
            },

            "sigmas": {
                "lidar": float(sigmas[0]),
                "ultrasonic": float(sigmas[1]),
                "radar": float(sigmas[2]),
            }, "diagnostics": diagnostics,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "Advanced Clean Adaptive Reliability-Aware Bayesian Fusion",
            "fusion": "Reliability-Weighted Log-Linear Opinion Pool",
            "weight_formula": "w_i ∝ priors_i × R_i^γ × soft context gate × health gate, then uniform blending",
            "sigma_formula": "σ_i = scale / (√R_i + offset), then gentle context expansion",

            "weighting_strategy": {
                "gamma": self.gamma,
                "weight_blend": self.weight_blend,
                "context_blend_boost": self.context_blend_boost,
                "weight_floor": self.weight_floor,
                "adaptation_threshold": self.adaptation_threshold,
                "transition_width": self.transition_width,
            },

            "quality_gate": {
                "enabled": True,
                "dynamic_threshold": True,
                "base_threshold": self.quality_threshold,
                "quality_margin": self.quality_margin,
                "health_floor": self.health_floor,
                "quality_components": [
                    "Reliability",
                    "Measurement Consistency",
                    "Sigma Quality",
                    "Sensor Health",
                ],
            },

            "uncertainty_model": {
                "sigma_scale": self.sigma_scale,
                "sigma_offset": self.sigma_offset,
                "sigma_bounds": (self.sigma_min, self.sigma_max),
                "context_sigma_boost": self.context_sigma_boost,
            },

            "posterior_outputs": [
                "MAP Estimate",
                "Expected Depth",
                "Variance",
                "Entropy",
                "Confidence",
                "Adaptive Weights",
                "Adaptive Sigmas",
                "95% Credible Interval" ],

            "context_inputs": [
                "Scene Complexity",
                "Sensor Agreement",
                "Reliability Spread",
                "Effective Sensor Count",
                "Dominance Ratio",
                "Measurement Spread",
                "Water Depth",
                "NTU",
                "Active Sensor Mask",
                "Sensor Health",
                "Sensor Quality Gate" ],
                
            "calibration": "Merged stable architecture, context-aware enhancements, and dropout-aware sensor health gating",
        }