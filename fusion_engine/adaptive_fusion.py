import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion_engine.bayesian_fusion import BayesianFusionCore

class AdaptiveFusionEngine:

    def __init__(
        self, core: BayesianFusionCore,
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

        if self.sigma_min <= 0.0:
            raise ValueError("sigma_min must be positive")
        if self.sigma_max < self.sigma_min:
            raise ValueError("sigma_max must be >= sigma_min")
        if self.weight_floor < 0.0:
            raise ValueError("weight_floor must be non-negative")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.health_floor < 0.0 or self.health_floor > 1.0:
            raise ValueError("health_floor must be in [0, 1]")
        if self.weight_blend < 0.0 or self.weight_blend > 1.0:
            raise ValueError("weight_blend must be in [0, 1]")
        if self.context_blend_boost < 0.0:
            raise ValueError("context_blend_boost must be non-negative")
        if self.context_sigma_boost < 0.0:
            raise ValueError("context_sigma_boost must be non-negative")
        if self.adaptation_threshold < 0.0 or self.adaptation_threshold > 1.0:
            raise ValueError("adaptation_threshold must be in [0, 1]")
        if self.transition_width <= 0.0:
            raise ValueError("transition_width must be positive")
        if self.quality_threshold < 0.0 or self.quality_threshold > 1.0:
            raise ValueError("quality_threshold must be in [0, 1]")
        if self.quality_margin <= 0.0:
            raise ValueError("quality_margin must be positive")

    # ------------------------------------------------------------------
    # Input Sanitization / Masks
    # ------------------------------------------------------------------
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

    def _measurement_valid_mask(self, measurements: Sequence[Any]) -> np.ndarray:
        if len(measurements) != 3:
            raise ValueError("measurements must contain exactly 3 sensor values")
        valid = []
        for value in measurements:
            if value is None:
                valid.append(False)
                continue
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                valid.append(False)
                continue
            valid.append(bool(np.isfinite(value_float)))
        return np.asarray(valid, dtype=bool)

    def _resolve_active_mask(
        self, measurements: Sequence[Any], active_mask: Optional[Sequence[bool]]) -> Tuple[np.ndarray, np.ndarray]:
        measurement_valid_mask = self._measurement_valid_mask(measurements)
        if active_mask is None:
            requested_active = measurement_valid_mask.copy()
        else:
            requested_active = np.asarray(active_mask, dtype=bool)
            if requested_active.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")
        resolved_active = requested_active & measurement_valid_mask
        return measurement_valid_mask, resolved_active

    def _resolve_health(
        self, sensor_health: Optional[Dict[str, float]], active_mask: np.ndarray) -> np.ndarray:
        source = sensor_health if sensor_health is not None else self.sensor_health
        health = np.asarray([source.get("lidar", 1.0), source.get("ultrasonic", 1.0), source.get("radar", 1.0)], dtype=float)
        health = np.nan_to_num(health, nan=1.0, posinf=1.0, neginf=1.0)
        health = np.clip(health, 0.0, 1.0)
        health = np.where(active_mask, health, 0.0)
        return health

    # ------------------------------------------------------------------
    # Priors / Health / Weight Utilities
    # ------------------------------------------------------------------
    def _normalize_priors(self) -> np.ndarray:
        priors = np.asarray([
            self.sensor_prior.get("lidar", 1.0),
            self.sensor_prior.get("ultrasonic", 1.0),
            self.sensor_prior.get("radar", 1.0),
        ], dtype=float)

        priors = np.nan_to_num(priors, nan=1.0, posinf=1.0, neginf=1.0)
        priors = np.clip(priors, 1e-6, None)
        mean_prior = float(np.mean(priors))

        if not np.isfinite(mean_prior) or mean_prior <= 1e-12:
            return np.ones(3, dtype=float)
        return priors / mean_prior

    def _normalize_health(
        self, active_mask: Optional[Sequence[bool]] = None) -> np.ndarray:
        active = (np.ones(3, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool))
        if active.shape != (3,):
            raise ValueError("active_mask must have shape (3,)")
        return self._resolve_health(None, active)

    def _apply_weight_floor(
        self, weights: Sequence[float], active_mask: Optional[Sequence[bool]] = None) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if weights.shape != (3,):
            raise ValueError("weights must have shape (3,)")
        if active_mask is None:
            active = np.ones(3, dtype=bool)
        else:
            active = np.asarray(active_mask, dtype=bool)
            if active.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")

        weights = np.where(active, np.maximum(weights, 0.0), 0.0)
        active_count = int(np.sum(active))
        if active_count <= 0:
            return np.zeros(3, dtype=float)
        if float(np.sum(weights)) <= 1e-12:
            weights[active] = 1.0 / active_count
            return weights

        weights = np.where(active, np.maximum(weights, self.weight_floor), 0.0)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 1e-12:
            weights[:] = 0.0
            weights[active] = 1.0 / active_count
            return weights
        return weights / total

    def _apply_active_mask(
        self, weights: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).copy()
        active_mask = np.asarray(active_mask, dtype=bool)
        if weights.shape != (3,):
            raise ValueError("weights must have shape (3,)")
        if active_mask.shape != (3,):
            raise ValueError("active_mask must have shape (3,)")

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        active_count = int(np.sum(active_mask))
        if active_count <= 0:
            return np.zeros(3, dtype=float)

        weights = np.where(active_mask, np.maximum(weights, 0.0), 0.0)
        active_weights = np.maximum(weights[active_mask], self.weight_floor)
        total = float(np.sum(active_weights))
        if not np.isfinite(total) or total <= 1e-12:
            active_weights = np.full(active_count, 1.0 / active_count, dtype=float)
        else:
            active_weights = active_weights / total

        weights[:] = 0.0
        weights[active_mask] = active_weights
        return weights

    def _renormalize_active_weights(
        self, weights: Sequence[float], active_mask: Sequence[bool]) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).copy()
        active_mask = np.asarray(active_mask, dtype=bool)
        if weights.shape != (3,):
            raise ValueError("weights must have shape (3,)")
        if active_mask.shape != (3,):
            raise ValueError("active_mask must have shape (3,)")

        weights = np.where(active_mask, np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        active_count = int(np.sum(active_mask))
        if active_count <= 0:
            return np.zeros(3, dtype=float)

        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 1e-12:
            weights[:] = 0.0
            weights[active_mask] = 1.0 / active_count
            return weights
        weights /= total
        return weights

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _effective_sensor_count(self, weights: Sequence[float]) -> float:
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(weights))
        if total <= 1e-12: return 0.0
        weights = weights / total
        denom = float(np.sum(np.square(weights)))
        if denom <= 1e-12: return 0.0
        return float(1.0 / denom)

    def _fusion_entropy(self, weights: Sequence[float]) -> float:
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(weights))
        if total <= 1e-12: return 0.0
        weights = weights / total
        positive = weights > 0.0
        if not np.any(positive): return 0.0
        return float(-np.sum(weights[positive] * np.log(weights[positive])))

    def _dominance_ratio(self, weights: Sequence[float]) -> float:
        weights = np.asarray(weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(weights))
        if total <= 1e-12: return 0.0
        weights = weights / total
        dominant_idx = int(np.argmax(weights))
        dominant_val = float(np.max(weights))
        others = np.delete(weights, dominant_idx)
        if len(others) == 0: return 1.0
        other_mean = float(np.mean(others))
        if other_mean <= 1e-12: return float("inf")
        return float(dominant_val / (other_mean + 1e-12))

    # ------------------------------------------------------------------
    # Context Difficulty
    # ------------------------------------------------------------------
    def _context_difficulty(
        self, scene_complexity: float = 0.0,
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

        water = self._clip01(
            0.0 if water_depth is None else self._sanitize_scalar(water_depth, 0.0) / 20.0, 0.0)
        turb = self._clip01(
            0.0 if ntu is None else self._sanitize_scalar(ntu, 0.0) / 500.0, 0.0)

        if effective_sensor_count is None:
            eff_score = 1.0
        else:
            eff = self._sanitize_scalar(effective_sensor_count, 3.0)
            eff_score = float(np.clip(eff / 3.0, 0.0, 1.0))

        if dominance_ratio is None:
            dom_score = 0.0
        else:
            try:
                dom = float(dominance_ratio)
            except (TypeError, ValueError):
                dom = 1.0
            if np.isnan(dom):
                dom_score = 0.0
            elif np.isinf(dom):
                dom_score = 1.0
            else:
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
        x = float(np.clip(x, -60.0, 60.0))
        gate = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(gate, 0.0, 1.0))

    def _observable_sensor_agreement(
        self, measurements: Sequence[Any], active_mask: Sequence[bool]) -> float:
        values = np.asarray(measurements, dtype=float)
        active = np.asarray(active_mask, dtype=bool)
        valid = active & np.isfinite(values)
        values = values[valid]
        if values.size < 2: return 1.0 if values.size == 1 else 0.0
        pairwise = []
        for i in range(values.size):
            for j in range(i + 1, values.size):
                pairwise.append(abs(float(values[i]) - float(values[j])))
        pairwise_mean = float(np.mean(pairwise)) if pairwise else 0.0
        mean_abs = float(np.mean(np.abs(values))) + 1e-6
        agreement = 1.0 / (1.0 + pairwise_mean / mean_abs)
        return float(np.clip(agreement, 0.0, 1.0))

    def _observable_measurement_spread(
        self, measurements: Sequence[Any], active_mask: Sequence[bool]) -> float:
        values = np.asarray(measurements, dtype=float)
        active = np.asarray(active_mask, dtype=bool)
        values = values[active & np.isfinite(values)]
        if values.size < 2: return 0.0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return float(1.4826 * mad)

    # ------------------------------------------------------------------
    # Measurement Consistency
    # ------------------------------------------------------------------
    def _measurement_consistency(
        self, z_lidar: Optional[float], z_ultra: Optional[float], z_radar: Optional[float],
        active_mask: Optional[Sequence[bool]] = None) -> np.ndarray:
        measurements = np.asarray([z_lidar, z_ultra, z_radar], dtype=float)
        valid_mask = np.isfinite(measurements)
        if active_mask is None:
            participation_mask = valid_mask.copy()
        else:
            active = np.asarray(active_mask, dtype=bool)
            if active.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")
            participation_mask = active & valid_mask

        result = np.ones(3, dtype=float)
        valid_values = measurements[participation_mask]
        valid_count = int(valid_values.size)
        if valid_count < 2: return result

        if valid_count == 2:
            diff = abs(float(valid_values[0]) - float(valid_values[1]))
            score = float(np.exp(-0.5 * (diff / 3.5) ** 2))
            score = float(np.clip(score, 0.45, 1.0))
            result[participation_mask] = score
            return result

        consensus = float(np.median(valid_values))
        deviations = np.abs(valid_values - consensus)
        mad = float(np.median(deviations))
        scale = max(1.4826 * mad, 0.35)
        consistency = np.exp(-np.square(deviations / scale))
        result[participation_mask] = np.clip(consistency, 0.05, 1.0)
        return result

    # ------------------------------------------------------------------
    # Sensor Quality / Gating
    # ------------------------------------------------------------------
    def _compute_sensor_quality(
        self, reliability: float, sigma: float, consistency: float, health: float = 1.0, active: bool = True) -> float:
        if not active: return 0.0
        reliability = self._clip01(reliability, 0.0)
        consistency = self._clip01(consistency, 0.5)
        health = self._clip01(health, 1.0)
        sigma_range = max(self.sigma_max - self.sigma_min, 1e-12)
        sigma_score = 1.0 - np.clip((sigma - self.sigma_min) / sigma_range, 0.0, 1.0)
        quality = (0.42 * reliability + 0.22 * consistency + 0.16 * sigma_score + 0.20 * health)
        return float(np.clip(quality, 0.0, 1.0))

    def _adaptive_quality_threshold(
        self, difficulty: float, agreement: float, consistency: float,
        reliability_spread: float) -> float:
        threshold = (self.quality_threshold
            + 0.12 * difficulty
            + 0.08 * (1.0 - agreement)
            + 0.08 * (1.0 - consistency)
            + 0.05 * reliability_spread)

        return float(np.clip(threshold, 0.18, 0.45))

    # ------------------------------------------------------------------
    # Adaptive Weighting
    # ------------------------------------------------------------------
    def _normalize_weights(
        self, r_lidar: float, r_ultra: float, r_radar: float,
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
        sensor_health: Optional[Sequence[float]] = None) -> Tuple[float, float, float]:

        reliabilities = np.asarray([r_lidar, r_ultra, r_radar], dtype=float)
        reliabilities = np.clip(
            np.nan_to_num(reliabilities, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if active_mask is None:
            active_mask_arr = np.array([True, True, True], dtype=bool)
        else:
            active_mask_arr = np.asarray(active_mask, dtype=bool)
            if active_mask_arr.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")

        health_arr = np.asarray(
            sensor_health if sensor_health is not None else [1.0, 1.0, 1.0], dtype=float)
        if health_arr.shape != (3,):
            raise ValueError("sensor_health must have shape (3,)")

        health_arr = np.nan_to_num(health_arr, nan=1.0, posinf=1.0, neginf=1.0)
        health_arr = np.clip(health_arr, 0.0, 1.0)
        health_arr = np.where(active_mask_arr, health_arr, 0.0)

        reliabilities = np.where(active_mask_arr, reliabilities, 0.0)
        priors = self._normalize_priors()

        trust = np.asarray([
            self._sensor_trust_strength(
                "lidar", reliabilities[0],
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth,
                ntu=ntu, health=health_arr[0]),

            self._sensor_trust_strength(
                "ultrasonic", reliabilities[1],
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth,
                ntu=ntu, health=health_arr[1]),

            self._sensor_trust_strength(
                "radar", reliabilities[2],
                scene_complexity=scene_complexity,
                sensor_agreement=sensor_agreement,
                reliability_spread=reliability_spread,
                effective_sensor_count=effective_sensor_count,
                dominance_ratio=dominance_ratio,
                confidence=confidence,
                measurement_spread=measurement_spread,
                water_depth=water_depth,
                ntu=ntu, health=health_arr[2]),
            ], dtype=float)

        trust *= priors
        trust *= np.where(active_mask_arr, 1.0, 0.0)
        trust *= self.health_floor + (1.0 - self.health_floor) * health_arr

        measurement_consistency = self._measurement_consistency(
            z_lidar, z_ultra, z_radar, active_mask=active_mask_arr)

        participating_count = int(np.sum(active_mask_arr))
        if participating_count >= 2:
            active_consistency = np.clip(
                measurement_consistency[active_mask_arr], 0.05, 1.0)
            trust[active_mask_arr] *= np.power(active_consistency, 0.65)

        trust = np.where(active_mask_arr, np.maximum(trust, self.weight_floor), 0.0)
        total = float(np.sum(trust))

        if not np.isfinite(total) or total <= 1e-12:
            if participating_count > 0:
                raw_weights = np.where(active_mask_arr, 1.0 / participating_count, 0.0)
            else:
                raw_weights = np.zeros(3, dtype=float)
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

        if participating_count > 0:
            uniform = np.where(active_mask_arr, 1.0 / participating_count, 0.0)
        else:
            uniform = np.zeros(3, dtype=float)

        weights = retain_raw * raw_weights + (1.0 - retain_raw) * uniform
        weights = self._apply_active_mask(weights, active_mask_arr)
        return tuple(float(w) for w in weights)

    # ------------------------------------------------------------------
    # Adaptive Uncertainty
    # ------------------------------------------------------------------
    def _infer_sigmas(
        self, r_lidar: float, r_ultra: float, r_radar: float,
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
        provided_sigmas: Optional[Sequence[float]] = None) -> Tuple[float, float, float]:

        reliabilities = np.asarray([r_lidar, r_ultra, r_radar], dtype=float)
        reliabilities = np.clip(
            np.nan_to_num(reliabilities, nan=1e-3, posinf=1.0, neginf=1e-3), 1e-3, 1.0)

        if active_mask is None:
            active_mask_arr = np.array([True, True, True], dtype=bool)
        else:
            active_mask_arr = np.asarray(active_mask, dtype=bool)
            if active_mask_arr.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")

        if provided_sigmas is not None:
            base_sigmas = np.asarray(provided_sigmas, dtype=float)
            if base_sigmas.shape != (3,):
                raise ValueError("provided_sigmas must have shape (3,)")

            base_sigmas = np.nan_to_num(
                base_sigmas, nan=self.sigma_scale, posinf=self.sigma_max, neginf=self.sigma_min)
            base_sigmas = np.clip(base_sigmas, self.sigma_min, self.sigma_max)
        else:
            base_sigmas = self.sigma_scale / (np.sqrt(reliabilities) + self.sigma_offset)

        if sensor_health is None:
            health_arr = np.ones(3, dtype=float)
        else:
            health_arr = np.asarray(sensor_health, dtype=float)
            if health_arr.shape != (3,):
                raise ValueError("sensor_health must have shape (3,)")

            health_arr = np.nan_to_num(health_arr, nan=1.0, posinf=1.0, neginf=1.0)
            health_arr = np.clip(health_arr, 0.0, 1.0)

        health_arr = np.where(active_mask_arr, health_arr, 0.0)

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

        meas_norm = (0.0 if measurement_spread is None
            else float(np.clip(self._sanitize_scalar(measurement_spread, 0.0) / 5.0, 0.0, 1.0)))

        water = self._clip01(
            0.0 if water_depth is None else self._sanitize_scalar(water_depth, 0.0) / 20.0, 0.0)
        turb = self._clip01(
            0.0 if ntu is None else self._sanitize_scalar(ntu, 0.0) / 500.0, 0.0)

        gate = self._adaptation_gate(difficulty)
        context_multiplier = 1.0 + gate * (self.context_sigma_boost * difficulty + 0.05 * spread)
        sigmas = np.asarray(base_sigmas, dtype=float) * context_multiplier

        if measurement_consistency is None:
            measurement_consistency = self._measurement_consistency(
                z_lidar, z_ultra, z_radar, active_mask=active_mask_arr)
        else:
            measurement_consistency = np.asarray(measurement_consistency, dtype=float)
            if measurement_consistency.shape != (3,):
                raise ValueError("measurement_consistency must have shape (3,)")

            measurement_consistency = np.nan_to_num(
                measurement_consistency, nan=1.0, posinf=1.0, neginf=0.0)
            measurement_consistency = np.clip(measurement_consistency, 0.0, 1.0)

        sigma_penalty = 1.0 + 0.22 * (1.0 - measurement_consistency)
        sigmas *= sigma_penalty
        sigmas = np.where(active_mask_arr, sigmas, self.sigma_max)

        sigmas[0] *= 1.0 + gate * (0.08 * difficulty
            + 0.08 * meas_norm
            + 0.06 * turb
            + 0.05 * water
            + 0.03 * (1.0 - agreement))

        sigmas[1] *= 1.0 + gate * (0.08 * difficulty
            + 0.10 * meas_norm
            + 0.10 * water
            + 0.03 * (1.0 - agreement))

        sigmas[2] *= 1.0 + gate * (0.07 * difficulty
            + 0.12 * meas_norm
            + 0.06 * water
            + 0.04 * turb
            + 0.03 * (1.0 - agreement))

        sigmas = np.clip(sigmas, self.sigma_min, self.sigma_max)
        sigmas = np.where(active_mask_arr, sigmas, self.sigma_max)
        return tuple(float(s) for s in sigmas)

    # ------------------------------------------------------------------
    # Sensor-Specific Trust
    # ------------------------------------------------------------------
    def _sensor_trust_strength(
        self, sensor_name: str, reliability: float,
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

        water = self._clip01(0.0 if water_depth is None
        else self._sanitize_scalar(water_depth, 0.0) / 20.0, 0.0)
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
            try:
                dom = float(dominance_ratio)
            except (TypeError, ValueError):
                dom = 1.0
            if np.isnan(dom):
                dom_penalty = 0.0
            elif np.isinf(dom):
                dom_penalty = 0.45
            else:
                dom_penalty = float(np.clip((dom - 1.0) / 3.0, 0.0, 0.45))

        meas = (0.0 if measurement_spread is None
            else float(np.clip(self._sanitize_scalar(measurement_spread, 0.0) / 5.0, 0.0, 1.0)))
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

        else: context = 1.0
        context = float(np.clip(context, 0.55, 1.55))
        trust = (0.92 + 0.08 * eff_bonus) * sharpened * context * (0.98 + 0.02 * conf)
        return float(np.clip(trust, 1e-6, 10.0))

    # ------------------------------------------------------------------
    # Posterior Confidence Diagnostic
    # ------------------------------------------------------------------
    def _fusion_confidence(
        self, posterior, weights, scene_complexity: float = 0.0) -> float:
        posterior_mass = self.core.posterior_mass(posterior)
        entropy = self.core.entropy(posterior)
        max_entropy = float(np.log(max(len(posterior_mass), 1)))
        entropy_score = (0.0 if max_entropy <= 1e-12
            else float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0)))

        variance = self.core.posterior_variance(posterior)
        span = max(self.core.depth_max - self.core.depth_min, 1e-6)
        variance_scale = max(0.12 * span * span, 1e-6)

        variance_score = float(np.clip(np.exp(-variance / variance_scale), 0.0, 1.0))
        peak_score = float(np.clip(np.max(posterior_mass) / 0.05, 0.0, 1.0))

        ci_lower, ci_upper = self.core.credible_interval(posterior, mass_level=0.95)
        interval_score = float(np.clip(1.0 - ((ci_upper - ci_lower) / span), 0.0, 1.0))

        balance_score = self._effective_sensor_count(weights) / 3.0
        scene_score = 1.0 - self._clip01(scene_complexity, 0.0)

        confidence = (0.30 * entropy_score
            + 0.18 * variance_score
            + 0.20 * peak_score
            + 0.12 * interval_score
            + 0.20 * balance_score * scene_score)

        return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Main Fusion
    # ------------------------------------------------------------------
    def fuse(
        self, z_lidar, z_ultra, z_radar,
        r_lidar, r_ultra, r_radar, sigmas=None,
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
        sensor_health: Optional[Dict[str, float]] = None) -> Dict[str, Any]:

        measurements = (z_lidar, z_ultra, z_radar)
        measurement_valid_mask, active_mask_arr = self._resolve_active_mask(
            measurements, active_mask)
        active_count = int(np.sum(active_mask_arr))

        if active_count == 0:
            sigmas_arr = np.full(3, self.sigma_max, dtype=float)

            return {
                "status": "NO_VALID_SENSOR",
                "posterior": None,
                "depth_map": None,
                "depth_mean": None,
                "variance": None,
                "entropy": None,
                "confidence": 0.0,
                "weights": {"lidar": 0.0, "ultrasonic": 0.0, "radar": 0.0},

                "sigmas": {
                    "lidar": float(sigmas_arr[0]),
                    "ultrasonic": float(sigmas_arr[1]),
                    "radar": float(sigmas_arr[2]),
                },

                "diagnostics": {
                    "measurement_valid_mask": measurement_valid_mask.tolist(),
                    "active_mask": active_mask_arr.tolist(),
                    "active_sensor_count": 0,
                    "final_mask": [False, False, False],
                    "measurement_consistency_mean": None,
                    "context_difficulty": None,
                    "adaptation_gate": None,
                    "sensor_health": {"lidar": 0.0, "ultrasonic": 0.0, "radar": 0.0},

                    "quality_gate": {
                        "quality_threshold": None,
                        "quality_margin": self.quality_margin,
                        "quality_scores": [0.0, 0.0, 0.0],
                        "quality_gate_scores": [0.0, 0.0, 0.0],
                        "quality_active_mask": [False, False, False],
                    },

                    "pre_quality_weights": {"lidar": 0.0, "ultrasonic": 0.0, "radar": 0.0},
                    "post_quality_weights": {"lidar": 0.0, "ultrasonic": 0.0, "radar": 0.0},
                    "final_weights": {"lidar": 0.0, "ultrasonic": 0.0, "radar": 0.0},
                },
            }

        if effective_sensor_count is None:
            context_effective_sensor_count = float(active_count)
        else:
            context_effective_sensor_count = self._sanitize_scalar(
                effective_sensor_count, float(active_count))
            context_effective_sensor_count = float(
                np.clip(context_effective_sensor_count, 1.0, float(active_count)))

        # Derive agreement/spread from active observations when they are not explicitly supplied
        if sensor_agreement is None:
            sensor_agreement = self._observable_sensor_agreement(
                measurements, active_mask_arr)
        if measurement_spread is None:
            measurement_spread = self._observable_measurement_spread(
                measurements, active_mask_arr)

        difficulty = self._context_difficulty(
            scene_complexity=scene_complexity,
            sensor_agreement=sensor_agreement,
            reliability_spread=reliability_spread,
            effective_sensor_count=context_effective_sensor_count,
            dominance_ratio=dominance_ratio,
            confidence=confidence,
            measurement_spread=measurement_spread,
            water_depth=water_depth, ntu=ntu)

        measurement_consistency = self._measurement_consistency(
            z_lidar, z_ultra, z_radar, active_mask=active_mask_arr)
        sensor_health_arr = self._resolve_health(sensor_health, active_mask_arr)

        # --------------------------------------------------------------
        # Adaptive Weights
        # --------------------------------------------------------------
        w_lidar, w_ultra, w_radar = self._normalize_weights(
            r_lidar, r_ultra, r_radar,
            z_lidar=z_lidar, z_ultra=z_ultra, z_radar=z_radar,
            scene_complexity=scene_complexity,
            sensor_agreement=sensor_agreement,
            reliability_spread=reliability_spread,
            effective_sensor_count=context_effective_sensor_count,
            dominance_ratio=dominance_ratio,
            confidence=confidence,
            measurement_spread=measurement_spread,
            water_depth=water_depth,
            ntu=ntu, difficulty=difficulty,
            active_mask=active_mask_arr,
            sensor_health=sensor_health_arr)

        # --------------------------------------------------------------
        # Adaptive Sigmas
        # --------------------------------------------------------------
        base_sigmas = None if sigmas is None else sigmas
        sigmas = self._infer_sigmas(
            r_lidar, r_ultra, r_radar,
            z_lidar=z_lidar, z_ultra=z_ultra, z_radar=z_radar,
            scene_complexity=scene_complexity,
            sensor_agreement=sensor_agreement,
            reliability_spread=reliability_spread,
            effective_sensor_count=context_effective_sensor_count,
            dominance_ratio=dominance_ratio,
            confidence=confidence,
            measurement_spread=measurement_spread,
            water_depth=water_depth,
            ntu=ntu, difficulty=difficulty,
            measurement_consistency=measurement_consistency,
            active_mask=active_mask_arr,
            sensor_health=sensor_health_arr,
            provided_sigmas=base_sigmas)

        raw_weights = np.asarray([w_lidar, w_ultra, w_radar], dtype=float)
        consistency_values = measurement_consistency

        # --------------------------------------------------------------
        # Quality Score
        # --------------------------------------------------------------
        quality_scores = np.asarray([
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
                active=active_mask_arr[2]),
        ], dtype=float)

        agreement = (0.5 if sensor_agreement is None else self._clip01(sensor_agreement, 0.5))
        active_consistency_values = consistency_values[active_mask_arr]
        consistency_mean = (float(np.mean(active_consistency_values))
            if active_consistency_values.size > 0 else 1.0)
        spread = (0.0 if reliability_spread is None else self._clip01(reliability_spread, 0.0))

        adaptive_threshold = self._adaptive_quality_threshold(
            difficulty=difficulty,
            agreement=agreement,
            consistency=consistency_mean,
            reliability_spread=spread)

        weights = raw_weights.copy()
        quality_gate_scores = quality_scores.copy()

        for i in range(3):
            if not active_mask_arr[i]:
                weights[i] = 0.0
                continue

            q = float(quality_gate_scores[i])
            if q < adaptive_threshold:
                if active_count >= 3:
                    weights[i] *= self.weight_floor
                else:
                    severe_quality_floor = 0.12
                    if q < severe_quality_floor:
                        weights[i] *= self.weight_floor
                    else:
                        weights[i] *= 0.35

            elif q < (adaptive_threshold + self.quality_margin):
                scale = (q - adaptive_threshold) / max(self.quality_margin, 1e-6)
                scale = float(np.clip(scale, 0.0, 1.0))
                if active_count >= 3:
                    weights[i] *= self.weight_floor + 0.95 * scale**2
                else:
                    weights[i] *= 0.35 + 0.65 * scale**2

        pre_quality_weights = raw_weights.copy()
        post_quality_weights = weights.copy()

        # --------------------------------------------------------------
        # Final Participating Sensor Mask
        # --------------------------------------------------------------
        active_indices = np.flatnonzero(active_mask_arr)

        if active_count == 1:
            final_mask = active_mask_arr.copy()
        elif active_count == 2:
            final_mask = active_mask_arr.copy()
        else:
            final_mask = (weights >= self.weight_floor) & active_mask_arr
            if not np.any(final_mask):
                best_idx = int(active_indices[np.argmax(quality_gate_scores[active_indices])])
                final_mask = np.zeros(3, dtype=bool)
                final_mask[best_idx] = True

        if not np.any(final_mask):
            raise RuntimeError(
                "Fusion produced no participating sensor despite having active valid measurements")

        participating_weights = np.where(final_mask, np.maximum(weights, 0.0), 0.0)
        participating_total = float(np.sum(participating_weights))

        if not np.isfinite(participating_total) or participating_total <= 1e-12:
            participating_count = int(np.sum(final_mask))
            participating_weights[:] = 0.0
            participating_weights[final_mask] = 1.0 / participating_count
        else:
            participating_weights /= participating_total

        weights = participating_weights
        w_lidar, w_ultra, w_radar = weights
        final_weights = weights.copy()

        # --------------------------------------------------------------
        # Likelihood Construction
        # --------------------------------------------------------------
        likelihoods = []
        likelihood_weights = []
        sensor_measurements = (z_lidar, z_ultra, z_radar)

        for measurement, sigma, weight, participate in zip(
            sensor_measurements, sigmas, weights, final_mask):
            if not participate or measurement is None: continue
            try:
                measurement_value = float(measurement)
            except (TypeError, ValueError): continue
            if not np.isfinite(measurement_value): continue
            likelihoods.append(
                self.core.compute_likelihood(measurement_value, float(sigma)))
            likelihood_weights.append(float(weight))

        if not likelihoods:
            raise RuntimeError("No valid likelihoods remained after final sensor selection")
        if len(likelihoods) != len(likelihood_weights):
            raise RuntimeError("Internal fusion error: likelihood/weight count mismatch")

        # --------------------------------------------------------------
        # Bayesian Posterior Fusion
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------
        diagnostics = {
            "effective_sensor_count": float(self._effective_sensor_count(final_weights)),
            "dominance_ratio": float(self._dominance_ratio(final_weights)),
            "weight_entropy": float(self._fusion_entropy(final_weights)),
            "confidence": float(self._fusion_confidence(posterior, final_weights, scene_complexity)),
            "posterior_peak": posterior_peak,
            "map_index": map_idx,
            "credible_interval_95": (float(ci_lower), float(ci_upper)),
            "scene_complexity": float(self._clip01(scene_complexity, 0.0)),

            "sensor_agreement": (None if sensor_agreement is None
                else float(self._clip01(sensor_agreement, 0.5))),
            "reliability_spread": (None if reliability_spread is None
                else float(self._clip01(reliability_spread, 0.0))),
            "measurement_spread": (None if measurement_spread is None
                else float(self._sanitize_scalar(measurement_spread, 0.0))),

            "context_difficulty": float(difficulty),
            "adaptation_gate": float(self._adaptation_gate(difficulty)),
            "measurement_consistency_mean": float(consistency_mean),
            "measurement_valid_mask": measurement_valid_mask.tolist(),
            "active_mask": active_mask_arr.tolist(),
            "active_sensor_count": int(np.sum(final_mask)),

            "sensor_health": {
                "lidar": float(sensor_health_arr[0]),
                "ultrasonic": float(sensor_health_arr[1]),
                "radar": float(sensor_health_arr[2]),
            },

            "quality_gate": {
                "quality_threshold": float(adaptive_threshold),
                "quality_margin": float(self.quality_margin),
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
            "status": "OK",
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

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "Reliability-Weighted Generalized Bayesian Fusion",
            "fusion": "Tempered logarithmic opinion pool with observation-adaptive sensor weights",

            "weight_formula": (
                "w_i ∝ prior_i × R_i^gamma × context_factor × health_factor × "
                "consistency_factor, followed by active-sensor normalization "
                "and bounded uniform blending"),

            "sigma_formula": (
                "sigma_i is derived from reliability-based uncertainty or a "
                "supplied fixed/calibration base sigma, then always adjusted "
                "by context difficulty, active-sensor consistency, and "
                "sensor-specific environmental factors within bounded limits"),

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
                "supports_provided_sigmas_as_base_calibration": True,
                "provided_sigmas_are_final": False,
            },

            "measurement_handling": {
                "validity_mask": True,
                "active_mask_intersection": True,
                "active_valid_consistency_only": True,
                "two_sensor_consistency": True,
                "single_sensor_support": True,
                "zero_sensor_behavior": "Return status NO_VALID_SENSOR",
                "likelihood_weight_alignment": True,
            },

            "posterior_outputs": [
                "MAP Estimate",
                "Expected Depth",
                "Variance",
                "Entropy",
                "Confidence",
                "Adaptive Weights",
                "Adaptive Sigmas",
                "95% Credible Interval",
            ],

            "context_inputs": [
                "Scene Complexity",
                "Sensor Agreement",
                "Reliability Spread",
                "Effective Sensor Count",
                "Dominance Ratio",
                "Measurement Spread",
                "Estimated Water Depth",
                "Estimated NTU",
                "Active Sensor Mask",
                "Measurement Validity Mask",
                "Sensor Health",
                "Sensor Quality Gate",
            ],

            "scientific_interpretation": {
                "posterior_validity": (
                    "The posterior is obtained using a generalized/tempered Bayesian rule, explicitly accounting for data-dependent adaptive weights, rather than conditionally independent standard Bayes."),
                "reliability":
                    "Contextual fusion trust score supplied by the phenomenological Reliability Engine.",
                "consistency":
                    "Robust inter-sensor agreement score computed only over active and valid measurements.",
                "health":
                    "Operational sensor availability / trust factor, bounded to [0, 1].",
                "sigma":
                    "Fusion likelihood scale representing modeled measurement uncertainty; not a manufacturer-calibrated sensor error specification.",
                "confidence":
                    "Engineered posterior-concentration diagnostic, not a calibrated probability of correctness.",
            },

            "assumptions": [
                "Reliability values are bounded contextual fusion trust scores.",
                "Sensor priors are engineering-defined relative preferences "
                "rather than empirical probabilities.",
                "The dual application of reliability (modulating both fusion "
                "weight and likelihood variance) provides complementary "
                "information, requiring explicit ablation validation.",
                "Inter-sensor consistency is evaluated only among active "
                "and numerically valid observations.",
                "Sensor health describes operational status/trust and is "
                "bounded to [0, 1].",
                "Adaptive context coefficients and quality thresholds are "
                "engineering-defined parameters.",
            ],

            "calibration": (
                "Refined from the established adaptive architecture by enforcing "
                "active-valid measurement consistency, explicit no-estimate "
                "handling, bounded uncertainty, a single authoritative adaptive "
                "sigma path, missing-data-safe context metrics, single-path health "
                "gating, and aligned likelihood construction without changing "
                "the core fusion strategy."),
        }