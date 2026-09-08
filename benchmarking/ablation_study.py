import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion_engine.bayesian_fusion import BayesianFusionCore
from fusion_engine.fixed_fusion import FixedFusionEngine
from fusion_engine.adaptive_fusion import AdaptiveFusionEngine
from fusion_engine.uncertainty import UncertaintyQuantification
from fusion_engine.hazard_assessment import HazardAssessment

class AblationFramework:

    SENSOR_NAMES = ("lidar", "ultrasonic", "radar")
    DEFAULT_BASE_SIGMAS = (0.85, 0.75, 0.60)

    EXPERIMENT_LABELS = {
        "A_All_Sensors": "Proposed adaptive fusion with the row's valid sensors",
        "B_No_LiDAR": "Adaptive fusion with LiDAR removed",
        "C_No_Radar": "Adaptive fusion with radar removed",
        "D_No_Ultra": "Adaptive fusion with ultrasonic removed",
        "E_No_Reliability": "Equal-weight Bayesian pooling without reliability/health weighting",
        "F_No_Context": "Adaptive fusion with contextual adaptation neutralized",
        "G_No_Adaptive": "Fixed Bayesian fusion baseline using calibrated static sigmas",
    }

    def __init__(
        self, adaptive_engine: AdaptiveFusionEngine,
        fixed_engine: FixedFusionEngine,
        core: BayesianFusionCore,
        uq: Optional[UncertaintyQuantification] = None,
        hazard_assessor: Optional[HazardAssessment] = None) -> None:

        self.adaptive = adaptive_engine
        self.fixed = fixed_engine
        self.core = core
        self.uq = uq or UncertaintyQuantification()
        self.hazard_assessor = HazardAssessment(threshold_cm=10.0, decision_boundary=0.80)

        if not callable(getattr(self.adaptive, "_infer_sigmas", None)):
            raise TypeError(
                "AdaptiveFusionEngine must expose _infer_sigmas() for the no-reliability ablation")

    # ------------------------------------------------------------------
    # Basic validation / sanitization
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(value: Any, default: float = np.nan) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if np.isfinite(out) else float(default)

    @staticmethod
    def _clip01(value: Any, default: float = 0.0) -> float:
        out = AblationFramework._safe_float(value, default)
        return float(np.clip(out, 0.0, 1.0))

    @staticmethod
    def _measurement(value: Any) -> float:
        return AblationFramework._safe_float(value, np.nan)

    @staticmethod
    def _reliability(value: Any) -> float:
        out = AblationFramework._safe_float(value, 0.0)
        return float(np.clip(out, 0.0, 1.0))

    @staticmethod
    def _sanitize_sigmas(values: Sequence[Any], label: str) -> Tuple[float, float, float]:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"{label} must contain exactly three sigma values")
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{label} must contain finite positive values")
        return tuple(float(v) for v in arr)

    @classmethod
    def _resolve_base_sigmas(
        cls, fixed_sigmas: Optional[Tuple[float, float, float]],
        base_sigmas: Optional[Tuple[float, float, float]],
        adaptive_sigmas: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        candidates = [
            ("fixed_sigmas", fixed_sigmas),
            ("base_sigmas", base_sigmas),
            ("adaptive_sigmas", adaptive_sigmas)]

        supplied = [(name, value) for name, value in candidates if value is not None]
        if not supplied: return cls.DEFAULT_BASE_SIGMAS

        resolved_name, resolved_value = supplied[0]
        resolved = cls._sanitize_sigmas(resolved_value, resolved_name)

        for name, value in supplied[1:]:
            other = cls._sanitize_sigmas(value, name)
            if not np.allclose(resolved, other, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "fixed_sigmas, base_sigmas, and adaptive_sigmas must refer "
                    "to the same calibrated base sigma triplet when more than one is supplied")
        return resolved

    # ------------------------------------------------------------------
    # Sensor-state helpers
    # ------------------------------------------------------------------
    def _resolve_active_mask(
        self, measurements: Sequence[float],
        reliabilities: Sequence[float],
        active_mask: Optional[Sequence[bool]]) -> np.ndarray:

        measurements_arr = np.asarray(measurements, dtype=float)
        reliabilities_arr = np.asarray(reliabilities, dtype=float)
        if measurements_arr.shape != (3,) or reliabilities_arr.shape != (3,):
            raise ValueError("measurements and reliabilities must each have shape (3,)")

        measurement_valid = np.isfinite(measurements_arr)
        reliability_valid = reliabilities_arr > 0.0
        resolved = measurement_valid & reliability_valid

        if active_mask is not None:
            requested = np.asarray(active_mask, dtype=bool)
            if requested.shape != (3,):
                raise ValueError("active_mask must have shape (3,)")
            resolved &= requested
        return resolved.astype(bool)

    def _resolve_sensor_health(
        self, sensor_health: Optional[Dict[str, float]],
        active_mask: np.ndarray) -> np.ndarray:
        source = sensor_health or {}
        values = np.asarray([
            self._clip01(source.get("lidar", 1.0), 1.0),
            self._clip01(source.get("ultrasonic", 1.0), 1.0),
            self._clip01(source.get("radar", 1.0), 1.0),
        ], dtype=float)
        return np.where(active_mask, values, 0.0)

    # ------------------------------------------------------------------
    # Observable context helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _observable_sensor_agreement(
        measurements: Sequence[float],
        active_mask: Sequence[bool]) -> float:
        values = np.asarray(measurements, dtype=float)
        active = np.asarray(active_mask, dtype=bool)
        values = values[active & np.isfinite(values)]

        if values.size == 0: return 0.0
        if values.size == 1: return 1.0

        pairwise = []
        for i in range(values.size):
            for j in range(i + 1, values.size):
                pairwise.append(abs(float(values[i] - values[j])))

        pairwise_mean = float(np.mean(pairwise)) if pairwise else 0.0
        mean_abs = float(np.mean(np.abs(values))) + 1e-6
        agreement = 1.0 / (1.0 + pairwise_mean / mean_abs)
        return float(np.clip(agreement, 0.0, 1.0))

    @staticmethod
    def _observable_measurement_spread(
        measurements: Sequence[float],
        active_mask: Sequence[bool]) -> float:
        values = np.asarray(measurements, dtype=float)
        active = np.asarray(active_mask, dtype=bool)
        values = values[active & np.isfinite(values)]

        if values.size < 2: return 0.0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return float(1.4826 * mad)

    @staticmethod
    def _reliability_spread(
        reliabilities: Sequence[float],
        active_mask: Sequence[bool]) -> float:
        r = np.asarray(reliabilities, dtype=float)
        active = np.asarray(active_mask, dtype=bool)
        values = r[active]
        if values.size < 2: return 0.0
        return float(np.clip(np.max(values) - np.min(values), 0.0, 1.0))

    @staticmethod
    def _reliability_dominance_ratio(
        reliabilities: Sequence[float],
        active_mask: Sequence[bool]) -> float:
        r = np.asarray(reliabilities, dtype=float)
        active = np.asarray(active_mask, dtype=bool)
        values = np.clip(r[active], 0.0, 1.0)

        if values.size < 2: return 1.0
        dominant = float(np.max(values))
        others = values[values < dominant + 1e-12]
        if others.size == 0: return 1.0

        other_mean = float(np.mean(others))
        if other_mean <= 1e-12: return float("inf")
        return float(max(1.0, dominant / (other_mean + 1e-12)))

    def _build_observable_context(
        self, measurements: np.ndarray,
        reliabilities: np.ndarray,
        active_mask: np.ndarray,
        scene_complexity: Any,
        water_context_depth_proxy: Any,
        ntu: Any) -> Dict[str, float]:
        active_count = int(np.sum(active_mask))

        return {
            "scene_complexity": self._clip01(scene_complexity, 0.0),
            "sensor_agreement": self._observable_sensor_agreement(measurements, active_mask),
            "reliability_spread": self._reliability_spread(reliabilities, active_mask),
            "effective_sensor_count": float(active_count),
            "dominance_ratio": self._reliability_dominance_ratio(reliabilities, active_mask),
            "confidence": 0.5,
            "measurement_spread": self._observable_measurement_spread(measurements, active_mask),
            "water_depth": max(0.0, self._safe_float(water_context_depth_proxy, 0.0)),
            "ntu": max(0.0, self._safe_float(ntu, 0.0)),
        }

    @staticmethod
    def _neutral_context(active_sensor_count: int) -> Dict[str, float]:
        return {
            "scene_complexity": 0.0,
            "sensor_agreement": 1.0,
            "reliability_spread": 0.0,
            "effective_sensor_count": 3.0,
            "dominance_ratio": 1.0,
            "confidence": 1.0,
            "measurement_spread": 0.0,
            "water_depth": 0.0,
            "ntu": 0.0,
        }

    @staticmethod
    def _reliability_neutral_context(
        observable_context: Dict[str, float],
        active_sensor_count: int) -> Dict[str, float]:
        return {
            "scene_complexity": observable_context["scene_complexity"],
            "sensor_agreement": observable_context["sensor_agreement"],
            "reliability_spread": 0.0,
            "effective_sensor_count": float(active_sensor_count),
            "dominance_ratio": 1.0,
            "confidence": 0.5,
            "measurement_spread": observable_context["measurement_spread"],
            "water_depth": observable_context["water_depth"],
            "ntu": observable_context["ntu"],
        }

    # ------------------------------------------------------------------
    # Sigma helpers
    # ------------------------------------------------------------------
    def _adaptive_base_sigmas(
        self, base_sigmas: Tuple[float, float, float],
        reliabilities: Sequence[float],
        measurements: Sequence[float],
        context: Dict[str, float],
        active_mask: np.ndarray,
        sensor_health: Sequence[float]) -> Tuple[float, float, float]:

        sigmas = self.adaptive._infer_sigmas(
            float(reliabilities[0]), float(reliabilities[1]), float(reliabilities[2]),
            z_lidar=float(measurements[0]) if np.isfinite(measurements[0]) else np.nan,
            z_ultra=float(measurements[1]) if np.isfinite(measurements[1]) else np.nan,
            z_radar=float(measurements[2]) if np.isfinite(measurements[2]) else np.nan,
            provided_sigmas=base_sigmas,
            scene_complexity=float(context["scene_complexity"]),
            sensor_agreement=float(context["sensor_agreement"]),
            reliability_spread=float(context["reliability_spread"]),
            effective_sensor_count=float(context["effective_sensor_count"]),
            dominance_ratio=float(context["dominance_ratio"]),
            confidence=float(context["confidence"]),
            measurement_spread=float(context["measurement_spread"]),
            water_depth=float(context["water_depth"]),
            ntu=float(context["ntu"]),
            active_mask=tuple(bool(v) for v in active_mask),
            sensor_health=tuple(float(v) for v in sensor_health))
        return self._sanitize_sigmas(sigmas, "adaptive sigmas")

    # ------------------------------------------------------------------
    # Posterior/result normalization
    # ------------------------------------------------------------------
    def _normalize_posterior_result(
        self, posterior: Optional[np.ndarray],
        true_depth: Optional[float] = None,
        scene_complexity: float = 0.0,
        status: str = "OK", failure_reason: Optional[str] = None) -> Dict[str, Any]:

        if posterior is None:
            result: Dict[str, Any] = {
                "posterior": None,
                "posterior_mass": None,
                "map_depth": None,
                "expected_depth": None,
                "variance": None,
                "std": None,
                "entropy": None,
                "posterior_peak": None,
                "confidence": 0.0,
                "ci_lower": None,
                "ci_upper": None,
                "ci_width_95": None,
                "entropy_score": None,
                "peak_score": None,
                "interval_score": None,
                "variance_score": None,
                "hazard": None,
                "true_depth": None if true_depth is None else float(true_depth),
                "status": status,
            }

            if failure_reason is not None:
                result["failure_reason"] = failure_reason
            return result

        posterior_arr = np.asarray(posterior, dtype=float)
        if posterior_arr.ndim != 1 or posterior_arr.size != np.asarray(self.core.depths).size:
            raise ValueError("posterior shape is incompatible with the fusion depth grid")
        if not np.all(np.isfinite(posterior_arr)):
            raise ValueError("posterior contains non-finite values")

        summary = self.uq.compute(self.core.depths, posterior_arr, self.core.delta_x)
        hazard = self.hazard_assessor.evaluate(
            self.core.depths, posterior_arr, self.core.delta_x,
            fusion_confidence=float(summary["confidence"]),
            scene_complexity=float(scene_complexity))

        result = {
            "posterior": posterior_arr,
            "posterior_mass": self.core.posterior_mass(posterior_arr),
            "map_depth": float(summary["map_depth"]),
            "expected_depth": float(summary["expected_depth"]),
            "variance": float(summary["variance"]),
            "std": float(summary["std"]),
            "entropy": float(summary["entropy"]),
            "posterior_peak": float(summary["posterior_peak"]),
            "confidence": float(summary["confidence"]),
            "ci_lower": float(summary["ci_lower"]),
            "ci_upper": float(summary["ci_upper"]),
            "ci_width_95": float(summary.get(
                "ci_width_95", summary["ci_upper"] - summary["ci_lower"])),
            "entropy_score": float(summary["entropy_score"]),
            "peak_score": float(summary["peak_score"]),
            "interval_score": float(summary["interval_score"]),
            "variance_score": float(summary["variance_score"]),
            "hazard": hazard,
            "true_depth": None if true_depth is None else float(true_depth),
            "status": status,
        }

        if true_depth is not None and np.isfinite(true_depth):
            td = float(true_depth)
            result["abs_error_map"] = abs(result["map_depth"] - td)
            result["abs_error_mean"] = abs(result["expected_depth"] - td)

            try:
                mass = result["posterior_mass"]
                ci_lo, ci_hi = self.core.credible_interval(posterior_arr, mass_level=0.95)
                result["coverage_95"] = float(ci_lo <= td <= ci_hi)
                density_at_truth = float(np.interp(td, self.core.depths, posterior_arr))
                result["nll"] = float(-np.log(max(density_at_truth, 1e-15)))
                cdf = np.cumsum(mass)
                indicator = (np.asarray(self.core.depths) >= td).astype(float)
                result["crps"] = float(np.sum((cdf - indicator) ** 2) * self.core.delta_x)
            except Exception as exc: result["scoring_error"] = str(exc)
        return result

    def _wrap_experiment(
        self, name: str, mode: str,
        posterior_result: Dict[str, Any],
        weights: Dict[str, float],
        sigmas: Dict[str, float],
        active_mask: Sequence[bool],
        diagnostics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        payload = dict(posterior_result)
        payload.update({
            "experiment": name,
            "mode": mode,
            "weights": {k: float(v) for k, v in weights.items()},
            "sigmas": {k: float(v) for k, v in sigmas.items()},
            "active_mask": [bool(v) for v in active_mask],
            "diagnostics": diagnostics or {},
        })
        return payload

    @staticmethod
    def _zero_weights() -> Dict[str, float]:
        return {name: 0.0 for name in AblationFramework.SENSOR_NAMES}

    @staticmethod
    def _weights_from_active_mask(active_mask: np.ndarray) -> Dict[str, float]:
        count = int(np.sum(active_mask))
        weights = AblationFramework._zero_weights()
        if count == 0: return weights
        value = 1.0 / count
        for idx, name in enumerate(AblationFramework.SENSOR_NAMES):
            if bool(active_mask[idx]): weights[name] = float(value)
        return weights

    @staticmethod
    def _sigma_dict(sigmas: Sequence[float]) -> Dict[str, float]:
        return {
            name: float(value) for name, value in zip(AblationFramework.SENSOR_NAMES, sigmas)}

    # ------------------------------------------------------------------
    # Experiment executors
    # ------------------------------------------------------------------
    def _adaptive_case(
        self, label: str,
        measurements: np.ndarray,
        reliabilities: np.ndarray,
        active_mask: np.ndarray,
        context: Dict[str, float],
        true_depth: Optional[float],
        sensor_health: np.ndarray,
        base_sigmas: Tuple[float, float, float]) -> Dict[str, Any]:

        if not np.any(active_mask):
            posterior_result = self._normalize_posterior_result(
                None, true_depth=true_depth,
                scene_complexity=context["scene_complexity"],
                status="NO_VALID_SENSOR",
                failure_reason="No valid active sensor is available for this ablation case")

            return self._wrap_experiment(
                label, "adaptive-masked", posterior_result,
                self._zero_weights(),
                self._sigma_dict(base_sigmas),
                active_mask, {"active_sensor_count": 0})

        health_dict = {
            name: float(value) for name, value in zip(self.SENSOR_NAMES, sensor_health)}

        result = self.adaptive.fuse(
            z_lidar=float(measurements[0]) if np.isfinite(measurements[0]) else np.nan,
            z_ultra=float(measurements[1]) if np.isfinite(measurements[1]) else np.nan,
            z_radar=float(measurements[2]) if np.isfinite(measurements[2]) else np.nan,
            r_lidar=float(reliabilities[0]),
            r_ultra=float(reliabilities[1]),
            r_radar=float(reliabilities[2]),
            sigmas=base_sigmas,
            scene_complexity=float(context["scene_complexity"]),
            sensor_agreement=float(context["sensor_agreement"]),
            reliability_spread=float(context["reliability_spread"]),
            effective_sensor_count=float(context["effective_sensor_count"]),
            dominance_ratio=float(context["dominance_ratio"]),
            confidence=float(context["confidence"]),
            measurement_spread=float(context["measurement_spread"]),
            water_depth=float(context["water_depth"]),
            ntu=float(context["ntu"]),
            active_mask=tuple(bool(v) for v in active_mask),
            sensor_health=health_dict)

        status = str(result.get("status", "UNKNOWN"))
        posterior = result.get("posterior")

        posterior_result = self._normalize_posterior_result(
            posterior, true_depth=true_depth,
            scene_complexity=context["scene_complexity"],
            status=status, failure_reason=(
                None if posterior is not None else "Adaptive fusion returned no posterior"))

        return self._wrap_experiment(
            label, "adaptive-masked", posterior_result,
            result.get("weights", self._weights_from_active_mask(active_mask)),
            result.get("sigmas", self._sigma_dict(base_sigmas)), active_mask,
            {
                "adaptive_engine_status": status,
                "context": dict(context),
                "sensor_health": health_dict,
                "context_features_recomputed_for_ablation": True,
            })

    def _equal_weight_case(
        self, measurements: np.ndarray,
        active_mask: np.ndarray,
        true_depth: Optional[float],
        observable_context: Dict[str, float],
        base_sigmas: Tuple[float, float, float]) -> Dict[str, Any]:

        active_count = int(np.sum(active_mask))
        if active_count == 0:
            posterior_result = self._normalize_posterior_result(
                None, true_depth=true_depth,
                scene_complexity=observable_context["scene_complexity"],
                status="NO_VALID_SENSOR",
                failure_reason="No valid active sensor is available for the no-reliability ablation")

            return self._wrap_experiment(
                "E_No_Reliability", "equal-weight-opinion-pool", posterior_result,
                self._zero_weights(), self._sigma_dict(base_sigmas),
                active_mask, {"active_sensor_count": 0, "reliability_removed": True})

        neutral_rel_context = self._reliability_neutral_context(
            observable_context, active_sensor_count=active_count)
        all_unit_reliability = np.ones(3, dtype=float)
        all_healthy = np.ones(3, dtype=float)

        no_rel_sigmas = self._adaptive_base_sigmas(
            base_sigmas=base_sigmas,
            reliabilities=all_unit_reliability,
            measurements=measurements,
            context=neutral_rel_context,
            active_mask=active_mask,
            sensor_health=all_healthy)

        likelihoods = []
        active_indices = []
        for idx in range(3):
            if bool(active_mask[idx]) and np.isfinite(measurements[idx]):
                likelihoods.append(
                    self.core.compute_likelihood(float(measurements[idx]), float(no_rel_sigmas[idx])))
                active_indices.append(idx)

        if not likelihoods:
            posterior_result = self._normalize_posterior_result(
                None, true_depth=true_depth,
                scene_complexity=observable_context["scene_complexity"],
                status="NO_VALID_SENSOR",
                failure_reason="No finite likelihood input remains after reliability-neutral masking")

            return self._wrap_experiment(
                "E_No_Reliability", "equal-weight-opinion-pool", posterior_result,
                self._zero_weights(), self._sigma_dict(no_rel_sigmas),
                active_mask, {"active_sensor_count": 0, "reliability_removed": True})

        weights = np.ones(len(likelihoods), dtype=float)
        pooled = self.core.combine_likelihoods(likelihoods, weights)
        posterior = self.core.compute_posterior(pooled)

        equal_weights = self._zero_weights()
        weight_value = 1.0 / len(active_indices)
        for idx in active_indices:
            equal_weights[self.SENSOR_NAMES[idx]] = float(weight_value)

        posterior_result = self._normalize_posterior_result(
            posterior, true_depth=true_depth, scene_complexity=observable_context["scene_complexity"])

        return self._wrap_experiment(
            "E_No_Reliability", "equal-weight-opinion-pool",
            posterior_result, equal_weights,
            self._sigma_dict(no_rel_sigmas), active_mask,
            {
                "fusion": "equal_weights",
                "active_sensor_count": int(len(active_indices)),
                "reliability_removed": True,
                "health_removed": True,
                "context": neutral_rel_context,
            })

    def _neutral_adaptive_case(
        self, measurements: np.ndarray,
        reliabilities: np.ndarray,
        active_mask: np.ndarray,
        true_depth: Optional[float],
        base_sigmas: Tuple[float, float, float],
        sensor_health: np.ndarray) -> Dict[str, Any]:

        active_count = int(np.sum(active_mask))
        neutral = self._neutral_context(active_count)

        return self._adaptive_case(
            "F_No_Context", measurements, reliabilities, active_mask,
            neutral, true_depth, sensor_health, base_sigmas)

    def _fixed_case(
        self, measurements: np.ndarray,
        active_mask: np.ndarray,
        true_depth: Optional[float],
        scene_complexity: float,
        fixed_sigmas: Tuple[float, float, float]) -> Dict[str, Any]:

        active_count = int(np.sum(active_mask))
        sigma_dict = self._sigma_dict(fixed_sigmas)

        if active_count == 0:
            posterior_result = self._normalize_posterior_result(
                None, true_depth=true_depth,
                scene_complexity=scene_complexity,
                status="NO_VALID_SENSOR",
                failure_reason="No valid active sensor is available for the fixed baseline")

            return self._wrap_experiment(
                "G_No_Adaptive", "fixed", posterior_result, self._zero_weights(), sigma_dict,
                active_mask, {"fusion": "fixed_equal_weight_llop", "active_sensor_count": 0})

        if active_count == 3:
            result = self.fixed.estimate(
                z_lidar=float(measurements[0]),
                z_ultra=float(measurements[1]),
                z_radar=float(measurements[2]),
                sigmas=fixed_sigmas)

            posterior = result.get("posterior")
            weights = result.get("weights", self._weights_from_active_mask(active_mask))
        else:
            likelihoods = []
            active_indices = []
            for idx in range(3):
                if bool(active_mask[idx]) and np.isfinite(measurements[idx]):
                    likelihoods.append(self.core.compute_likelihood(
                        float(measurements[idx]), float(fixed_sigmas[idx])))
                    active_indices.append(idx)

            if not likelihoods:
                posterior = None
                weights = self._zero_weights()
            else:
                pooled = self.core.combine_likelihoods(
                    likelihoods,
                    np.ones(len(likelihoods), dtype=float))
                posterior = self.core.compute_posterior(pooled)
                weights = self._weights_from_active_mask(active_mask)

        status = "OK" if posterior is not None else "NO_VALID_SENSOR"
        posterior_result = self._normalize_posterior_result(
            posterior, true_depth=true_depth,
            scene_complexity=scene_complexity, status=status,
            failure_reason=(None if posterior is not None else "Fixed fusion returned no posterior"))

        return self._wrap_experiment(
            "G_No_Adaptive", "fixed", posterior_result, weights, sigma_dict, active_mask, {
                "fusion": "fixed_equal_weight_llop",
                "active_sensor_count": active_count,
                "sigmas_are_calibrated_base_values": True,
            })

    # ------------------------------------------------------------------
    # Main execution pipeline
    # ------------------------------------------------------------------
    def run_experiments(
        self, z_lidar: Any, z_ultra: Any, z_radar: Any,
        r_lidar: Any, r_ultra: Any, r_radar: Any,*,
        true_depth: Optional[float] = None,
        scene_complexity: float = 0.0,
        sensor_agreement: Optional[float] = None,
        reliability_spread: Optional[float] = None,
        effective_sensor_count: Optional[float] = None,
        dominance_ratio: Optional[float] = None,
        confidence: Optional[float] = None,
        measurement_spread: Optional[float] = None,
        water_context_depth_proxy: Optional[float] = None,
        ntu: Optional[float] = None,
        water_depth: Optional[float] = None,
        sensor_health: Optional[Dict[str, float]] = None,
        fixed_sigmas: Optional[Tuple[float, float, float]] = None,
        base_sigmas: Optional[Tuple[float, float, float]] = None,
        adaptive_sigmas: Optional[Tuple[float, float, float]] = None,
        active_mask: Optional[Sequence[bool]] = None) -> Dict[str, Dict[str, Any]]:

        measurements = np.asarray([
            self._measurement(z_lidar),
            self._measurement(z_ultra),
            self._measurement(z_radar),
        ], dtype=float)

        reliabilities = np.asarray([
            self._reliability(r_lidar),
            self._reliability(r_ultra),
            self._reliability(r_radar),
        ], dtype=float)

        resolved_active = self._resolve_active_mask(
            measurements, reliabilities, active_mask)
        sensor_health_arr = self._resolve_sensor_health(
            sensor_health, resolved_active)

        if water_context_depth_proxy is None and water_depth is not None:
            water_context_depth_proxy = water_depth

        base_sigmas_resolved = self._resolve_base_sigmas(
            fixed_sigmas=fixed_sigmas,
            base_sigmas=base_sigmas,
            adaptive_sigmas=adaptive_sigmas)

        true_depth_value = None
        if true_depth is not None:
            true_depth_value = self._safe_float(true_depth, np.nan)
            if not np.isfinite(true_depth_value):
                true_depth_value = None

        context_all = self._build_observable_context(
            measurements, reliabilities, resolved_active,
            scene_complexity, water_context_depth_proxy, ntu)

        results: Dict[str, Dict[str, Any]] = {}
        results["A_All_Sensors"] = self._adaptive_case(
            "A_All_Sensors", measurements, reliabilities, resolved_active,
            context_all, true_depth_value, sensor_health_arr, base_sigmas_resolved)

        sensor_drop_masks = {
            "B_No_LiDAR": np.asarray([False, True, True], dtype=bool),
            "C_No_Radar": np.asarray([True, True, False], dtype=bool),
            "D_No_Ultra": np.asarray([True, False, True], dtype=bool),
        }

        for label, keep_mask in sensor_drop_masks.items():
            case_mask = resolved_active & keep_mask
            case_health = np.where(case_mask, sensor_health_arr, 0.0)
            case_context = self._build_observable_context(
                measurements, reliabilities, case_mask, scene_complexity,
                water_context_depth_proxy, ntu)
            results[label] = self._adaptive_case(
                label, measurements, reliabilities, case_mask,
                case_context, true_depth_value, case_health, base_sigmas_resolved)

        no_reliability_active = np.isfinite(measurements)
        if active_mask is not None:
            no_reliability_active &= np.asarray(active_mask, dtype=bool)

        results["E_No_Reliability"] = self._equal_weight_case(
            measurements, no_reliability_active, true_depth_value,
            self._build_observable_context(
                measurements, np.ones(3, dtype=float), no_reliability_active,
                scene_complexity, water_context_depth_proxy, ntu), base_sigmas_resolved)

        results["F_No_Context"] = self._neutral_adaptive_case(
            measurements, reliabilities, resolved_active, true_depth_value,
            base_sigmas_resolved, sensor_health_arr)

        results["G_No_Adaptive"] = self._fixed_case(
            measurements, resolved_active, true_depth_value,
            context_all["scene_complexity"], base_sigmas_resolved)
        return results

__all__ = ["AblationFramework"]