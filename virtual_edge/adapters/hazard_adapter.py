from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from fusion_engine.hazard_assessment import HazardAssessment

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)

def _posterior_confidence_from_fusion(fusion_result: Dict[str, Any]) -> float:
    if not isinstance(fusion_result, dict):
        return 0.5

    candidates = (
        fusion_result.get("posterior_confidence", None),
        fusion_result.get("fusion_confidence", None),
        (fusion_result.get("uncertainty", {}) or {}).get("confidence", None))

    for candidate in candidates:
        try:
            if candidate is None:
                continue
            value = float(candidate)
            if np.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
        except Exception:
            continue
    return 0.5

def _extract_scene_complexity(fusion_result: Dict[str, Any], fallback: float) -> float:
    if not isinstance(fusion_result, dict):
        return float(np.clip(fallback, 0.0, 1.0))

    for key in ("scene_complexity", "scene_factor", "context_complexity"):
        if key in fusion_result:
            value = _safe_float(fusion_result.get(key), fallback)
            return float(np.clip(value, 0.0, 1.0))
    return float(np.clip(fallback, 0.0, 1.0))

def _validate_and_normalize_posterior(posterior: np.ndarray, delta_x: float, eps: float = 1e-12) -> tuple[np.ndarray, float]:
    posterior = np.asarray(posterior, dtype=float).ravel()
    posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)

    if posterior.size == 0:
        raise ValueError("posterior must not be empty")

    if not np.all(np.isfinite(posterior)):
        raise ValueError("posterior contains non-finite values")

    delta_x = float(delta_x)
    if not np.isfinite(delta_x) or delta_x <= 0.0:
        raise ValueError("delta_x must be a positive finite value")

    mass = posterior * delta_x
    mass_sum = float(np.sum(mass))

    if not np.isfinite(mass_sum) or mass_sum <= eps:
        raise ValueError("posterior mass is degenerate or invalid")

    # Normalize to a valid density if needed
    target_mass = 1.0
    if not np.isclose(mass_sum, target_mass, atol=1e-3, rtol=1e-3):
        posterior = posterior / mass_sum
    return posterior, mass_sum

@dataclass
class HazardAdapter:

    threshold_cm: float = 10.0
    decision_boundary: float = 0.75

    def __post_init__(self) -> None:
        self.hazard = HazardAssessment(threshold_cm=self.threshold_cm, decision_boundary=self.decision_boundary)

    def evaluate(self, fusion_result: Dict[str, Any], scene_complexity: float = 0.0) -> Dict[str, Any]:
        if not isinstance(fusion_result, dict):
            raise TypeError("fusion_result must be a dictionary")

        if "core" not in fusion_result:
            raise RuntimeError("fusion_result is missing required key: 'core'")

        core = fusion_result["core"]
        if core is None or not hasattr(core, "depths") or not hasattr(core, "delta_x"):
            raise RuntimeError("fusion_result['core'] must expose 'depths' and 'delta_x'")

        if "posterior" not in fusion_result:
            raise RuntimeError("fusion_result is missing required key: 'posterior'")

        posterior = np.asarray(fusion_result["posterior"], dtype=float)
        posterior, incoming_mass_sum = _validate_and_normalize_posterior(posterior, float(core.delta_x))
        normalized_mass_sum = float(np.sum(posterior) * core.delta_x)

        confidence = _posterior_confidence_from_fusion(fusion_result)
        scene_complexity_val = _extract_scene_complexity(fusion_result, scene_complexity)

        hazard_output = self.hazard.evaluate(
            depth_space=core.depths,
            posterior=posterior,
            delta_x=core.delta_x,
            fusion_confidence=confidence,
            scene_complexity=scene_complexity_val)

        out = dict(fusion_result)
        out.update({
            "posterior": posterior,
            "incoming_posterior_mass": float(incoming_mass_sum),
            "posterior_mass_sum": float(normalized_mass_sum),
            "posterior_confidence": float(confidence),
            "scene_complexity": float(scene_complexity_val),
            "hazard": dict(hazard_output), **hazard_output,

            "metadata": {
                "hazard_engine": "Bayesian Hazard Assessment",
                "threshold_cm": self.threshold_cm,
                "decision_boundary": self.decision_boundary,
            }})

        return out