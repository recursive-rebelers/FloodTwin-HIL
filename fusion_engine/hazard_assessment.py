import numpy as np

class HazardAssessment:

    def __init__(self, threshold_cm=10.0, decision_boundary=0.75):
        self.threshold = float(threshold_cm)
        self.decision_boundary = float(decision_boundary)

    def evaluate(self, depth_space, posterior, delta_x, fusion_confidence=None, scene_complexity=None):
        depth_space = np.asarray(depth_space, dtype=float).ravel()
        posterior = np.asarray(posterior, dtype=float).ravel()
        delta_x = float(delta_x)

        if depth_space.ndim != 1 or posterior.ndim != 1:
            raise ValueError("depth_space and posterior must be one-dimensional arrays")

        if len(depth_space) != len(posterior):
            raise ValueError("depth_space and posterior must have the same length")

        if len(depth_space) == 0:
            raise ValueError("depth_space and posterior must not be empty")

        mass = posterior * delta_x
        mass_sum = float(np.sum(mass))

        if not np.isfinite(mass_sum) or mass_sum <= 1e-12:
            mass = np.ones_like(depth_space, dtype=float) / len(depth_space)
        else:
            mass = mass / mass_sum

        hazard_mask = depth_space >= self.threshold
        hazard_prob = float(np.sum(mass[hazard_mask]))

        if fusion_confidence is None:
            confidence_factor = 1.0
        else:
            confidence_factor = float(np.clip(fusion_confidence, 0.0, 1.0))

        effective_probability = (hazard_prob * (0.80 + 0.20 * confidence_factor))
        effective_probability = float(np.clip(effective_probability, 0.0, 1.0))

        if effective_probability < 0.45:
            status = "SAFE"
        elif effective_probability < self.decision_boundary:
            status = "CAUTION"
        else:
            status = "HAZARD"

        max_depth = float(np.max(depth_space))
        depth_span = max(max_depth, 1e-6)

        hazard_depth = depth_space[hazard_mask]
        hazard_mass = mass[hazard_mask]

        if hazard_prob > 1e-12:
            expected_hazard_depth = float(np.sum(hazard_mass * hazard_depth) / hazard_prob)
        else:
            expected_hazard_depth = self.threshold

        severity_index = float(np.clip((
            expected_hazard_depth - self.threshold) / max(depth_span - self.threshold, 1e-6), 0.0, 1.0))

        if scene_complexity is None:
            scene_factor = 0.0
        else:
            scene_factor = float(np.clip(scene_complexity, 0.0, 1.0))

        operational_risk = (
            0.55 * hazard_prob
            + 0.25 * severity_index
            + 0.12 * (1.0 - confidence_factor)
            + 0.08 * scene_factor)

        risk_score = float(np.clip(operational_risk * 100.0, 0.0, 100.0))

        return {
            "hazard_probability": round(hazard_prob, 5),
            "severity_index": round(severity_index, 5),
            "risk_score": round(risk_score, 2),
            "status": status,
            "threshold_cm": self.threshold,
            "decision_boundary": self.decision_boundary,
            "depth_span_cm": round(depth_span, 3),
            "fusion_confidence": round(confidence_factor, 5),
            "scene_complexity": round(scene_factor, 5),
            "effective_probability": round(effective_probability,5),
            "expected_hazard_depth": round(expected_hazard_depth,3),
        }

    def metadata(self):
        return {
            "method": "Posterior Hazard Exceedance Assessment",
            "threshold_cm": self.threshold,
            "decision_boundary": self.decision_boundary,
            "hazard_definition": "Hazard is defined as posterior probability mass at or above the configured depth threshold.",
            "hazard_probability": "Posterior exceedance probability P(D >= threshold | observations)",
            "decision_rule": "Confidence-adjusted hazard score with SAFE, CAUTION, and HAZARD decision regions.",
            "risk_model": "Weighted engineering risk index combining hazard exceedance, conditional depth severity, fusion confidence, and scene complexity.",
            "risk_score_range": [0.0, 100.0],

            "outputs": [
                "hazard_probability",
                "decision_score",
                "severity_index",
                "expected_hazard_depth",
                "risk_score",
                "status",
                "threshold_cm",
                "decision_boundary",
                "depth_span_cm",
                "fusion_confidence",
                "scene_complexity",
            ],

            "assumptions": [
                "The hazard threshold is an engineering-defined decision criterion.",
                "Hazard probability is computed from posterior probability mass above the threshold.",
                "Severity index represents normalized conditional expected depth above the hazard threshold.",
                "Fusion confidence is an engineered posterior-concentration score rather than a calibrated probability.",
                "Scene complexity is a bounded contextual factor in the range [0, 1].",
                "The operational risk score is an engineering index and is not a calibrated probability of physical risk."
            ]
        }