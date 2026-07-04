import numpy as np

class HazardAssessment:

    def __init__(self, threshold_cm=10.0, decision_boundary=0.75):

        self.threshold = float(threshold_cm)
        self.decision_boundary = float(decision_boundary)

    def evaluate(self, depth_space, posterior, delta_x):

        depth_space = np.asarray(depth_space, dtype=float)
        posterior = np.asarray(posterior, dtype=float)

        if depth_space.ndim != 1 or posterior.ndim != 1:
            raise ValueError("depth_space and posterior must be one-dimensional arrays")

        if len(depth_space) != len(posterior):
            raise ValueError("depth_space and posterior must have the same length")

        mass = posterior * float(delta_x)
        mass_sum = mass.sum()

        if not np.isfinite(mass_sum) or mass_sum <= 1e-12:
            mass = np.ones_like(depth_space, dtype=float) / len(depth_space)
        else:
            mass = mass / mass_sum

        hazard_mask = depth_space >= self.threshold
        hazard_prob = float(np.sum(mass[hazard_mask]))

        if hazard_prob < 0.45:
            status = "SAFE"
        elif hazard_prob < self.decision_boundary:
            status = "CAUTION"
        else:
            status = "HAZARD"

        hazard_depth = depth_space[hazard_mask]
        severity = np.sum(mass[hazard_mask] * hazard_depth)
        risk_score = (0.7 * hazard_prob + 0.3 * (severity / 30)) * 100

        return {
            "hazard_probability": round(hazard_prob, 5),
            "hazard_prob": round(hazard_prob, 5),
            "risk_score": round(risk_score, 2),
            "status": status,
            "threshold_cm": self.threshold,
            "decision_boundary": self.decision_boundary
        }

    def metadata(self):

        return {
            "method": "Bayesian Exceedance Probability",
            "threshold_cm": self.threshold,
            "decision_boundary": self.decision_boundary
        }