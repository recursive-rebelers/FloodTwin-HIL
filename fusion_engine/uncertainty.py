import numpy as np

class UncertaintyQuantification:

    def compute(self, depth_space, posterior, delta_x):

        depth_space = np.asarray(depth_space, dtype=float).ravel()
        posterior = np.asarray(posterior, dtype=float).ravel()
        delta_x = float(delta_x)

        if depth_space.ndim != 1 or posterior.ndim != 1:
            raise ValueError("depth_space and posterior must be one-dimensional arrays")

        if len(depth_space) != len(posterior):
            raise ValueError("depth_space and posterior must have the same length")

        mass = posterior * delta_x
        mass_sum = float(np.sum(mass))

        if not np.isfinite(mass_sum) or mass_sum <= 1e-12:
            mass = np.ones_like(depth_space, dtype=float) / len(depth_space)
        else:
            mass = mass / mass_sum

        map_idx = int(np.argmax(mass))
        map_depth = float(depth_space[map_idx])

        expected_depth = float(np.sum(depth_space * mass))
        variance = float(np.sum(((depth_space - expected_depth) ** 2) * mass))
        std = float(np.sqrt(max(variance, 0.0)))

        clipped_mass = np.clip(mass, 1e-12, None)
        entropy = float(-np.sum(mass * np.log(clipped_mass)))
        posterior_peak = float(np.max(mass))

        uniform = np.ones_like(depth_space, dtype=float) / len(depth_space)
        max_entropy = float(-np.sum(uniform * np.log(uniform)))
        entropy_score = np.clip(1.0 - (entropy / max_entropy), 0.0, 1.0)

        reference_peak = 0.038
        peak_score = np.clip(posterior_peak / reference_peak, 0.0, 1.0)

        cdf = np.cumsum(mass)
        lower_idx = min(np.searchsorted(cdf, 0.025), len(depth_space) - 1)
        upper_idx = min(np.searchsorted(cdf, 0.975), len(depth_space) - 1)

        ci_lower = float(depth_space[lower_idx])
        ci_upper = float(depth_space[upper_idx])

        span = max(float(depth_space[-1] - depth_space[0]), delta_x)
        interval_score = np.clip(1.0 - ((ci_upper - ci_lower) / span), 0.0, 1.0)

        variance_score = np.exp(-variance / 3.0)
        variance_score = np.clip(variance_score, 0, 1)
       
        peak_term = 0.85 * peak_score # Posterior concentration        
        interval_term = 0.50 * interval_score # Distribution compactness       
        entropy_term = -0.70 * entropy_score # Information uncertainty penalty       
        variance_term = -0.35 * variance_score # Dispersion penalty

        confidence = float(np.clip(peak_term + interval_term + entropy_term + variance_term, 0.0, 1.0))

        return {
            "map_depth": round(map_depth, 3),
            "expected_depth": round(expected_depth, 3),
            "variance": round(variance, 5),
            "std": round(std, 5),
            "entropy": round(entropy, 5),
            "posterior_peak": round(posterior_peak, 5),
            "confidence": round(confidence, 5),
            "ci_lower": round(ci_lower, 3),
            "ci_upper": round(ci_upper, 3),
            "entropy_score": entropy_score,
            "peak_score": peak_score,
            "interval_score": interval_score,
            "variance_score": variance_score
        }

    def metadata(self):

        return {
            "method": "Bayesian Uncertainty Quantification",
            "confidence": "Weighted Posterior Confidence Index",
            "confidence_components": {
                "posterior_concentration": "0.85 × peak_score",
                "interval_compactness": "0.50 × interval_score",
                "entropy_penalty": "-0.70 × entropy_score",
                "variance_penalty": "-0.35 × variance_score"
        },
            "reference_peak": reference_peak,
            "interval": "95% Credible Interval",
            "confidence_calibration": "Empirically calibrated over simulated deployment scenarios",
            "posterior_metrics": [
                "MAP Estimate",
                "Expected Value",
                "Variance",
                "Entropy",
                "Posterior Peak",
                "95% Credible Interval"
            ]
        }