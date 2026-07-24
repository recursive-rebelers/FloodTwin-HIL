import numpy as np

class UncertaintyQuantification:
    
    def __init__(self, reference_peak_fraction=0.04, mass_level=0.95):
        self.reference_peak_fraction = float(reference_peak_fraction)
        self.mass_level = float(mass_level)

    @staticmethod
    def _as_1d_array(values):
        arr = np.asarray(values, dtype=float).ravel()
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _safe_clip01(value):
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _safe_entropy(mass):
        mass = np.clip(np.asarray(mass, dtype=float), 1e-12, 1.0)
        mass = mass / np.sum(mass)
        return float(-np.sum(mass * np.log(mass)))

    @staticmethod
    def _effective_support(mass):
        mass = np.asarray(mass, dtype=float)
        mass = np.clip(mass, 1e-12, 1.0)
        denom = np.sum(mass ** 2)
        if denom <= 1e-12:
            return float(len(mass))
        return float(1.0 / denom)

    @staticmethod
    def _credible_interval_from_mass(depth_space, mass, mass_level=0.95):
        mass_level = float(np.clip(mass_level, 0.50, 0.999))
        cdf = np.cumsum(mass)
        lower_q = (1.0 - mass_level) / 2.0
        upper_q = 1.0 - lower_q

        lower_idx = int(np.searchsorted(cdf, lower_q, side="left"))
        upper_idx = int(np.searchsorted(cdf, upper_q, side="left"))
        lower_idx = min(max(lower_idx, 0), len(depth_space) - 1)
        upper_idx = min(max(upper_idx, 0), len(depth_space) - 1)
        return float(depth_space[lower_idx]), float(depth_space[upper_idx])

    def posterior_mass(self, posterior_density, delta_x):
        posterior_density = self._as_1d_array(posterior_density)
        delta_x = float(delta_x)

        if posterior_density.size == 0:
            raise ValueError("posterior must contain at least one element")
        if not np.isfinite(delta_x) or delta_x <= 0.0:
            raise ValueError("delta_x must be a positive finite value")

        mass = posterior_density * delta_x
        mass = np.clip(mass, 0.0, None)
        total = float(np.sum(mass))

        if not np.isfinite(total) or total <= 1e-12:
            mass = np.ones_like(mass, dtype=float) / len(mass)
        else:
            mass = mass / total
        return mass

    def compute(self, depth_space, posterior, delta_x, mass_level=None):
        depth_space = self._as_1d_array(depth_space)
        posterior = self._as_1d_array(posterior)
        delta_x = float(delta_x)

        if depth_space.size == 0 or posterior.size == 0:
            raise ValueError("depth_space and posterior must be non-empty")
        if depth_space.size != posterior.size:
            raise ValueError("depth_space and posterior must have the same length")
        if not np.isfinite(delta_x) or delta_x <= 0.0:
            raise ValueError("delta_x must be a positive finite value")

        mass = self.posterior_mass(posterior, delta_x)
        map_idx = int(np.argmax(mass))
        map_depth = float(depth_space[map_idx])

        expected_depth = float(np.sum(depth_space * mass))
        variance = float(np.sum(((depth_space - expected_depth) ** 2) * mass))
        std = float(np.sqrt(max(variance, 0.0)))

        entropy = self._safe_entropy(mass)
        posterior_peak = float(np.max(mass))

        ci_level = self.mass_level if mass_level is None else float(mass_level)
        ci_lower, ci_upper = self._credible_interval_from_mass(depth_space, mass, ci_level)
        ci_width = float(max(ci_upper - ci_lower, 0.0))

        span = float(max(depth_space[-1] - depth_space[0], delta_x))
        map_expected_gap = abs(map_depth - expected_depth)
        gap_score = self._safe_clip01(1.0 - map_expected_gap / span)
        max_entropy = float(np.log(len(depth_space))) if len(depth_space) > 1 else 1.0
        uniform_peak = 1.0 / max(len(depth_space), 1)
        uniform_variance = (span ** 2) / 12.0 if span > 0.0 else 1.0

        entropy_score = self._safe_clip01(1.0 - (entropy / max(max_entropy, 1e-12)))
        peak_reference = max(self.reference_peak_fraction, uniform_peak * 1.5)
        peak_score = self._safe_clip01(
            (posterior_peak - uniform_peak) / max(peak_reference - uniform_peak, 1e-12))

        interval_score = self._safe_clip01(1.0 - (ci_width / span))
        variance_score = self._safe_clip01(1.0 - (variance / max(uniform_variance, 1e-12)))

        effective_support = self._effective_support(mass)
        support_fraction = self._safe_clip01(1.0 - ((effective_support - 1.0) / max(len(mass) - 1.0, 1.0)))

        # Confidence is a concentration index: higher when posterior mass is sharp,
        # compact, and less entropic, all terms are on [0,1]
        confidence = (
            0.27 * peak_score +
            0.25 * interval_score +
            0.23 * entropy_score +
            0.17 * variance_score +
            0.08 * support_fraction)

        confidence = self._safe_clip01(confidence)

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
            "ci_width_95": round(ci_width, 3),
            "entropy_score": round(entropy_score, 5),
            "peak_score": round(peak_score, 5),
            "interval_score": round(interval_score, 5),
            "variance_score": round(variance_score, 5),
            "mass_level": round(float(ci_level), 3),
            "effective_support": round(effective_support,5),
            "support_fraction": round(support_fraction,5),
            "map_expected_gap": round(map_expected_gap,5),
            "gap_score": round(gap_score,5),

            "confidence_components": {
                "peak": round(0.27 * peak_score,5),
                "interval": round(0.25 * interval_score,5),
                "entropy": round(0.23 * entropy_score,5),
                "variance": round(0.17 * variance_score,5),
                "support": round(0.08 * support_fraction,5),
            },
        }

    def credible_interval(self, depth_space, posterior, delta_x, mass_level=None):
        depth_space = self._as_1d_array(depth_space)
        posterior = self._as_1d_array(posterior)
        delta_x = float(delta_x)
        
        if depth_space.size != posterior.size:
            raise ValueError("depth_space and posterior must have the same length")

        mass = self.posterior_mass(posterior, delta_x)
        ci_level = self.mass_level if mass_level is None else float(mass_level)
        return self._credible_interval_from_mass(depth_space, mass, ci_level)

    def metadata(self):
        return {
            "method": "Bayesian Uncertainty Quantification",
            "mass_level": self.mass_level,
            "reference_peak_fraction": self.reference_peak_fraction,
            "confidence": "Concentration index from posterior peak, interval width, entropy, and variance",

            "confidence_components": {
                "posterior_concentration": "0.27 × peak_score",
                "interval_compactness": "0.25 × interval_score",
                "entropy_concentration": "0.23 × entropy_score",
                "variance_concentration": "0.17 × variance_score",
                "support_concentration": "0.08 × support_fraction",
            },

            "posterior_metrics": [
                "MAP Estimate",
                "Expected Value",
                "Variance",
                "Standard Deviation",
                "Entropy",
                "Posterior Peak",
                "95% Credible Interval",
                "95% Interval Width",
                "Normalized Scores",
                "Effective Support",
                "Support Fraction",
                "MAP-Expected Gap",
                "Confidence Component Breakdown",
            ],

            "notes": [
                "Input posterior is normalized to a valid mass distribution before scoring",
                "Confidence is bounded to [0,1] and remains stable under degenerate posteriors",
            ],
        }