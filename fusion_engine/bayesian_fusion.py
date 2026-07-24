import math
import numpy as np
from typing import Optional, Sequence, Tuple, Dict, Any

class BayesianFusionCore:
    
    def __init__(self, depth_min: float = 0.0, depth_max: float = 30.0, resolution: int = 300):
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)

        if resolution < 2:
            raise ValueError("resolution must be at least 2")

        self.depths = np.linspace(self.depth_min, self.depth_max, int(resolution), dtype=float)
        self.delta_x = float(self.depths[1] - self.depths[0])

        # Uniform prior over the grid
        self.prior = np.ones_like(self.depths, dtype=float)
        self.prior /= float(np.sum(self.prior) * self.delta_x)

    def _broadcast_to_grid(self, values: Any, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape == self.depths.shape:
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            arr = np.broadcast_to(arr, self.depths.shape).astype(float)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        except ValueError as exc:
            raise ValueError(f"{name} must be broadcastable to the depth grid shape") from exc

    def _normalize_density(self, density: Any) -> np.ndarray:
        density = np.asarray(density, dtype=float)
        density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
        density = np.clip(density, 0.0, None)
        total = float(np.sum(density) * self.delta_x)

        if not np.isfinite(total) or total <= 1e-12:
            return np.ones_like(self.depths, dtype=float) / (len(self.depths) * self.delta_x)
        return density / total

    def _posterior_mass_from_density(self, posterior_density: Any) -> np.ndarray:
        mass = np.asarray(posterior_density, dtype=float) * self.delta_x
        mass = np.nan_to_num(mass, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(mass))

        if not np.isfinite(total) or total <= 1e-12:
            return np.ones_like(self.depths, dtype=float) / len(self.depths)
        return mass / total

    def _log_gaussian_density(self, measurement: float, sigma: float) -> np.ndarray:
        sigma = float(max(sigma, 1e-6))
        variance = sigma * sigma

        # Exact Gaussian log-density up to the normalization term
        exponent = -0.5 * ((self.depths - measurement) ** 2) / variance
        exponent -= float(np.max(exponent))
        return exponent

    def _effective_support(self, posterior_mass: np.ndarray) -> float:
        posterior_mass = np.asarray(posterior_mass, dtype=float)
        posterior_mass = np.nan_to_num(posterior_mass, nan=0.0, posinf=0.0, neginf=0.0)
        denom = float(np.sum(np.square(posterior_mass)))
        if denom <= 1e-12:
            return float(len(posterior_mass))
        return float(1.0 / denom)

    def posterior_mass(self, posterior_density: Any) -> np.ndarray:
        return self._posterior_mass_from_density(posterior_density)

    def compute_likelihood(self, measurement: float, sigma: float) -> np.ndarray:
        measurement = float(np.clip(measurement, self.depth_min, self.depth_max))
        sigma = float(max(sigma, 1e-6))
        exponent = self._log_gaussian_density(measurement, sigma)
        likelihood = np.exp(exponent)
        return self._normalize_density(likelihood)

    def combine_likelihoods(self,
        likelihoods: Sequence[np.ndarray],
        weights: Optional[Sequence[float]] = None,
        temperature: float = 0.92,
    ) -> np.ndarray:

        if not likelihoods:
            raise ValueError("At least one likelihood is required")

        likelihoods = [self._broadcast_to_grid(L, "likelihood") for L in likelihoods]

        if weights is None:
            weights = np.ones(len(likelihoods), dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)

        if weights.ndim != 1 or len(weights) != len(likelihoods):
            raise ValueError("weights must be a 1D array with the same length as likelihoods")

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weight_sum = float(np.sum(weights))

        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            weights = np.ones(len(likelihoods), dtype=float) / len(likelihoods)
        else:
            weights = weights / weight_sum

        log_pool = np.zeros_like(self.depths, dtype=float)

        for likelihood, weight in zip(likelihoods, weights):
            log_pool += float(weight) * np.log(np.clip(likelihood, 1e-15, None))

        adaptive_temperature = (0.92 + 0.05 * (1.0 - np.max(weights)))
        temperature = float(np.clip(0.5 * temperature + 0.5 * adaptive_temperature, 0.88, 1.00))

        if temperature != 1.0:
            log_pool /= temperature

        log_pool -= float(np.max(log_pool))
        pooled = np.exp(log_pool)
        return self._normalize_density(pooled)

    def compute_posterior(self, likelihood: Any, prior: Optional[Any] = None) -> np.ndarray:
        if prior is None:
            prior_density = self.prior
        else:
            prior_density = self._broadcast_to_grid(prior, "prior")
            prior_density = np.clip(prior_density, 0.0, None)

        likelihood = self._broadcast_to_grid(likelihood, "likelihood")
        posterior = prior_density * likelihood
        return self._normalize_density(posterior)

    def map_estimate(self, posterior: Any) -> float:
        mass = self.posterior_mass(posterior)
        return float(self.depths[int(np.argmax(mass))])

    def expected_depth(self, posterior: Any) -> float:
        mass = self.posterior_mass(posterior)
        return float(np.sum(self.depths * mass))

    def posterior_variance(self, posterior: Any) -> float:
        mass = self.posterior_mass(posterior)
        mean = self.expected_depth(posterior)
        return float(np.sum(((self.depths - mean) ** 2) * mass))

    def entropy(self, posterior: Any) -> float:
        mass = self.posterior_mass(posterior)
        return float(-np.sum(mass * np.log(np.clip(mass, 1e-12, None))))

    def credible_interval(self, posterior: Any, mass_level: float = 0.95) -> Tuple[float, float]:
        mass_level = float(mass_level)
        
        if not (0.0 < mass_level < 1.0):
            raise ValueError("mass_level must be between 0 and 1")

        mass = self.posterior_mass(posterior)
        cdf = np.cumsum(mass)

        lower_q = (1.0 - mass_level) / 2.0
        upper_q = 1.0 - lower_q

        lower_idx = int(min(np.searchsorted(cdf, lower_q), len(self.depths) - 1))
        upper_idx = int(min(np.searchsorted(cdf, upper_q), len(self.depths) - 1))
        return float(self.depths[lower_idx]), float(self.depths[upper_idx])

    def summary(self, posterior: Any, mass_level: float = 0.95) -> Dict[str, Any]:
        posterior_density = self._normalize_density(posterior)
        posterior_mass = self.posterior_mass(posterior_density)
        ci_lower, ci_upper = self.credible_interval(posterior_density, mass_level=mass_level)

        map_idx = int(np.argmax(posterior_mass))
        map_depth = float(self.depths[map_idx])
        mean_depth = self.expected_depth(posterior_density)
        variance = self.posterior_variance(posterior_density)
        std = float(math.sqrt(max(variance, 0.0)))
        entropy = self.entropy(posterior_density)

        uniform_mass = np.ones_like(self.depths, dtype=float) / len(self.depths)
        uniform_entropy = float(-np.sum(uniform_mass * np.log(uniform_mass)))
        entropy_score = float(np.clip(1.0 - (entropy / max(uniform_entropy, 1e-12)), 0.0, 1.0))

        posterior_peak = float(np.max(posterior_mass))
        reference_peak = 1.0 / len(self.depths)
        peak_score = float(np.clip((posterior_peak - reference_peak) / max(1.0 - reference_peak, 1e-12), 0.0, 1.0))

        span = max(float(self.depths[-1] - self.depths[0]), self.delta_x)
        interval_score = float(np.clip(1.0 - ((ci_upper - ci_lower) / span), 0.0, 1.0))

        uniform_variance_ref = (span ** 2) / 12.0
        variance_score = float(np.clip(1.0 - (variance / max(uniform_variance_ref, 1e-12)), 0.0, 1.0))

        effective_support = self._effective_support(posterior_mass)
        support_fraction = float(np.clip(1.0 - ((effective_support - 1.0) / max(len(self.depths) - 1.0, 1.0)), 0.0, 1.0))

        map_expected_gap = abs(map_depth - mean_depth)
        map_expected_gap_score = float(np.clip(1.0 - (map_expected_gap / span), 0.0, 1.0))

        confidence = np.clip(
            0.31 * entropy_score +
            0.26 * peak_score +
            0.19 * interval_score +
            0.16 * variance_score +
            0.08 * support_fraction,
            0.0, 1.0)

        return {
            "map_depth": round(map_depth, 3),
            "expected_depth": round(mean_depth, 3),
            "variance": round(variance, 5),
            "std": round(std, 5),
            "entropy": round(entropy, 5),
            "posterior_peak": round(posterior_peak, 5),
            "confidence": round(confidence, 5),
            "ci_lower": round(ci_lower, 3),
            "ci_upper": round(ci_upper, 3),

            "entropy_score": round(entropy_score, 5),
            "peak_score": round(peak_score, 5),
            "interval_score": round(interval_score, 5),
            "variance_score": round(variance_score, 5),

            "effective_support": round(effective_support, 5),
            "support_fraction": round(support_fraction, 5),
            "map_expected_gap": round(map_expected_gap, 5),
            "map_expected_gap_score": round(map_expected_gap_score, 5),

            "mass_sum": round(float(np.sum(posterior_density) * self.delta_x), 6),
            "mass_sum_raw": round(float(np.sum(np.asarray(posterior, dtype=float)) * self.delta_x), 6),
            "confidence_components": {
                "entropy_weighted": round(0.31 * entropy_score, 5),
                "peak_weighted": round(0.26 * peak_score, 5),
                "interval_weighted": round(0.19 * interval_score, 5),
                "variance_weighted": round(0.16 * variance_score, 5),
                "support_weighted": round(0.08 * support_fraction, 5),
            },
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "Bayesian Fusion Core",
            "depth_range_cm": (float(self.depths[0]), float(self.depths[-1])),
            "resolution": len(self.depths),
            "delta_x": float(self.delta_x),
            "likelihood_model": "Gaussian on depth grid",
            "pooling": "Tempered weighted log-opinion pool",
        }