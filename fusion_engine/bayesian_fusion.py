import numpy as np

class BayesianFusionCore:

    def __init__(self, depth_min=0.0, depth_max=30.0, resolution=300):

        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.depths = np.linspace(depth_min, depth_max, resolution)

        if len(self.depths) < 2:
            raise ValueError("resolution must be at least 2")

        self.delta_x = self.depths[1] - self.depths[0]
        self.prior = np.ones(resolution, dtype=float)
        self.prior /= (np.sum(self.prior) * self.delta_x)

    def _normalize_density(self, density):

        density = np.asarray(density, dtype=float)
        total = np.sum(density) * self.delta_x

        if not np.isfinite(total) or total <= 1e-12:
            return np.ones_like(self.depths) / (len(self.depths) * self.delta_x)

        return density / total

    def posterior_mass(self, posterior_density):

        mass = np.asarray(posterior_density, dtype=float) * self.delta_x
        total = mass.sum()

        if not np.isfinite(total) or total <= 1e-12:
            return np.ones_like(self.depths) / len(self.depths)

        return mass / total

    def compute_likelihood(self, measurement, sigma):

        variance = max(float(sigma) ** 2, 1e-6)
        exponent = (-0.5 * ((self.depths - measurement) ** 2 / variance))
        exponent -= np.max(exponent)

        likelihood = np.exp(exponent)
        return self._normalize_density(likelihood)

    def combine_likelihoods(self, likelihoods, weights=None):

        if not likelihoods:
            raise ValueError("At least one likelihood is required")

        likelihoods = [np.asarray(L, dtype=float)
            for L in likelihoods
        ]

        if weights is None:
            weights = np.ones(len(likelihoods), dtype=float)

        weights = np.asarray(weights, dtype=float)
        weight_sum = weights.sum()

        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            weights = np.ones(len(likelihoods), dtype=float) / len(likelihoods)
        else:
            weights = weights / weight_sum

        log_pool = np.zeros_like(self.depths, dtype=float)

        for likelihood, weight in zip(likelihoods, weights):
            log_pool += weight * np.log(np.clip(likelihood, 1e-15, None))

        log_pool -= np.max(log_pool)
        pooled = np.exp(log_pool)

        return self._normalize_density(pooled)

    def compute_posterior(self, likelihood):

        posterior = self.prior * np.asarray(likelihood, dtype=float)
        return self._normalize_density(posterior)

    def map_estimate(self, posterior):

        posterior_mass = self.posterior_mass(posterior)
        return float(self.depths[np.argmax(posterior_mass)])

    def expected_depth(self, posterior):

        mass = self.posterior_mass(posterior)
        return float(np.sum(self.depths * mass))

    def posterior_variance(self, posterior):

        mass = self.posterior_mass(posterior)
        mean = self.expected_depth(posterior)
        return float(np.sum(((self.depths - mean) ** 2) * mass))

    def entropy(self, posterior):

        mass = self.posterior_mass(posterior)
        return float(-np.sum(mass * np.log(np.clip(mass, 1e-12, None))))

    def credible_interval(self, posterior, mass_level=0.95):

        mass = self.posterior_mass(posterior)
        cdf = np.cumsum(mass)

        lower_q = (1.0 - mass_level) / 2.0
        upper_q = 1.0 - lower_q

        lower_idx = min(np.searchsorted(cdf, lower_q), len(self.depths) - 1)
        upper_idx = min(np.searchsorted(cdf, upper_q), len(self.depths) - 1)

        return float(self.depths[lower_idx]), float(self.depths[upper_idx])

    def metadata(self):

        return {
            "method": "Bayesian Fusion Core",
            "depth_range_cm": (float(self.depths[0]), float(self.depths[-1])),
            "resolution": len(self.depths),
            "delta_x": float(self.delta_x)
        }