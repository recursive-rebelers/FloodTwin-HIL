import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fusion_engine.bayesian_fusion import BayesianFusionCore

class AdaptiveFusionEngine:

    def __init__(self, core: BayesianFusionCore,
        gamma=1.40,
        weight_blend=0.60,
        sigma_scale=1.0,
        sigma_offset=0.08,
        sigma_min=0.45,
        sigma_max=2.85):

        self.core = core
        self.gamma = float(gamma)
        self.weight_blend = float(weight_blend)
        self.sigma_scale = float(sigma_scale)
        self.sigma_offset = float(sigma_offset)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def _normalize_weights(self, r_lidar, r_ultra, r_radar):

        reliabilities = np.array([r_lidar, r_ultra, r_radar], dtype=float)
        reliabilities = np.clip(reliabilities, 0.0, 1.0)
        reliabilities = np.power(reliabilities, self.gamma)

        total = float(np.sum(reliabilities))

        if not np.isfinite(total) or total <= 1e-12:
            return (1 / 3, 1 / 3, 1 / 3)

        raw_weights = reliabilities / total
        uniform = np.full(3, 1.0 / 3.0, dtype=float)

        blend = float(np.clip(self.weight_blend, 0.0, 1.0))
        weights = blend * raw_weights + (1.0 - blend) * uniform
        weights /= np.sum(weights)

        return tuple(float(w) for w in weights)

    def _infer_sigmas(self, r_lidar, r_ultra, r_radar):

        reliabilities = np.array([r_lidar, r_ultra, r_radar], dtype=float)
        reliabilities = np.clip(reliabilities, 1e-3, 1.0)

        sigmas = self.sigma_scale / (np.sqrt(reliabilities) + self.sigma_offset)
        sigmas = np.clip(sigmas, self.sigma_min, self.sigma_max)

        return tuple(float(s) for s in sigmas)

    def fuse(self, z_lidar, z_ultra, z_radar, r_lidar, r_ultra, r_radar, sigmas=None):

        w_lidar, w_ultra, w_radar = self._normalize_weights(r_lidar, r_ultra, r_radar)

        if sigmas is None:
            sigmas = self._infer_sigmas(r_lidar, r_ultra, r_radar)
        else:
            sigmas = tuple(float(s) for s in sigmas)

        l_lidar = self.core.compute_likelihood(z_lidar, sigmas[0])
        l_ultra = self.core.compute_likelihood(z_ultra, sigmas[1])
        l_radar = self.core.compute_likelihood(z_radar, sigmas[2])

        pooled_likelihood = self.core.combine_likelihoods([l_lidar, l_ultra, l_radar], [w_lidar, w_ultra, w_radar])
        posterior = self.core.compute_posterior(pooled_likelihood)

        depth_map = self.core.map_estimate(posterior)
        depth_mean = self.core.expected_depth(posterior)
        
        variance = self.core.posterior_variance(posterior)
        entropy = self.core.entropy(posterior)

        return {
            "posterior": posterior,
            "depth_map": float(depth_map),
            "depth_mean": float(depth_mean),
            "variance": float(variance),
            "entropy": float(entropy),
            "weights": {
                "lidar": float(w_lidar),
                "ultrasonic": float(w_ultra),
                "radar": float(w_radar)
        },
            "sigmas": {
                "lidar": float(sigmas[0]),
                "ultrasonic": float(sigmas[1]),
                "radar": float(sigmas[2])
            }
        }

    def metadata(self):

        return {
            "method": "Adaptive Reliability-Aware Bayesian Fusion",
            "fusion": "Reliability-Weighted Log-Linear Opinion Pool",
            "weight_formula": "w_i ∝ R_i^γ",
            "sigma_formula": "σ_i = scale / (√R_i + offset)",

            "weighting_strategy": {
                "gamma": self.gamma,
                "weight_blend": self.weight_blend
        },
            "uncertainty_model": {
                "sigma_scale": self.sigma_scale,
                "sigma_offset": self.sigma_offset,
                "sigma_bounds": (self.sigma_min, self.sigma_max)
        },
            "posterior_outputs": [
                "MAP Estimate",
                "Expected Depth",
                "Variance",
                "Entropy",
                "Adaptive Weights",
                "Adaptive Sigmas"
        ],
            "calibration":
                "Empirically calibrated over simulated deployment scenarios"
        }