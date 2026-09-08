import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion_engine.bayesian_fusion import BayesianFusionCore

class FixedFusionEngine:

    def __init__(self, core: BayesianFusionCore):
        self.core = core
        self.weights = (1/3, 1/3, 1/3)

    def estimate(self, z_lidar, z_ultra, z_radar, sigmas=(1.0, 1.0, 1.0)):
        wl, wu, wr = self.weights
        l_lidar = self.core.compute_likelihood(z_lidar, sigmas[0])
        l_ultra = self.core.compute_likelihood(z_ultra, sigmas[1])
        l_radar = self.core.compute_likelihood(z_radar, sigmas[2])

        pooled_likelihood = self.core.combine_likelihoods([l_lidar, l_ultra, l_radar], self.weights)
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
                "lidar": float(wl),
                "ultrasonic": float(wu),
                "radar": float(wr),
            }
        }

    def metadata(self):
        return {
            "method": "Fixed-Weight Bayesian Posterior Fusion",
            "fusion": "Tempered Weighted Log-Opinion Pooling",
            "weighting_strategy": "Equal Fixed Weights",
            "weights": {"lidar": 1.0 / 3.0, "ultrasonic": 1.0 / 3.0, "radar": 1.0 / 3.0},
            "likelihood_model": "Gaussian Depth-Observation Likelihood",
            "posterior_formulation": "Uniform prior combined with the fixed-weight tempered pooled likelihood",

            "estimation_outputs": [
                "MAP Depth",
                "Expected Depth",
                "Posterior Variance",
                "Posterior Entropy"
            ],

            "assumptions": [
                "All sensor observations receive equal fixed fusion weights.",
                "Sensor likelihoods are represented as Gaussian depth-observation models.",
                "Likelihoods are combined using weighted logarithmic opinion pooling.",
                "The resulting pooled likelihood is combined with a uniform prior over the modeled depth domain.",
                "The fixed-weight configuration serves as a non-adaptive baseline for comparison with reliability-aware fusion."
            ]
        }