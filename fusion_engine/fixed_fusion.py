import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fusion_engine.bayesian_fusion import BayesianFusionCore

class FixedFusionEngine:

    def __init__(self, core: BayesianFusionCore):

        self.core = core
        self.weights = (1 / 3, 1 / 3, 1 / 3)

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
                "radar": float(wr)
            }
        }

    def metadata(self):

        return {
            "method": "Fixed Bayesian Fusion",
            "fusion": "Log-Linear Opinion Pool",
            "weights": {
                "lidar": 0.3333333333,
                "ultrasonic": 0.3333333333,
                "radar": 0.3333333333
            }
        }