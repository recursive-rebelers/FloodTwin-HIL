import numpy as np

class ReliabilityFusion:

    def __init__(self):
        self.sensor_prior = {
            "lidar": 1.05,
            "ultrasonic": 1.0,
            "radar": 1.20
        }

    def compute_adaptive_weights(self, r_lidar, r_ultra, r_radar):
        trust_lidar = (r_lidar * self.sensor_prior["lidar"])
        trust_ultra = (r_ultra * self.sensor_prior["ultrasonic"])
        trust_radar = (r_radar * self.sensor_prior["radar"])

        total = (trust_lidar + trust_ultra + trust_radar)

        if total < 1e-6:
            return (1/3, 1/3, 1/3)

        w_lidar = trust_lidar / total
        w_ultra = trust_ultra / total
        w_radar = trust_radar / total

        return w_lidar, w_ultra, w_radar

    def fusion_confidence(self, weights):
        return round(np.max(weights), 3)

    def metadata(self):

        return {
            "strategy": "Adaptive Reliability Fusion",
            "priors": self.sensor_prior
        }

fusion_engine = ReliabilityFusion()

def compute_adaptive_weights(r_lidar, r_ultra, r_radar):
    return fusion_engine.compute_adaptive_weights(r_lidar, r_ultra, r_radar)


def compute_confidence(r_lidar, r_ultra, r_radar):
    weights = compute_adaptive_weights(r_lidar, r_ultra, r_radar)
    return fusion_engine.fusion_confidence(weights)