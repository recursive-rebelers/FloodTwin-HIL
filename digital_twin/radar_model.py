import numpy as np
import random
# Radar sensor simulation
class RadarModel:

    # Initialize radar noise level
    def __init__(self, base_noise_sigma=0.1):
        self.base_noise_sigma = base_noise_sigma

    # Generate radar depth reading
    def generate(self, true_depth, water_depth):

        # Add Gaussian noise to the true depth
        radar_distance = true_depth + np.random.normal(
            0,
            self.base_noise_sigma
        )
        # Simulate clutter in deeper water
        if water_depth > 5:
            clutter_probability = 0.15

            # Apply clutter with 15% probability
            if random.random() < clutter_probability:
                radar_distance += np.random.normal(0.5, 0.2)

        # Return simulated radar reading
        return round(radar_distance, 2)
# Test the Radar model

if __name__ == "__main__":

    radar = RadarModel()

    for water in [0, 2, 5, 8, 10]:
        value = radar.generate(15, water)
        print(f"Water Depth = {water} cm -> Radar = {value} cm")