import numpy as np
import random

class UltrasonicModel:
    def __init__(self, base_noise_sigma=0.2):
        self.base_noise_sigma = base_noise_sigma

    def generate(self, true_depth, water_depth):
        if water_depth <= 0:
            # Dry road: accurate reading
            distance = true_depth + np.random.normal(0, self.base_noise_sigma)
            return round(distance, 2)

        # Flooded road: Sound bounces off the surface or the bottom
        surface_echo_prob = min(0.90, 0.5 + (water_depth * 0.05)) 
        
        if random.random() < surface_echo_prob:
            # ERROR: Reads the water surface
            surface_depth = true_depth - water_depth
            ultrasonic_distance = surface_depth + np.random.normal(0, self.base_noise_sigma * 2) 
        else:
            # SUCCESS: Reads the bottom
            ultrasonic_distance = true_depth + np.random.normal(0, self.base_noise_sigma)

        return round(ultrasonic_distance, 2)