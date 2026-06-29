import numpy as np

class LidarModel:
    def __init__(self, base_noise_sigma=0.5, refraction_factor=0.15):


        """
        Initializes the VL53L1X LiDAR twin with base physical properties.
        """

        self.base_noise_sigma = base_noise_sigma
        self.refraction_factor = refraction_factor

    def generate(self, true_depth, water_depth, ntu):


        """
        Simulates measured distance based on true depth and environmental factors.
        """
        # 1. Start with baseline noise and zero bias
        noise_sigma = self.base_noise_sigma
        refraction_bias = 0.0
        measurement_uncertainty_multiplier = 1.0 

        # 2. Water Effects: Increases base noise and adds refraction bias
        if water_depth > 0:
            refraction_bias = water_depth * self.refraction_factor
            noise_sigma += 1.2 

        # 3. Turbidity (NTU) Effects: Muddy water scatters the signal
        if ntu > 0:
            measurement_uncertainty_multiplier += (ntu / 100.0)

        # Calculate the final dynamic Gaussian noise
        dynamic_noise = np.random.normal(
            loc=0.0, 
            scale=(noise_sigma * measurement_uncertainty_multiplier)
        )

        # 4. Final Sensor Output Calculation
        lidar_distance = true_depth + refraction_bias + dynamic_noise

        return round(lidar_distance, 2)