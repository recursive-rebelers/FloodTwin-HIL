import numpy as np

# Turbidity (NTU) simulation model
class TurbidityModel:

    # Initialize turbidity noise level
    def __init__(self, noise_sigma=5):
        self.noise_sigma = noise_sigma

    # Generate a simulated NTU reading
    def generate(self, true_ntu):

        # Add Gaussian noise to the true NTU value
        measured_ntu = true_ntu + np.random.normal(
            0,
            self.noise_sigma
        )

        # NTU cannot be negative
        measured_ntu = max(0, measured_ntu)

        # Return simulated NTU value
        return round(measured_ntu, 2)