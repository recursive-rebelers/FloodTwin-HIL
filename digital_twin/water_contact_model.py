import random

SEED = 42
random.seed(SEED)

class FSIR01Model:

    def __init__(self, noise_sigma=0.15):
        self.noise_sigma = noise_sigma

    def detect(self, water_depth):

        effective_depth = max(0, water_depth + random.gauss(0, self.noise_sigma))

        if effective_depth <= 0.5:
            state = "Dry"
            confidence = 0.98

        elif effective_depth <= 3:
            state = "Shallow Puddle"
            confidence = 0.90

        else:
            state = "Flooded Pothole"
            confidence = 0.95

        return {
            "state": state,
            "water_present": effective_depth > 0.5,
            "confidence": round(confidence, 3),
            "reliability": round(confidence, 3)
        }