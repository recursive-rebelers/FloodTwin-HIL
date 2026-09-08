import numpy as np
import random

class UltrasonicModel:

    def __init__(self, base_noise_sigma=0.15, seed=42):
        self.base_noise_sigma = float(base_noise_sigma)
        random.seed(seed)
        np.random.seed(seed)

    def generate(self, true_depth, water_depth):
        water_depth = max(0.0, float(water_depth))
        true_depth = float(true_depth)

        # Dry road / no water case
        if water_depth <= 0.0:
            measurement = true_depth + np.random.normal(0.0, self.base_noise_sigma)
            return {
                "distance": round(max(0.0, measurement), 2),
                "surface_echo": 0.0,
                "bottom_echo": 1.0,
                "bottom_confidence": 1.0,
                "echo_ambiguity": 0.0,
                "measurement_type": "Bottom",
                "surface_sigma": 0.0,
                "bottom_sigma": round(self.base_noise_sigma, 3),
            }

        # Echo probabilities
        surface_echo = 0.15 + 0.38 * (1.0 - np.exp(-water_depth / 8.5))
        surface_echo = np.clip(surface_echo, 0.15, 0.60)

        bottom_echo = (1.0 - surface_echo) * np.exp(-0.012 * water_depth)
        bottom_echo = np.clip(bottom_echo, 0.05, 0.90)

        # Geometry / penetration effect
        penetration_factor = 0.45 + 0.15 * np.exp(-water_depth / 10.0)
        effective_surface_depth = penetration_factor * water_depth

        # Noise grows mildly with water depth
        surface_sigma = self.base_noise_sigma * (2.2 + 0.08 * water_depth)
        bottom_sigma = self.base_noise_sigma * (1.0 + 0.05 * water_depth)

        # Candidate measurements
        surface_distance = (true_depth - effective_surface_depth + np.random.normal(0.0, surface_sigma))
        bottom_bias = 0.02 * water_depth + 0.002 * water_depth ** 2
        bottom_distance = (true_depth + bottom_bias + np.random.normal(0.0, bottom_sigma))

        # Confidence / ambiguity
        echo_ambiguity = np.clip(1.0 - abs(bottom_echo - surface_echo), 0.0, 1.0)

        bottom_confidence = bottom_echo * np.exp(-0.04 * water_depth)
        bottom_confidence *= np.exp(-0.35 * self.base_noise_sigma)
        bottom_confidence = np.clip(bottom_confidence, 0.05, 0.95)

        # Mixed readings become more likely when echoes are ambiguous
        mixed_probability = np.clip(0.30 * echo_ambiguity, 0.0, 0.45)

        if random.random() < mixed_probability:
            distance = (0.60 * bottom_distance + 0.40 * surface_distance + np.random.normal(
                0.0, self.base_noise_sigma * 3.0))
            measurement_type = "Mixed"

        elif random.random() < bottom_confidence:
            distance = bottom_distance
            measurement_type = "Bottom"

        else:
            distance = surface_distance
            measurement_type = "Surface"

        return {
            "distance": round(max(0.0, distance), 2),
            "surface_echo": round(surface_echo, 3),
            "bottom_echo": round(bottom_echo, 3),
            "bottom_confidence": round(bottom_confidence, 3),
            "echo_ambiguity": round(echo_ambiguity, 3),
            "measurement_type": measurement_type,
            "surface_sigma": round(surface_sigma, 3),
            "bottom_sigma": round(bottom_sigma, 3),
        }

    def reliability(self,
        water_depth,
        measurement_type=None,
        surface_echo=None,
        bottom_echo=None,
        bottom_confidence=None,
        echo_ambiguity=None,
        ultrasonic_error=None):

        water_depth = max(0.0, float(water_depth))

        # Fallback echo estimates if the caller only supplies water depth
        if surface_echo is None:
            surface_echo = 0.15 + 0.38 * (1.0 - np.exp(-water_depth / 8.5))

        if bottom_echo is None:
            bottom_echo = max(0.05, 1.0 - float(surface_echo))

        surface_echo = float(np.clip(surface_echo, 0.0, 1.0))
        bottom_echo = float(np.clip(bottom_echo, 0.0, 1.0))

        total_echo = surface_echo + bottom_echo + 1e-9
        echo_ratio = bottom_echo / total_echo

        if bottom_confidence is None:
            bottom_confidence = bottom_echo

        if echo_ambiguity is None:
            echo_ambiguity = np.clip(1.0 - abs(bottom_echo - surface_echo), 0.0, 1.0)

        bottom_confidence = float(np.clip(bottom_confidence, 0.0, 1.0))
        echo_ambiguity = float(np.clip(echo_ambiguity, 0.0, 1.0))

        # Depth penalty: water reduces trust in ultrasonic ranging
        depth_factor = np.exp(-0.045 * water_depth)

        # Clear separation between echoes should improve trust
        clarity = abs(bottom_echo - surface_echo) / total_echo
        clarity_factor = np.clip(clarity, 0.0, 1.0)

        # Measurement-type trust
        if measurement_type is None:
            type_factor = 0.85
        else:
            mt = str(measurement_type).strip().lower()
            
            if mt == "bottom":
                type_factor = 1.00
            elif mt == "mixed":
                type_factor = 0.68
            elif mt == "surface":
                type_factor = 0.46                
            else:
                type_factor = 0.80

        # Composite reliability
        reliability = (0.10
            + 0.34 * echo_ratio
            + 0.18 * depth_factor
            + 0.30 * bottom_confidence
            + 0.18 * type_factor
            + 0.10 * clarity_factor)

        reliability *= np.exp(-0.95 * echo_ambiguity)
        error_penalty = 1.0

        if ultrasonic_error is not None:
            ultrasonic_error = abs(float(ultrasonic_error))
            error_penalty = np.exp(-0.12 * ultrasonic_error)

        reliability *= error_penalty
        reliability = np.clip(reliability, 0.25, 1.00)
        return round(float(reliability), 3)

    def degradation_factor(self,
        water_depth,
        measurement_type=None,
        surface_echo=None,
        bottom_echo=None,
        bottom_confidence=None,
        echo_ambiguity=None,
        ultrasonic_error=None):

        degradation = 1.0 - self.reliability(
            water_depth=water_depth,
            measurement_type=measurement_type,
            surface_echo=surface_echo,
            bottom_echo=bottom_echo,
            bottom_confidence=bottom_confidence,
            echo_ambiguity=echo_ambiguity,
            ultrasonic_error=ultrasonic_error)
        return round(float(degradation), 3)

    def metadata(self):
        return {
            "sensor": "JSN-SR04T",
            "sensor_type": "Ultrasonic_ToF",
            "base_noise_sigma": self.base_noise_sigma,
            "model_type": "Phenomenological Digital Twin",
            "measurement_interpretation":
                "Depth-equivalent ultrasonic observation",

            "physical_model": [
                "Surface Echo Competition",
                "Water-Layer Echo Geometry",
                "Bottom Echo Detection",
                "Echo Ambiguity Modeling",
                "Gaussian Time-of-Flight Noise",
                "Exponential Reliability Decay",
                "Mixed Echo Generation",
                "Bottom Confidence Estimation",
                "Measurement-Type-Aware Reliability",
            ],

            "assumptions": [
                "Surface and bottom echo behavior is represented "
                "using engineering-defined phenomenological relationships.",
                "The model abstracts ultrasonic sensing behavior "
                "rather than simulating full acoustic propagation.",
                "Noise, echo, confidence, and reliability parameters "
                "are synthetic Digital Twin parameters.",
                "The distance output represents a depth-equivalent "
                "observation rather than a manufacturer-calibrated "
                "submerged pothole-depth measurement."
            ]
        }