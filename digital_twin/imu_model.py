import math
import random

SEED = 42
random.seed(SEED)
G = 9.81

SURFACE_FACTOR = {
    "Asphalt": 1.0,
    "Concrete": 0.85,
    "Gravel": 2.5,
    "Dirt": 3.0,
    "Mud": 3.5
}

ENVIRONMENT_FACTOR = {
    "Urban": 1.0,
    "Rural": 1.2
}

def IMUModel(
    road_type, road_environment,
    vehicle_speed, pothole_depth, water_depth,
    base_pitch=0, base_roll=0):

    roughness = (SURFACE_FACTOR.get(road_type, 1.0) * ENVIRONMENT_FACTOR.get(road_environment, 1.0))
    water_dampening = max(0.65, 1.0 - (water_depth / max(pothole_depth, 1)) * 0.35)

    pitch = base_pitch + pothole_depth * random.uniform(0.12, 0.28)
    roll = base_roll + pothole_depth * random.uniform(0.08, 0.22)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    ax = -G * math.sin(pitch_rad)
    ay = G * math.sin(roll_rad) * math.cos(pitch_rad)
    az = G * math.cos(roll_rad) * math.cos(pitch_rad)

    normalized_speed = vehicle_speed / 30.0
    vibration = 0.02 + 0.05 * roughness * (normalized_speed ** 2)

    ax += random.gauss(0, vibration)
    ay += random.gauss(0, vibration)
    az += random.gauss(0, vibration)

    impact = (pothole_depth * 0.15) * (vehicle_speed / 20.0) * water_dampening

    ax += random.uniform(-impact, impact)
    ay += random.uniform(-impact * 0.3, impact * 0.3)
    az += random.uniform(-impact * 1.2, impact * 0.2)

    roll_est = math.atan2(ay, math.sqrt(ax**2 + az**2))
    pitch_est = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    acceleration = math.sqrt(ax**2 + ay**2 + az**2)
    reliability = max(0.25, math.exp(-1.5 * vibration))

    return {
        "acceleration": round(acceleration, 3),
        "pitch": round(math.degrees(pitch_est), 3),
        "roll": round(math.degrees(roll_est), 3),
        "reliability": round(reliability, 3)
    }

def metadata(self):
    return {
        "sensor": "MPU6050",
        "sensor_type": "6-DoF_IMU",
        "model_type": "Phenomenological Digital Twin",
        "measurement_interpretation":
            "Synthetic vehicle-motion response under pothole traversal",

        "physical_model": [
            "Gravity-Projection Under Vehicle Attitude",
            "Road-Surface-Dependent Vibration",
            "Speed-Dependent Vibration",
            "Water-Dependent Impact Dampening",
            "Pothole-Induced Impact Response",
            "Stochastic Measurement Noise",
            "Motion-Condition Reliability"
        ],

        "assumptions": [
            "Vehicle pitch and roll responses are represented using "
            "engineering-defined phenomenological relationships.",
            "Road roughness factors are synthetic relative parameters.",
            "The model does not simulate full vehicle suspension or "
            "multi-body dynamics.",
            "Reliability represents simulation-derived confidence "
            "rather than a manufacturer-specified sensor reliability."
        ]
    }