import math
import random

SEED = 42
random.seed(SEED)
G = 9.81

SURFACE_FACTOR = {
    "Asphalt": 1.0,
    "Concrete": 0.9,
    "Gravel": 2.2,
    "Dirt": 2.8,
    "Mud": 3.2
}

ENVIRONMENT_FACTOR = {
    "Urban": 1.0,
    "Rural": 1.3
}

def generate_mpu6050_metrics(
    road_type, road_environment,
    vehicle_speed, pothole_depth, water_depth,
    base_pitch=0, base_roll=0):

    roughness = (SURFACE_FACTOR.get(road_type, 1.0) * ENVIRONMENT_FACTOR.get(road_environment, 1.0))
    water_factor = max(0.7, 1 - (water_depth / max(pothole_depth, 1)) * 0.3)

    pitch = base_pitch + pothole_depth * random.uniform(0.12, 0.28)
    roll = base_roll + pothole_depth * random.uniform(0.08, 0.22)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    ax = -G * math.sin(pitch_rad)
    ay = G * math.sin(roll_rad) * math.cos(pitch_rad)
    az = G * math.cos(roll_rad) * math.cos(pitch_rad)

    vibration = (0.03 + 0.04 * roughness * (vehicle_speed / 12)) * water_factor

    ax += random.gauss(0, vibration)
    ay += random.gauss(0, vibration)
    az += random.gauss(0, vibration)

    impact = math.sqrt(pothole_depth) * 0.25 * water_factor

    ax += random.uniform(-impact, impact)
    ay += random.uniform(-impact / 2, impact / 2)
    az += random.uniform(-impact, impact)

    roll_est = math.atan2(ay, math.sqrt(ax**2 + az**2))
    pitch_est = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    acceleration = math.sqrt(ax**2 + ay**2 + az**2)
    reliability = max(0.6, 1 - vibration / 2)

    return {
        "acceleration": round(acceleration, 3),
        "pitch": round(math.degrees(pitch_est), 3),
        "roll": round(math.degrees(roll_est), 3),
        "reliability": round(reliability, 3)
    }