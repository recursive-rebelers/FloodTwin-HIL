import csv
import math
import os
import random


def generate_mpu6050_metrics(road_type,
                             vehicle_speed,
                             pothole_depth,
                             base_pitch=0.0,
                             base_roll=0.0):
    """
    Simulate MPU6050 IMU readings.
    """

    g = 9.81

    # Road roughness multiplier
    road_roughness = {
        "Asphalt": 1.0,
        "Concrete": 0.8,
        "Urban": 1.2,
        "Rural": 1.5,
        "Gravel": 2.5,
        "Dirt": 3.0,
        "Mud": 3.5
    }.get(road_type, 1.0)

    # Vehicle tilts slightly when crossing deeper potholes
    base_pitch += pothole_depth * random.uniform(0.15, 0.30)
    base_roll += pothole_depth * random.uniform(0.10, 0.25)

    pitch_rad = math.radians(base_pitch)
    roll_rad = math.radians(base_roll)

    # Gravity components
    ax = -g * math.sin(pitch_rad)
    ay = g * math.sin(roll_rad) * math.cos(pitch_rad)
    az = g * math.cos(roll_rad) * math.cos(pitch_rad)

    # Road vibration + speed + pothole
    noise = (
        0.02
        + (vehicle_speed / 8.0) * 0.05 * road_roughness
        + (pothole_depth / 20.0) * 0.10
    )

    ax += random.gauss(0, noise)
    ay += random.gauss(0, noise)
    az += random.gauss(0, noise)

    # Sudden impact caused by pothole
    impact = pothole_depth * 0.08

    ax += random.uniform(-impact, impact)
    ay += random.uniform(-impact / 2, impact / 2)
    az += random.uniform(-impact, impact)

    # Calculate pitch and roll
    calc_roll = math.atan2(ay, math.sqrt(ax ** 2 + az ** 2))
    calc_pitch = math.atan2(-ax, math.sqrt(ay ** 2 + az ** 2))

    # Total acceleration
    acceleration = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    return {
        "acceleration": round(acceleration, 2),
        "pitch": round(math.degrees(calc_pitch), 2),
        "roll": round(math.degrees(calc_roll), 2)
    }


def process_registry(registry_dir):

    file_path = os.path.join(registry_dir, "scenario_registry.csv")

    if not os.path.exists(file_path):
        print("Scenario Registry not found.")
        return

    print("\n Virtual MPU6050 Sensor \n")

    with open(file_path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            scenario_id = row["scenario_id"]
            road_type = row["road_type"]
            pothole_depth = float(row["pothole_depth"])
            vehicle_speed = float(row["vehicle_speed"])

            # Slight initial vehicle inclination
            base_pitch = 2.0 if road_type in ["Mud", "Dirt"] else 0.0
            base_roll = -1.5 if road_type == "Gravel" else 0.0

            imu = generate_mpu6050_metrics(
                road_type=road_type,
                vehicle_speed=vehicle_speed,
                pothole_depth=pothole_depth,
                base_pitch=base_pitch,
                base_roll=base_roll
            )

            print(f"Scenario ID : {scenario_id}")
            print(f"Road Type   : {road_type}")
            print(f"Pothole     : {pothole_depth:.1f} cm")
            print(f"Speed       : {vehicle_speed:.2f} m/s")
            print(f"Acceleration: {imu['acceleration']} m/s²")
            print(f"Pitch       : {imu['pitch']}°")
            print(f"Roll        : {imu['roll']}°")
            print("-" * 45)


if __name__ == "__main__":

    workspace_path = r"C:/B.Tech-IT 2023-2027/PROJECT_INTERVIEW/FloodTwin-HIL/scenarios"

    process_registry(workspace_path)