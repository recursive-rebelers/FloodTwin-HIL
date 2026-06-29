import csv
import os
import random


class FSIR01Model:
    
   # Virtual FS-IR01 Water Detection Sensor
   # Output: categorical water state
    

    def detect_water_state(self, water_depth):

        # small sensor noise (realistic uncertainty)
        noise = random.gauss(0, 0.15)
        effective_depth = water_depth + noise

        # classification logic
        if effective_depth <= 0.5:
            return "Dry"

        elif 0.5 < effective_depth <= 3.0:
            return "Shallow Puddle"

        else:
            return "Flooded Pothole"


def process_registry(registry_dir):

    file_path = os.path.join(registry_dir, "scenario_registry.csv")

    if not os.path.exists(file_path):
        print("Scenario registry not found.")
        return

    sensor = FSIR01Model()

    print("\n FS-IR01 Water Condition Sensor \n")

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:

            scenario_id = row["scenario_id"]
            water_depth = float(row["water_depth"])

            water_state = sensor.detect_water_state(water_depth)

            print(f"Scenario ID   : {scenario_id}")
            print(f"Water Depth   : {water_depth:.2f} cm")
            print(f"Water State   : {water_state}")
            print("-" * 45)


if __name__ == "__main__":

    path = r"C:/B.Tech-IT 2023-2027/PROJECT_INTERVIEW/FloodTwin-HIL/scenarios"

    process_registry(path)