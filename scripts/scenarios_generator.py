import csv
import random
from pathlib import Path

TARGET = ((Path(__file__).resolve().parent.parent) / "scenarios")
SEED = 42
random.seed(SEED)

ROAD_TYPES = [
    'Asphalt',
    'Concrete',
    'Gravel',
    'Dirt',
    'Mud' ]

ROAD_ENVIRONMENTS = [
    'Urban',
    'Rural' ]

WEATHER_OPTIONS = [
    'Clear',
    'Cloudy',
    'Drizzle',
    'Rainy',
    'Foggy',
    'Mist',
    'Thunderstorm' ]

LIGHTING_OPTIONS = [
    'Dawn',
    'Daylight',
    'Dusk',
    'Night_Streetlights',
    'Night_Dark' ]

def generate_ntu(weather):
    if weather == 'Clear':
        return random.randint(0,100)
    elif weather == 'Cloudy':
        return random.randint(50,180)
    elif weather == 'Drizzle':
        return random.randint(80,220)
    elif weather == 'Rainy':
        return random.randint(120,320)
    elif weather == 'Foggy':
        return random.randint(100,250)
    elif weather == 'Mist':
        return random.randint(80,200)
    elif weather == 'Thunderstorm':
        return random.randint(250,500)
    return random.randint(0,500)

def compute_severity(depth, water, ntu):
    score = (0.4 * water + 0.4 * (ntu / 500) * 10 + 0.2 * (depth / 20) * 10)

    if score < 3:
        return 'Low'
    elif score < 6:
        return 'Moderate'
    elif score < 8:
        return 'Severe'
    else:
        return 'Extreme'

def generate_speed():
    return round(random.uniform(2, 12), 2)

def generate_scenarios(output_dir='.', num_scenarios=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = (output_dir / 'scenario_registry.csv')
    scenarios = []

    for i in range(1, num_scenarios+1):
        scenario_id = f"SC{i:03d}"
        road_type = random.choice(ROAD_TYPES)
        road_environment = random.choice(ROAD_ENVIRONMENTS)
        weather = random.choice(WEATHER_OPTIONS)
        lighting = random.choice(LIGHTING_OPTIONS)
        pothole_depth = random.randint(5, 20)
        water_depth = random.randint(0, pothole_depth)
        ntu = generate_ntu(weather)
        vehicle_speed = generate_speed()
        severity = compute_severity(pothole_depth, water_depth, ntu)

        scenarios.append([
            scenario_id,
            road_environment,
            road_type,
            pothole_depth,
            water_depth,
            ntu,
            weather,
            lighting,
            vehicle_speed,
            severity ])

    headers = [
        'scenario_id',
        'road_environment',
        'road_type',
        'pothole_depth',
        'water_depth',
        'ntu',
        'weather',
        'lighting',
        'vehicle_speed',
        'severity' ]

    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(scenarios)

    print("\n"+"="*37)
    print(f"{num_scenarios} Scenarios Generated & Saved To: {file_path}")
    print(37*"="+"\n")

if __name__ == "__main__":
    generate_scenarios(output_dir=TARGET, num_scenarios=30000)