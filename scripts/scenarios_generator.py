import csv
import random
from pathlib import Path
from typing import Dict, Tuple

TARGET = Path(__file__).resolve().parent.parent / "datasets"
SEED = 42
_rng = random.Random(SEED)

# ------------------------------------------------------------------
# Scenario categorical domains
# ------------------------------------------------------------------
ROAD_TYPES = ["Asphalt", "Concrete", "Gravel", "Dirt", "Mud"]
ROAD_ENVIRONMENTS = ["Urban", "Rural"]

WEATHER_OPTIONS = ["Clear", "Cloudy", "Drizzle", "Rainy", "Foggy", "Mist", "Thunderstorm"]
LIGHTING_OPTIONS = ["Dawn", "Daylight", "Dusk", "Night_Streetlights", "Night_Dark"]

ROAD_TYPE_WEIGHTS = [45, 25, 12, 10, 8]
ROAD_ENVIRONMENT_WEIGHTS = [70, 30]
WEATHER_WEIGHTS = [28, 20, 14, 16, 6, 8, 8]
LIGHTING_WEIGHTS = [8, 40, 10, 28, 14]

# ------------------------------------------------------------------
# Physical / Engineering scenario domain
# ------------------------------------------------------------------
MIN_POTHOLE_DEPTH_CM = 3.0
MAX_POTHOLE_DEPTH_CM = 20.0

WATER_PRESENCE_PROBABILITY = {
    "Clear": 0.12,
    "Cloudy": 0.20,
    "Drizzle": 0.42,
    "Rainy": 0.72,
    "Foggy": 0.28,
    "Mist": 0.35,
    "Thunderstorm": 0.82,
}

WATER_FILL_RATIO_BY_WEATHER: Dict[str, Tuple[float, float, float]] = {
    "Clear": (0.03, 0.30, 0.12),
    "Cloudy": (0.04, 0.40, 0.18),
    "Drizzle": (0.08, 0.50, 0.24),
    "Rainy": (0.12, 0.72, 0.38),
    "Foggy": (0.04, 0.40, 0.16),
    "Mist": (0.06, 0.46, 0.20),
    "Thunderstorm": (0.18, 0.85, 0.48),
}

MIN_VEHICLE_SPEED_KMH = {"Urban": 15.0, "Rural": 30.0}
MAX_VEHICLE_SPEED_KMH = {"Urban": 50.0, "Rural": 80.0}
SPEED_MODE_KMH = {"Urban": 35.0, "Rural": 60.0}

WEATHER_NTU_RANGES: Dict[str, Tuple[float, float, float]] = {
    "Clear": (0.0, 100.0, 20.0),
    "Cloudy": (20.0, 160.0, 75.0),
    "Drizzle": (50.0, 220.0, 125.0),
    "Rainy": (90.0, 350.0, 200.0),
    "Foggy": (70.0, 280.0, 155.0),
    "Mist": (50.0, 220.0, 120.0),
    "Thunderstorm": (180.0, 500.0, 360.0),
}

TEMPERATURE_RANGES_C: Dict[str, Tuple[float, float, float]] = {
    "Clear": (16.0, 36.0, 27.0),
    "Cloudy": (15.0, 32.0, 24.0),
    "Drizzle": (15.0, 29.0, 22.0),
    "Rainy": (14.0, 28.0, 21.0),
    "Foggy": (13.0, 27.0, 20.0),
    "Mist": (14.0, 29.0, 21.0),
    "Thunderstorm": (16.0, 30.0, 23.0),
}

HUMIDITY_RANGES_PCT: Dict[str, Tuple[float, float, float]] = {
    "Clear": (30.0, 75.0, 52.0),
    "Cloudy": (45.0, 88.0, 68.0),
    "Drizzle": (65.0, 95.0, 82.0),
    "Rainy": (72.0, 100.0, 90.0),
    "Foggy": (80.0, 100.0, 94.0),
    "Mist": (72.0, 100.0, 88.0),
    "Thunderstorm": (70.0, 100.0, 89.0),
}

LIGHTING_LUX_RANGES = {
    "Daylight": (8_000.0, 60_000.0, 25_000.0),
    "Dawn": (300.0, 8_000.0, 2_000.0),
    "Dusk": (150.0, 6_000.0, 1_400.0),
    "Night_Streetlights": (5.0, 250.0, 60.0),
    "Night_Dark": (0.1, 5.0, 0.8),
}

WATER_CONDUCTIVITY_BY_WEATHER = {
    "Clear": (0.55, 1.10),
    "Cloudy": (0.60, 1.15),
    "Drizzle": (0.70, 1.25),
    "Rainy": (0.75, 1.40),
    "Foggy": (0.65, 1.20),
    "Mist": (0.65, 1.25),
    "Thunderstorm": (0.80, 1.55),
}

ROAD_ROUGHNESS = {
    "Asphalt": 1.00,
    "Concrete": 0.90,
    "Gravel": 1.55,
    "Dirt": 2.00,
    "Mud": 2.35,
}

ENVIRONMENT_ROUGHNESS = {
    "Urban": 1.00,
    "Rural": 1.15,
}

SURFACE_REFLECTANCE = {
    "Asphalt": 0.18,
    "Concrete": 0.45,
    "Gravel": 0.32,
    "Dirt": 0.24,
    "Mud": 0.12,
}

RADAR_RETURN_STRENGTH = {
    "Asphalt": 0.80,
    "Concrete": 0.95,
    "Gravel": 0.90,
    "Dirt": 0.72,
    "Mud": 0.62,
}

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(float(value), high))

def _weighted_triangular(low: float, high: float, mode: float) -> float:
    mode = _clip(mode, low, high)
    return _rng.triangular(low, high, mode)

# ------------------------------------------------------------------
# Latent environmental variables
# ------------------------------------------------------------------
def generate_ntu(weather: str) -> float:
    low, high, mode = WEATHER_NTU_RANGES[weather]
    return round(_weighted_triangular(low, high, mode), 2)

def generate_temperature(weather: str) -> float:
    low, high, mode = TEMPERATURE_RANGES_C[weather]
    return round(_weighted_triangular(low, high, mode), 2)

def generate_humidity(weather: str) -> float:
    low, high, mode = HUMIDITY_RANGES_PCT[weather]
    return round(_weighted_triangular(low, high, mode), 2)

def generate_ambient_lux(lighting: str) -> float:
    low, high, mode = LIGHTING_LUX_RANGES[lighting]
    return round(_weighted_triangular(low, high, mode), 3)

def generate_water_depth_cm(pothole_depth_cm: float, weather: str) -> float:
    presence_probability = WATER_PRESENCE_PROBABILITY[weather]
    if _rng.random() > presence_probability: return 0.0

    low, high, mode = WATER_FILL_RATIO_BY_WEATHER[weather]
    fill_ratio = _weighted_triangular(low, high, mode)
    depth = pothole_depth_cm * fill_ratio

    if depth < 0.20: return 0.0
    return round(_clip(depth, 0.0, pothole_depth_cm), 2)

def generate_speed(environment: str, weather: str) -> float:
    low = MIN_VEHICLE_SPEED_KMH[environment]
    high = MAX_VEHICLE_SPEED_KMH[environment]
    mode = SPEED_MODE_KMH[environment]

    if weather in {"Rainy", "Foggy", "Mist"}:
        high *= 0.82
        mode *= 0.88
    elif weather == "Drizzle":
        high *= 0.92
        mode *= 0.96
    elif weather == "Thunderstorm":
        high *= 0.65
        mode *= 0.72

    mode = _clip(mode, low, high)
    return round(_weighted_triangular(low, high, mode), 2)

def generate_pothole_depth_cm() -> float:
    return round(_weighted_triangular(MIN_POTHOLE_DEPTH_CM, MAX_POTHOLE_DEPTH_CM, 10.0), 2)

def generate_water_conductivity_factor(weather: str) -> float:
    low, high = WATER_CONDUCTIVITY_BY_WEATHER[weather]
    return round(_rng.uniform(low, high), 3)

# ------------------------------------------------------------------
# Observable/context priors consumed by the upgraded sensor twins
# ------------------------------------------------------------------
def road_roughness_factor(road_type: str, road_environment: str) -> float:
    return round(ROAD_ROUGHNESS[road_type] * ENVIRONMENT_ROUGHNESS[road_environment], 4)

def surface_reflectance_prior(road_type: str) -> float:
    return round(SURFACE_REFLECTANCE[road_type], 4)

def radar_return_strength_prior(road_type: str) -> float:
    return round(RADAR_RETURN_STRENGTH[road_type], 4)

# ------------------------------------------------------------------
# Evaluation-only severity annotation
# ------------------------------------------------------------------
def compute_severity(
    depth_cm: float, water_cm: float, ntu: float,
    environment: str, speed_kmh: float) -> str:
    depth_score = _clip(depth_cm / MAX_POTHOLE_DEPTH_CM, 0.0, 1.0)
    water_score = _clip(water_cm / MAX_POTHOLE_DEPTH_CM, 0.0, 1.0)
    ntu_score = _clip(ntu / 500.0, 0.0, 1.0)
    speed_score = _clip(speed_kmh / MAX_VEHICLE_SPEED_KMH[environment], 0.0, 1.0)

    score = (0.40 * depth_score
        + 0.20 * water_score
        + 0.10 * ntu_score
        + 0.30 * speed_score)

    if score < 0.30: return "Low"
    elif score < 0.48: return "Moderate"
    elif score < 0.72: return "Severe"
    else: return "Extreme"

def validate_configuration(num_scenarios: int) -> None:
    if int(num_scenarios) <= 0: raise ValueError("num_scenarios must be a positive integer")
    if MIN_POTHOLE_DEPTH_CM <= 0 or MAX_POTHOLE_DEPTH_CM <= MIN_POTHOLE_DEPTH_CM:
        raise ValueError("Invalid pothole depth limits")

    for weather in WEATHER_OPTIONS:
        if not (0.0 <= WATER_PRESENCE_PROBABILITY[weather] <= 1.0):
            raise ValueError(f"Invalid water-presence probability for {weather}.")

# ------------------------------------------------------------------
# Main registry generation
# ------------------------------------------------------------------
def generate_scenarios(output_dir=".", num_scenarios=30000):
    validate_configuration(num_scenarios)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "scenario_registry.csv"
    rows = []

    for i in range(1, num_scenarios + 1):
        scenario_id = f"SC{i:05d}"
        road_type = _rng.choices(ROAD_TYPES, weights=ROAD_TYPE_WEIGHTS, k=1)[0]
        road_environment = _rng.choices(ROAD_ENVIRONMENTS, weights=ROAD_ENVIRONMENT_WEIGHTS, k=1)[0]
        weather = _rng.choices(WEATHER_OPTIONS, weights=WEATHER_WEIGHTS, k=1)[0]
        lighting = _rng.choices(LIGHTING_OPTIONS, weights=LIGHTING_WEIGHTS, k=1)[0]

        pothole_depth = generate_pothole_depth_cm()
        water_depth = generate_water_depth_cm(pothole_depth, weather)
        ntu = generate_ntu(weather)
        vehicle_speed = generate_speed(road_environment, weather)

        temperature_c = generate_temperature(weather)
        relative_humidity_pct = generate_humidity(weather)
        ambient_lux = generate_ambient_lux(lighting)
        water_conductivity_factor = generate_water_conductivity_factor(weather)

        severity = compute_severity(
            pothole_depth, water_depth, ntu, road_environment, vehicle_speed)

        rows.append([
            scenario_id, road_environment, road_type,
            pothole_depth, water_depth, ntu, weather,
            lighting, vehicle_speed, severity,

            temperature_c, relative_humidity_pct,
            ambient_lux, water_conductivity_factor,

            road_roughness_factor(road_type, road_environment),
            surface_reflectance_prior(road_type),
            radar_return_strength_prior(road_type),
        ])

    headers = [
        "scenario_id", "road_environment", "road_type",
        "pothole_depth", "water_depth", "ntu", "weather",
        "lighting", "vehicle_speed", "severity",
        "temperature_c", "relative_humidity_pct",
        "ambient_lux", "water_conductivity_factor",
        "road_roughness_factor", "surface_reflectance_prior",
        "radar_return_strength_prior"]

    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("Physics-Oriented Digital Twin Scenario Registry")
    print("=" * 60)
    print(f"Scenarios Generated : {num_scenarios:,}")
    print(f"Random Seed         : {SEED}")
    print(f"Saved To            : {file_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    generate_scenarios(output_dir=TARGET, num_scenarios=30000)