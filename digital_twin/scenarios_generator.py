import csv
import os
import random

def generate_scenarios(output_dir="."):
    # Ensure the target directory exists
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'scenario_registry.csv')
    
    scenarios = []
    
    # Configuration options for categorical variations
    road_types = ['Asphalt', 'Concrete', 'Gravel', 'Dirt', 'Mud','Urban','Rural']
    weather_options = ['Clear', 'Cloudy', 'Rainy', 'Drizzle','Foggy','Mist','Thunderstorm']
    lighting_options = ['Dawn','Daylight', 'Dusk', 'Night_Streetlights', 'Night_Dark']
    
    for i in range(1, 51):
        scenario_id = f"SC{i:03d}"
        road_type = random.choice(road_types)
        pothole_depth = random.randint(5, 20)
        water_depth = random.randint(0, 10)
        ntu = random.randint(0, 500)
        weather = random.choice(weather_options)
        lighting = random.choice(lighting_options)
        vehicle_speed = round(random.uniform(0.0, 8.0), 2)
            
        scenarios.append([
            scenario_id, road_type, pothole_depth, water_depth, 
            ntu, weather, lighting, vehicle_speed
        ])
        
    # Updated headers
    headers = [
        'scenario_id', 'road_type', 'pothole_depth', 'water_depth', 
        'ntu', 'weather', 'lighting', 'vehicle_speed'
    ]
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(scenarios)
        
    print(f"Successfully Generated 50 Gcenarios And Saved To: {file_path}")

if __name__ == "__main__":
    target_path = "C:/B.Tech-IT 2023-2027/PROJECT_INTERVIEW/FloodTwin-HIL/scenarios" 
    
    generate_scenarios(output_dir=target_path)