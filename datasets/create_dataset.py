import csv
import os

def initialize_dataset(output_dir="."):
    """
    Initializes an empty dataset structure in the designated directory path.
    """
    # Ensure the target directory exists
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'dataset_v1.csv')

    columns = [
        'scenario_id',
        'true_depth',
        'lidar',
        'ultrasonic',
        'radar',
        'imu',
        'ntu',
        'water_contact'
    ]
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        
    print(f"Successfully Initialized Structure At: {file_path}")

if __name__ == "__main__":
    target_path = "C:/B.Tech-IT 2023-2027/PROJECT_INTERVIEW/FloodTwin-HIL/datasets" 
    
    initialize_dataset(output_dir=target_path)