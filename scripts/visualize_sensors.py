import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup and Data Loading
os.makedirs('figures', exist_ok=True)
df = pd.read_csv('datasets/dataset_v1.csv')

# Calculate Absolute Errors for each sensor
df['lidar_error'] = abs(df['lidar'] - df['true_depth'])
df['ultrasonic_error'] = abs(df['ultrasonic'] - df['true_depth'])
df['radar_error'] = abs(df['radar'] - df['true_depth'])

sns.set_theme(style="whitegrid")

# 2. Plot 1: LiDAR Error vs NTU
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='ntu', y='lidar_error', alpha=0.5, color='red')
sns.regplot(data=df, x='ntu', y='lidar_error', scatter=False, color='darkred')
plt.title('LiDAR Measurement Error vs. Water Turbidity (NTU)')
plt.xlabel('Turbidity (NTU)')
plt.ylabel('Absolute Error (cm)')
plt.tight_layout()
plt.savefig('figures/figure_lidar.png', dpi=300)
plt.close()

# 3. Plot 2: Ultrasonic Error vs Water Depth
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='water_depth', y='ultrasonic_error', alpha=0.5, color='blue')
sns.regplot(data=df, x='water_depth', y='ultrasonic_error', scatter=False, color='darkblue')
plt.title('Ultrasonic Measurement Error vs. Water Depth')
plt.xlabel('Water Depth (cm)')
plt.ylabel('Absolute Error (cm)')
plt.tight_layout()
plt.savefig('figures/figure_ultrasonic.png', dpi=300)
plt.close()

# 4. Plot 3: Radar Stability vs Flood Depth
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='water_depth', y='radar_error', alpha=0.5, color='green')
sns.regplot(data=df, x='water_depth', y='radar_error', scatter=False, color='darkgreen')
plt.title('Radar Measurement Stability vs. Flood Depth')
plt.xlabel('Water Depth (cm)')
plt.ylabel('Absolute Error (cm)')
plt.ylim(0, df['lidar_error'].max()) # Match scale to show it is more stable
plt.tight_layout()
plt.savefig('figures/figure_radar.png', dpi=300)
plt.close()

print("✅ Scientific visualizations successfully generated in the /figures folder.")