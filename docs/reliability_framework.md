# Sensor Reliability Framework

**Definition:** Reliability ($R$) is defined as the probability that a sensor's measurement accurately represents the true depth of the environment under current physical conditions. 

**Scale:** $R \in [0.0, 1.0]$
* `0.0`: Completely unusable (e.g., LiDAR in heavy mud)
* `0.5`: Highly uncertain (e.g., Ultrasonic in shallow turbulent water)
* `1.0`: Perfect theoretical reliability

**Physics-Based Modeling Strategy:**
1. **LiDAR:** mode is driven by the Beer-Lambert Law for optical attenuation and Mie scattering (driven by NTU/Turbidity).
2. **Ultrasonic:** modeled around acoustic impedance mismatch. Reliability drops when surface reflection probability exceeds bottom reflection probability (multipath fading).
3. **Radar:** modeled using Signal-to-Clutter Ratio (SCR). mmWave penetrates water better, hence its reliability decays at a logarithmically slower rate than optical sensors.