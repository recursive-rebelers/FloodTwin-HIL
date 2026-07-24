# FloodTwin-HIL Failure Taxonomy

To rigorously evaluate the resilience of the perception ecosystem, we distinguish between **Environmental Failures** (e.g., mud, turbidity, rain) and **System Failures** (hardware and software malfunctions). Most multimodal studies exclusively test environmental conditions; this framework stresses both.

## Fault Categories

*   **Category A: Sensor Faults** 
    Hardware-level failures such as packet drops (dropouts) and transient electrical interference (noise).
*   **Category B: Communication Faults**
    Network-level failures such as latency and packet transmission delays across the vehicle's CAN bus or Ethernet backbone.
*   **Category C: Synchronization Faults**
    Software-level failures where multi-sensor calibration or timestamping drifts, leading to frame misalignment.
*   **Category D: Compound Faults**
    Worst-case deployment scenarios where environmental degradation occurs simultaneously with multiple system failures.

## Taxonomy Matrix

| Fault Type | Severity Levels | Expected Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **LiDAR Dropout** | High (50%-80%) | Missing Depth Data | Adaptive reliance on Radar/Ultrasonic |
| **Radar Dropout** | Medium (30%) | Reduced Robustness | Fallback to optical if NTU permits |
| **Sensor Noise** | Mild to Severe ($\sigma=1 \to 5$) | Estimation Variance | Reliability-weighted Opinion Pooling |
| **IMU Drift** | Medium (Accumulating) | Motion Estimation Error | Calibration & reset routines |
| **Comm Delay** | High (200 ms) | Late Decisions / Outdated Info | Predictive state estimation |
| **Sync Error** | High (10 Frames) | Fusion Misalignment | Temporal buffering |
| **Compound** | Extreme | System Stress Test | Core architectural resilience |