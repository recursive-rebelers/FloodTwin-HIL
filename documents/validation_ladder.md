# FloodTwin-HIL Validation Ladder

To bridge the gap between pure algorithmic simulation and practical deployment, FloodTwin-HIL follows a structured **6-Level Validation Ladder**. Each level progressively increases the realism and evaluation depth of the system, moving from software execution toward edge-oriented deployment.

* **Level 1: Software-in-the-Loop (SiL)**
    Basic execution of the perception and estimation pipeline on simulated/sanitized data to verify computational functionality and pipeline stability.

* **Level 2: Physics-Aware Digital Twin**
    Introduction of environment-dependent sensor degradation, including effects such as turbidity, acoustic interference, water interaction, and scene conditions, to emulate realistic sensing behavior.

* **Level 3: Reliability-Aware Fusion**
    Evaluation of dynamic sensor trust and adaptive fusion, allowing the system to adjust the contribution of individual sensors according to their estimated reliability and environmental conditions.

* **Level 4: Fault-Aware Evaluation**
    Stress-testing of the system under simulated sensor faults and degraded operating conditions, including sensor dropout, noise, delay, synchronization errors, and compound fault scenarios.

* **Level 5: Benchmark & Validation Assessment**
    Large-scale evaluation across a structured scenario set using comparative benchmarking, ablation studies, statistical analysis, and validation-evidence assessment to quantify the performance and consistency of the complete framework.

* **Level 6: Virtual Edge Deployment**
    Execution of the validated FloodTwin-HIL pipeline in a virtual edge-computing environment to evaluate deployment-oriented behavior, computational execution, and suitability for resource-constrained edge platforms before physical hardware deployment.