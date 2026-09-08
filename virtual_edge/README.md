# FloodTwin-HIL Virtual Edge

Research Grade Execution Layer For FloodTwin-HIL

**This Package Is Designed To Run The Complete Pipeline :**
Digital Twin → Fault Injection → Reliability Engine → Reliability Fusion →
Adaptive Bayesian Fusion → Uncertainty Quantification → Hazard Assessment →
Profiling → Logging

### Experiment Modes

- `Baseline Registry`: *python3 virtual_edge/scripts/experiment_runner.py --profile baseline_registry*
- `Baseline Dataset`: *python3 virtual_edge/scripts/experiment_runner.py --profile baseline_dataset*
- `Flood Severity`: *python3 virtual_edge/scripts/experiment_runner.py --profile flood_severity*
- `Muddy Water`: *python3 virtual_edge/scripts/experiment_runner.py --profile muddy_water*
- `High Speed`: *python3 virtual_edge/scripts/experiment_runner.py --profile high_speed*
- `Complex Environment`: *python3 virtual_edge/scripts/experiment_runner.py --profile complex_environment*
- `Sensor Failure`: *python3 virtual_edge/scripts/experiment_runner.py --profile sensor_failure*
- `Compound Fault`: *python3 virtual_edge/scripts/experiment_runner.py --profile compound_fault*
- `Uncertainty Analysis`: *python3 virtual_edge/scripts/experiment_runner.py --profile uncertainty_analysis*
- `Stress Test`: *python3 virtual_edge/scripts/experiment_runner.py --profile stress_test*

### Main Entry Points

- `Environment Check`: *python3 virtual_edge/scripts/check_environment.py*
- `Virtual Edge Runner`: *python3 virtual_edge/scripts/virtual_edge_runner.py*
- `Experiment Runner`: *python3 virtual_edge/scripts/experiment_runner.py*
- `Asset Generator`: *python3 virtual_edge/scripts/generate_assets.py*