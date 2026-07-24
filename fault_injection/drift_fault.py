import numpy as np
import pandas as pd

class DriftInjector:

    def __init__(self, random_seed=42):        
        self.seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.rates = {
            "none": 0.0,
            "low": 0.002,          # 2.0 deg after 1000 frames
            "medium": 0.005,   # 5.0 deg after 1000 frames
            "high": 0.010,        # 10.0 deg after 1000 frames
            "extreme": 0.015   # 15.0 deg after 1000 frames
        }

    def get_drift_rate(self, severity="medium"):        
        if severity not in self.rates:
            raise ValueError(f"Severity must be one of {list(self.rates.keys())}")
        return self.rates[severity]

    def inject(self, data_array, severity="medium", direction=1.0, walk_sigma=0.0, start_bias=0.0):               
        drift_rate = self.get_drift_rate(severity)
        data = np.asarray(data_array, dtype=float).copy()

        if len(data) == 0:
            return data
        if drift_rate == 0.0 and walk_sigma <= 0.0 and start_bias == 0.0:
            return data

        frames = np.arange(len(data), dtype=float)
        linear_bias = start_bias + (direction * drift_rate * frames)

        if walk_sigma > 0.0:
            random_walk = self.rng.normal(loc=0.0, scale=walk_sigma, size=len(data)).cumsum()
            linear_bias = linear_bias + random_walk
        return data + linear_bias

    def apply_to_dataframe(self, df, sensor_columns, severity="medium", direction=1.0, walk_sigma=0.0, start_bias=0.0):
        df_faulty = df.copy()
        for col in sensor_columns:
            if col in df_faulty.columns:
                df_faulty[col] = self.inject(df_faulty[col].to_numpy(), 
                severity=severity, direction=direction, walk_sigma=walk_sigma, start_bias=start_bias)
        return df_faulty

    def build_experiment_dataframe(self, frames=1000, true_pitch=0.0,
        severities=("low", "medium", "high", "extreme"), direction=1.0):                                   
        frame_idx = np.arange(frames)
        base_signal = np.full(frames, true_pitch, dtype=float)
        payload = { "frame": frame_idx, "true_pitch": base_signal }

        for severity in severities:
            payload[f"drift_{severity}"] = self.inject(base_signal, severity=severity, direction=direction)
        return pd.DataFrame(payload)

    def metadata(self):        
        return {
            "fault_type": "imu_drift",
            "model": "Linear accumulated bias with optional random walk",
            "seed": self.seed,
            "rates_deg_per_frame": self.rates,
            "recommended_frames": {
                "primary": 1000,
                "stress_test": 2000
            }
        }