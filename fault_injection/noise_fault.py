import numpy as np
import pandas as pd

class NoiseInjector:

    def __init__(self, random_seed=42):        
        self.seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.levels = {
            "none": 0.0,
            "mild": 1.0,          # sigma = 1 cm
            "moderate": 3.0,  # sigma = 3 cm
            "severe": 5.0       # sigma = 5 cm
        }

    def get_sigma(self, severity="moderate"):        
        if severity not in self.levels:
            raise ValueError(f"Severity must be one of {list(self.levels.keys())}")
        return self.levels[severity]

    def inject(self, data_array, severity="moderate", clip_min=None, clip_max=None):        
        sigma = self.get_sigma(severity)
        data = np.asarray(data_array, dtype=float).copy()
        if sigma <= 0.0: return data

        noise = self.rng.normal(loc=0.0, scale=sigma, size=len(data))
        noisy = data + noise
        if clip_min is not None or clip_max is not None:
            noisy = np.clip(noisy, clip_min, clip_max)
        return noisy

    def apply_to_dataframe(self, df, sensor_columns, severity="moderate",
        clip_min=None, clip_max=None):                           
        df_faulty = df.copy()
        for col in sensor_columns:
            if col in df_faulty.columns:
                df_faulty[col] = self.inject(df_faulty[col].to_numpy(),
                    severity=severity, clip_min=clip_min, clip_max=clip_max)
        return df_faulty

    def metadata(self):        
        return {
            "fault_type": "gaussian_noise",
            "model": "Zero-mean additive Gaussian noise",
            "seed": self.seed,
            "levels": self.levels
        }