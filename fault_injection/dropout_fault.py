import numpy as np
import pandas as pd

class DropoutInjector:

    def __init__(self, random_seed=42):        
        self.seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.levels = {
            "none": 0.00,
            "low": 0.10,          # 10% missing frames
            "medium": 0.30,   # 30% missing frames
            "high": 0.50,        # 50% missing frames
            "extreme": 0.80   # 80% missing frames
        }

    def get_dropout_probability(self, severity="low"):        
        if severity not in self.levels:
            raise ValueError(f"Severity must be one of {list(self.levels.keys())}")
        return self.levels[severity]

    def inject(self, data_array, severity="low", return_mask=False):        
        dropout_prob = self.get_dropout_probability(severity)
        data = np.asarray(data_array, dtype=float).copy()
        
        if dropout_prob <= 0.0:
            mask = np.zeros(len(data), dtype=bool)
            return (data, mask) if return_mask else data

        dropout_mask = self.rng.random(len(data)) < dropout_prob
        data[dropout_mask] = np.nan
        return (data, dropout_mask) if return_mask else data

    def apply_to_dataframe(self, df, sensor_columns, severity="medium"):        
        df_faulty = df.copy()
        for col in sensor_columns:
            if col in df_faulty.columns:
                df_faulty[col] = self.inject(df_faulty[col].to_numpy(), severity=severity)
        return df_faulty

    def metadata(self):        
        return {
            "fault_type": "sensor_dropout",
            "model": "Bernoulli frame dropout",
            "seed": self.seed,
            "levels": self.levels
        }