import numpy as np
import pandas as pd

class SyncInjector:

    def __init__(self):        
        self.levels = {
            "none": 0,
            "low": 1,           # 1 frame offset
            "medium": 5,    # 5 frames offset
            "high": 10,       # 10 frames offset
            "extreme": 20  # 20 frames offset
        }

    def get_offset(self, severity="medium"):        
        if severity not in self.levels:
            raise ValueError(f"Severity must be one of {list(self.levels.keys())}")
        return self.levels[severity]

    def inject(self, data_array, severity="medium", direction="backward"):        
        offset = self.get_offset(severity)
        data = np.asarray(data_array, dtype=float).copy()

        if offset == 0 or len(data) == 0:
            return data
        if direction not in {"backward", "forward"}:
            raise ValueError("direction must be either 'backward' or 'forward'")
        shifted = np.empty_like(data)

        if direction == "backward":
            shifted[:offset] = data[0]
            shifted[offset:] = data[:-offset]
        else:
            shifted[:-offset] = data[offset:]
            shifted[-offset:] = data[-1]
        return shifted

    def apply_to_dataframe(self, df, sensor_columns, severity="medium",
        direction="backward"):        
        df_faulty = df.copy()
        for col in sensor_columns:
            if col in df_faulty.columns:
                df_faulty[col] = self.inject(df_faulty[col].to_numpy(), severity=severity, direction=direction)
        return df_faulty

    def metadata(self):        
        return {
            "fault_type": "synchronization_error",
            "model": "Fixed frame-offset misalignment",
            "levels": self.levels
        }