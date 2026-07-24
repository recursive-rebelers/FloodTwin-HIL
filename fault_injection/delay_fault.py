import numpy as np

class DelayInjector:

    def __init__(self, fps=20):        
        if fps <= 0: raise ValueError("fps must be greater than 0")
        self.fps = fps
        self.ms_per_frame = 1000.0 / fps
        self.levels = {
            "normal": 0,        # 0 ms
            "low": 50,           # 50 ms
            "medium": 100,  # 100 ms
            "high": 200        # 200 ms
        }

    def get_delay_ms(self, severity="medium"):        
        if severity not in self.levels:
            raise ValueError(f"Severity must be one of {list(self.levels.keys())}")
        return self.levels[severity]

    def get_frame_shift(self, severity="medium"):
        delay_ms = self.get_delay_ms(severity)
        return int(round(delay_ms / self.ms_per_frame))

    def inject(self, data_array, severity="medium", hold_first=True):
        delay_ms = self.get_delay_ms(severity)
        data = np.asarray(data_array, dtype=float).copy()
        if delay_ms == 0: return data

        frame_shift = max(1, self.get_frame_shift(severity))
        delayed = np.empty_like(data)

        if hold_first:
            delayed[:frame_shift] = data[0]
        else:
            delayed[:frame_shift] = np.nan

        delayed[frame_shift:] = data[:-frame_shift]
        return delayed

    def metadata(self):
        return {
            "fault_type": "communication_delay",
            "model": "Fixed-latency causal frame shift",
            "fps": self.fps,
            "ms_per_frame": self.ms_per_frame,
            "levels": self.levels
        }