from dataclasses import dataclass, field
from statistics import mean, median
from typing import List
import psutil
import numpy as np

@dataclass
class CPUProfiler:

    samples: List[float] = field(default_factory=list)

    def __post_init__(self):
        psutil.cpu_percent(interval=None)

    def add(self) -> float:
        value = float(psutil.cpu_percent(interval=None))
        self.samples.append(value)
        return value

    def reset(self) -> None:
        self.samples.clear()

    def summary(self) -> dict:
        if not self.samples:
            return {
                "count": 0,
                "mean_percent": 0.0,
                "median_percent": 0.0,
                "min_percent": 0.0,
                "max_percent": 0.0,
                "std_percent": 0.0,
                "p95_percent": 0.0,
            }

        values = np.asarray(self.samples, dtype=float)

        return {
            "count": len(values),
            "mean_percent": float(values.mean()),
            "median_percent": float(np.median(values)),
            "min_percent": float(values.min()),
            "max_percent": float(values.max()),
            "std_percent": float(values.std()),
            "p95_percent": float(np.percentile(values, 95)),
        }