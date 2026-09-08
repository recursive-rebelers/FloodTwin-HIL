from dataclasses import dataclass, field
from statistics import fmean
from typing import List
import numpy as np

@dataclass
class LatencyProfiler:

    samples_ms: List[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        self.samples_ms.append(float(value_ms))

    def reset(self) -> None:
        self.samples_ms.clear()

    def summary(self) -> dict:
        if not self.samples_ms:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        values = np.asarray(self.samples_ms, dtype=float)

        return {
            "count": len(values),
            "mean_ms": float(fmean(values)),
            "median_ms": float(np.median(values)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
            "std_ms": float(values.std()),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
        }