from dataclasses import dataclass, field
from statistics import fmean
from typing import List
import numpy as np
import psutil

@dataclass
class MemoryProfiler:

    process: psutil.Process = field(default_factory=psutil.Process)
    samples_mb: List[float] = field(default_factory=list)

    def add_process(self) -> float:
        rss_mb = self.process.memory_info().rss / (1024 ** 2)
        self.samples_mb.append(rss_mb)
        return rss_mb

    def reset(self) -> None:
        self.samples_mb.clear()

    def summary(self) -> dict:
        if not self.samples_mb:
            return {
                "count": 0,
                "mean_mb": 0.0,
                "median_mb": 0.0,
                "min_mb": 0.0,
                "max_mb": 0.0,
                "std_mb": 0.0,
                "p95_mb": 0.0,
            }

        values = np.asarray(self.samples_mb, dtype=float)

        return {
            "count": len(values),
            "mean_mb": float(fmean(values)),
            "median_mb": float(np.median(values)),
            "min_mb": float(values.min()),
            "max_mb": float(values.max()),
            "std_mb": float(values.std()),
            "p95_mb": float(np.percentile(values, 95)),
        }