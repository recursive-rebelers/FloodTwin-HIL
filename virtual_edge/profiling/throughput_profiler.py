from dataclasses import dataclass

@dataclass
class ThroughputProfiler:

    count: int = 0
    elapsed_s: float = 0.0

    def update(self, count_delta: int, elapsed_delta_s: float) -> None:
        if count_delta < 0:
            raise ValueError("count_delta must be non-negative")
        if elapsed_delta_s < 0:
            raise ValueError("elapsed_delta_s must be non-negative")

        self.count += count_delta
        self.elapsed_s += elapsed_delta_s

    def reset(self) -> None:
        self.count = 0
        self.elapsed_s = 0.0

    def summary(self) -> dict:
        throughput = (self.count / self.elapsed_s if self.elapsed_s > 1e-12 else 0.0)

        return {
            "count": self.count,
            "elapsed_s": self.elapsed_s,
            "throughput_sps": throughput,
        }