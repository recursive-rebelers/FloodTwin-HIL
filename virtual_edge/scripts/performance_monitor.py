from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional
import os
import time
import psutil

def _read_text(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return None

def _parse_cpu_list(spec: str) -> int:
    total = 0
    for token in spec.split(","):
        token = token.strip()
        if not token: continue
        if "-" in token:
            start, end = token.split("-", 1)
            total += int(end) - int(start) + 1
        else:
            total += 1
    return total

def get_cgroup_cpu_quota_cores() -> Optional[float]:
    raw = _read_text(Path("/sys/fs/cgroup/cpu.max"))
    if not raw:
        return None
    parts = raw.split()
    if len(parts) != 2:
        return None
    quota_s, period_s = parts
    if quota_s == "max":
        return None
    quota = float(quota_s)
    period = float(period_s)
    return quota / period if period else None

def get_cgroup_cpuset_cores() -> Optional[int]:
    for candidate in (Path("/sys/fs/cgroup/cpuset.cpus.effective"), Path("/sys/fs/cgroup/cpuset.cpus")):
        raw = _read_text(candidate)
        if raw:
            try:
                return _parse_cpu_list(raw)
            except Exception:
                return None
    return None

def get_cgroup_memory_limit_gb() -> Optional[float]:
    raw = _read_text(Path("/sys/fs/cgroup/memory.max"))
    if not raw or raw == "max":
        return None
    try:
        return int(raw) / (1024 ** 3)
    except ValueError:
        return None

@dataclass(frozen=True, slots=True)
class ResourceSnapshot:

    timestamp_s: float
    label: str
    process_rss_mb: float
    process_vms_mb: float
    process_cpu_percent: float
    system_cpu_percent: float
    system_memory_percent: float
    cgroup_cpu_quota_cores: Optional[float]
    cgroup_cpuset_cores: Optional[int]
    cgroup_memory_limit_gb: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class ResourceMonitor:

    def __init__(self, label: str = "virtual_edge"):
        self.label = label
        self.process = psutil.Process(os.getpid())
        self._primed = False

    def _prime(self) -> None:
        if not self._primed:
            self.process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
            self._primed = True

    def snapshot(self, label: Optional[str] = None) -> ResourceSnapshot:
        self._prime()
        mem = self.process.memory_info()
        return ResourceSnapshot(
            timestamp_s=time.time(),
            label=label or self.label,
            process_rss_mb=mem.rss / (1024 ** 2),
            process_vms_mb=mem.vms / (1024 ** 2),
            process_cpu_percent=self.process.cpu_percent(interval=None),
            system_cpu_percent=psutil.cpu_percent(interval=None),
            system_memory_percent=psutil.virtual_memory().percent,
            cgroup_cpu_quota_cores=get_cgroup_cpu_quota_cores(),
            cgroup_cpuset_cores=get_cgroup_cpuset_cores(),
            cgroup_memory_limit_gb=get_cgroup_memory_limit_gb())

@contextmanager
def stage_timer() -> Iterator[dict[str, float | None]]:
    state: dict[str, float | None] = {"start_s": time.perf_counter(), "elapsed_ms": None}
    try: yield state
    finally: state["elapsed_ms"] = (time.perf_counter() - state["start_s"]) * 1000.0