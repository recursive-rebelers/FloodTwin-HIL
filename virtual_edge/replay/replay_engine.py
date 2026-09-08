from dataclasses import dataclass
from typing import Any, Iterator, Optional
from collections.abc import Mapping
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from virtual_edge.replay.dataset_loader import load_dataset
from virtual_edge.replay.scenario_loader import canonicalize_scenarios, filter_scenarios, load_scenario_registry

@dataclass(slots=True)
class ReplaySource:

    mode: str
    frame: pd.DataFrame
    original_count: int
    filtered_count: int
    loaded_count: int

def load_source(mode: str, registry_path: Path, dataset_path: Path, max_scenarios: Optional[int] = None, 
    filters: Optional[Mapping[str, Any]] = None) -> ReplaySource:

    mode = str(mode).strip().lower()
    if mode == "registry":
        frame = canonicalize_scenarios(load_scenario_registry(registry_path))
    elif mode == "dataset":
        frame = canonicalize_scenarios(load_dataset(dataset_path))
    elif mode == "live":
        raise NotImplementedError("live mode is reserved for future physical Raspberry Pi experiments")
    else:
        raise ValueError("source mode must be registry, dataset, or live")

    result = filter_scenarios(frame, filters or {})
    frame = result.frame

    if max_scenarios is not None:
        if max_scenarios <= 0:
            raise ValueError("max_scenarios must be greater than 0")
        frame = frame.head(max_scenarios).reset_index(drop=True)

    return ReplaySource(
        mode=mode, frame=frame,
        original_count=result.original_count,
        filtered_count=result.filtered_count,
        loaded_count=len(frame))

def iter_rows(frame: pd.DataFrame) -> Iterator[dict]:
    yield from frame.to_dict(orient="records")