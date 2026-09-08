from pathlib import Path
from typing import Any, Optional
import pandas as pd
from collections.abc import Mapping
from dataclasses import dataclass

@dataclass(slots=True)
class FilterResult:

    frame: pd.DataFrame
    original_count: int
    filtered_count: int

ALIASES = {
    "road_type": ["road_type", "surface_type", "pavement_type"],
    "road_environment": ["road_environment", "environment", "traffic_zone"],
    "pothole_depth": ["pothole_depth", "true_depth", "depth_cm"],
    "water_depth": ["water_depth", "flood_depth", "water_cm"],
    "ntu": ["ntu", "true_ntu", "turbidity_ntu"],
    "vehicle_speed": ["vehicle_speed", "speed_kmh", "speed"]}

def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None

def load_scenario_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scenario registry not found: {path}")
    df = pd.read_csv(path)
    if "scenario_id" not in df.columns:
        df.insert(0, "scenario_id", range(1, len(df) + 1))
    return df

def canonicalize_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for canonical, candidates in ALIASES.items():
        if canonical in out.columns: continue
        src = _pick_column(out, candidates)

        if src is not None:
            out[canonical] = out[src]
        elif canonical == "road_environment":
            out[canonical] = "Urban"
        elif canonical == "road_type":
            out[canonical] = "Asphalt"
        else: out[canonical] = 0.0
    
    out["ntu"] = pd.to_numeric(out["ntu"], errors="coerce")
    out["water_depth"] = pd.to_numeric(out["water_depth"], errors="coerce")
    out["pothole_depth"] = pd.to_numeric(out["pothole_depth"], errors="coerce")
    out["vehicle_speed"] = pd.to_numeric(out["vehicle_speed"], errors="coerce")
    if "scene_complexity" not in out.columns: out["scene_complexity"] = 0.0
    out["scene_complexity"] = pd.to_numeric(out["scene_complexity"], errors="coerce").fillna(0.0)

    if "true_depth" not in out.columns:
        out["true_depth"] = out["pothole_depth"]
    if "true_ntu" not in out.columns:
        out["true_ntu"] = out["ntu"]
    if "weather" not in out.columns:
        out["weather"] = "Clear"
    if "lighting" not in out.columns:
        out["lighting"] = "Day"
    if "severity" not in out.columns:
        out["severity"] = "medium"
    return out

def filter_scenarios(df: pd.DataFrame, filters: Mapping[str, Any]) -> FilterResult:
    original_count = len(df)
    out = df.copy()

    if not filters:
        return FilterResult(
            frame=out.reset_index(drop=True),
            original_count=original_count,
            filtered_count=original_count)

    if "water_depth_min" in filters:
        out = out[out["water_depth"] >= float(filters["water_depth_min"])]
    if "water_depth_max" in filters:
        out = out[out["water_depth"] <= float(filters["water_depth_max"])]
    if "ntu_min" in filters:
        out = out[out["ntu"] >= float(filters["ntu_min"])]
    if "ntu_max" in filters:
        out = out[out["ntu"] <= float(filters["ntu_max"])]
    if "vehicle_speed_min" in filters:
        out = out[out["vehicle_speed"] >= float(filters["vehicle_speed_min"])]
    if "vehicle_speed_max" in filters:
        out = out[out["vehicle_speed"] <= float(filters["vehicle_speed_max"])]
    if "road_type" in filters:
        out = out[out["road_type"].fillna("").str.lower() == str(filters["road_type"]).lower()]
    if "road_environment" in filters:
        out = out[out["road_environment"].fillna("").str.lower() == str(filters["road_environment"]).lower()]
    if "severity" in filters:
        out = out[out["severity"].fillna("").str.lower() == str(filters["severity"]).lower()]
    if "weather" in filters:
        out = out[out["weather"].fillna("").str.lower() == str(filters["weather"]).lower()]
    if "lighting" in filters:
        out = out[out["lighting"].fillna("").str.lower() == str(filters["lighting"]).lower()]
    if "scene_complexity_min" in filters:
        out = out[out["scene_complexity"] >= float(filters["scene_complexity_min"])]
    if "scene_complexity_max" in filters:
        out = out[out["scene_complexity"] <= float(filters["scene_complexity_max"])]
    if "pothole_depth_min" in filters:
        out = out[out["pothole_depth"] >= float(filters["pothole_depth_min"])]
    if "pothole_depth_max" in filters:
        out = out[out["pothole_depth"] <= float(filters["pothole_depth_max"])]

    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No scenarios matched filters: {dict(filters)}")

    return FilterResult(
        frame=out,
        original_count=original_count,
        filtered_count=len(out))