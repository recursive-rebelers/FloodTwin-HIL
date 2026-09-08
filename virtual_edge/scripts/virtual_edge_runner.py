import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import sys
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from virtual_edge.adapters.digital_twin_adapter import DigitalTwinAdapter
from virtual_edge.adapters.fault_adapter import FaultAdapter
from virtual_edge.adapters.fusion_adapter import FusionAdapter
from virtual_edge.adapters.hazard_adapter import HazardAdapter
from virtual_edge.adapters.reliability_adapter import ReliabilityAdapter
from virtual_edge.replay.random_seed import seed_everything
from virtual_edge.replay.replay_engine import load_source
from virtual_edge.scripts.config import VirtualEdgeConfig
from virtual_edge.scripts.metrics_logger import CSVLogger, JSONLogger
from virtual_edge.scripts.performance_monitor import ResourceMonitor, stage_timer

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

def _jsonify(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)

def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return _jsonify(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value

def _row_for_csv(payload: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for field in fieldnames:
        row[field] = _csv_value(payload.get(field, ""))
    return row

def _first_present(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            value = data[key]
            if isinstance(value, float) and np.isnan(value):
                continue
            return value
    return default

def _load_experiment_profile(repo_root: Path, profile_name: str) -> dict:
    profiles_path = repo_root / "virtual_edge" / "config" / "experiment_profiles.yaml"
    profiles = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
    experiments = profiles.get("experiments", {})
    if profile_name not in experiments:
        raise KeyError(f"Unknown experiment profile: {profile_name}")
    return experiments[profile_name]

def _scene_filters(experiment_profile: dict) -> dict:
    return experiment_profile.get("filters", {}) or {}

def _normalize_dataset_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(packet or {})

    p["road_type"] = _first_present(p, "road_type", "surface_type", default="Asphalt")
    p["road_environment"] = _first_present(p, "road_environment", "environment", default="Urban")
    p["weather"] = _first_present(p, "weather", default="Clear")
    p["lighting"] = _first_present(p, "lighting", default="Day")
    p["vehicle_speed"] = _safe_float(_first_present(p, "vehicle_speed", "speed_kmh", "speed", default=0.0), 0.0)

    p["true_depth"] = _safe_float(_first_present(p, "true_depth", "pothole_depth", "depth_cm", default=0.0), 0.0)
    p["pothole_depth"] = _safe_float(_first_present(
        p, "pothole_depth", "true_depth", "depth_cm", default=p["true_depth"]), p["true_depth"])
    p["water_depth"] = _safe_float(_first_present(p, "water_depth", "flood_depth", "water_cm", default=0.0), 0.0)
    p["true_ntu"] = _safe_float(_first_present(
        p, "true_ntu", "ntu", "turbidity_ntu", default=0.0), 0.0)
    p["ntu"] = _safe_float(_first_present(
        p, "ntu_observed", "ntu", "turbidity_ntu", default=np.nan), np.nan)
    p["ntu_observed"] = p["ntu"]
    p["ntu_available"] = int(np.isfinite(p["ntu_observed"]))
    p["scene_complexity"] = _safe_float(_first_present(p, "scene_complexity", "context_complexity", default=0.0), 0.0)

    p["lidar"] = _safe_float(_first_present(
        p, "lidar", "lidar_distance", "lidar_reading", default=np.nan), np.nan)
    p["radar"] = _safe_float(_first_present(
        p, "radar", "radar_distance", "radar_reading", default=np.nan), np.nan)
    p["ultrasonic"] = _safe_float(_first_present(
        p, "ultrasonic", "ultrasonic_distance", "ultrasonic_reading", default=np.nan), np.nan)
    p["imu_pitch"] = _safe_float(_first_present(p, "imu_pitch", "pitch", default=0.0), 0.0)
    p["imu_roll"] = _safe_float(_first_present(p, "imu_roll", "roll", default=0.0), 0.0)
    p["imu_acceleration"] = _safe_float(_first_present(p, "imu_acceleration", "acceleration", default=0.0), 0.0)
    p["turbidity"] = _safe_float(_first_present(
        p, "turbidity", "ntu_observed", default=np.nan), np.nan)
    p["water_contact"] = _safe_int(_first_present(p, "water_contact", "water_present", default=0), 0)

    p["lidar_reliability_prior"] = _safe_float(_first_present(p, "lidar_reliability_prior", "R_lidar", "r_lidar", default=1.0), 1.0)
    p["radar_reliability_prior"] = _safe_float(_first_present(p, "radar_reliability_prior", "R_radar", "r_radar", default=1.0), 1.0)
    p["ultrasonic_reliability_prior"] = _safe_float(_first_present(
        p, "ultrasonic_reliability_prior", "R_ultrasonic", "r_ultrasonic", default=1.0), 1.0)
    p["imu_reliability_prior"] = _safe_float(_first_present(p, "imu_reliability_prior", "R_imu", "r_imu", default=1.0), 1.0)
    p["turbidity_reliability_prior"] = _safe_float(_first_present(p, "turbidity_reliability_prior", "R_turbidity", default=1.0), 1.0)
    p["water_contact_reliability_prior"] = _safe_float(_first_present(
        p, "water_contact_reliability_prior", "R_water", "r_water", default=1.0), 1.0)

    p["lidar_noise_sigma"] = _safe_float(_first_present(p, "lidar_noise_sigma", "lidar_sigma", default=1.0), 1.0)
    p["radar_snr"] = _safe_float(_first_present(p, "radar_snr", "SNR", "snr", default=20.0), 20.0)
    p["radar_rcs"] = _safe_float(_first_present(p, "radar_rcs", "RCS", "rcs", default=1.0), 1.0)
    p["radar_clutter"] = _safe_float(_first_present(p, "radar_clutter", "clutter_probability", default=0.0), 0.0)
    p["ultrasonic_surface_echo"] = _safe_float(_first_present(p, "ultrasonic_surface_echo", "surface_echo", default=0.5), 0.5)
    p["ultrasonic_bottom_echo"] = _safe_float(_first_present(p, "ultrasonic_bottom_echo", "bottom_echo", default=0.5), 0.5)
    p["ultrasonic_bottom_confidence"] = _safe_float(_first_present(
        p, "ultrasonic_bottom_confidence", "bottom_confidence", default=0.5), 0.5)
    p["ultrasonic_echo_ambiguity"] = _safe_float(_first_present(
        p, "ultrasonic_echo_ambiguity", "echo_ambiguity", default=0.5), 0.5)
    p["ultrasonic_measurement_type"] = _first_present(
        p, "ultrasonic_measurement_type", "measurement_type", default="Bottom")
    p["ultrasonic_surface_sigma"] = _safe_float(_first_present(
        p, "ultrasonic_surface_sigma", "surface_sigma", default=1.0), 1.0)
    p["ultrasonic_bottom_sigma"] = _safe_float(_first_present(
        p, "ultrasonic_bottom_sigma", "bottom_sigma", default=1.0), 1.0)
    p["ultrasonic_sigma"] = _safe_float(_first_present(p, "ultrasonic_sigma", default=np.mean([p["ultrasonic_surface_sigma"],
    p["ultrasonic_bottom_sigma"]])), float(np.mean([p["ultrasonic_surface_sigma"], p["ultrasonic_bottom_sigma"]])))
    p["water_contact_sigma"] = _safe_float(_first_present(p, "water_contact_sigma", default=0.15), 0.15)
    p["imu_sigma"] = _safe_float(_first_present(p, "imu_sigma", default=1.0), 1.0)
    p["turbidity_sigma"] = _safe_float(_first_present(p, "turbidity_sigma", "turbidity_noise_sigma", default=5.0), 5.0)

    sensor_health = _first_present(p, "sensor_health", default=None)
    if not isinstance(sensor_health, dict):
        sensor_health = {
        "lidar": 1.0,
        "ultrasonic": 1.0,
        "radar": 1.0,
        "imu": 1.0,
        "turbidity": 1.0,
        "water_contact": 1.0 }  
    p["sensor_health"] = sensor_health
    sensor_availability = _first_present(p, "sensor_availability", default=None)

    if not isinstance(sensor_availability, dict):
        sensor_availability = {
            "lidar": 1.0,
            "ultrasonic": 1.0,
            "radar": 1.0,
            "imu": 1.0,
            "turbidity": 1.0,
            "water_contact": 1.0 }
    p["sensor_availability"] = sensor_availability
    return p

def _scenario_payload_to_common(packet: Dict[str, Any], source_mode: str) -> Dict[str, Any]:
    p = dict(packet or {})
    if source_mode == "dataset":
        p = _normalize_dataset_packet(p)
    return p

def _timing_stage_row(*,
    experiment_profile: str, source_mode: str, fault_profile: str, scenario_id: int,
    frame_idx: int, stage: str, elapsed_ms: float, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment_profile": experiment_profile,
        "source_mode": source_mode,
        "fault_profile": fault_profile,
        "scenario_id": scenario_id,
        "frame_idx": frame_idx,
        "stage": stage,
        "elapsed_ms": elapsed_ms,
        "timestamp_s": snapshot.get("timestamp_s", ""),
        "label": snapshot.get("label", ""),
        "process_rss_mb": snapshot.get("process_rss_mb", ""),
        "process_vms_mb": snapshot.get("process_vms_mb", ""),
        "process_cpu_percent": snapshot.get("process_cpu_percent", ""),
        "system_cpu_percent": snapshot.get("system_cpu_percent", ""),
        "system_memory_percent": snapshot.get("system_memory_percent", ""),
        "cgroup_cpu_quota_cores": snapshot.get("cgroup_cpu_quota_cores", ""),
        "cgroup_cpuset_cores": snapshot.get("cgroup_cpuset_cores", ""),
        "cgroup_memory_limit_gb": snapshot.get("cgroup_memory_limit_gb", "")
    }

def _build_per_scenario_record(*,
    experiment_profile: str, source_mode: str, profile_name: str, scenario_id: int, frame_idx: int, packet: Dict[str, Any],
    faulted_packet: Dict[str, Any], reliability: Dict[str, Any], final_result: Dict[str, Any], sensor_ms: float, fault_ms: float,
    reliability_ms: float, fusion_ms: float, hazard_ms: float, total_ms: float, process_now: Dict[str, Any]) -> Dict[str, Any]:

    metadata = {}
    metadata.update(packet.get("metadata") or {})
    metadata.update(faulted_packet.get("metadata") or {})
    metadata.update(reliability.get("metadata") or {})
    metadata.update(final_result.get("metadata") or {})

    hazard_fields = {
        "hazard_probability": final_result.get("hazard_probability", 0.0),
        "severity_index": final_result.get("severity_index", 0.0),
        "risk_score": final_result.get("risk_score", 0.0),
        "status": final_result.get("status", ""),
        "threshold_cm": final_result.get("threshold_cm", ""),
        "decision_boundary": final_result.get("decision_boundary", ""),
        "depth_span_cm": final_result.get("depth_span_cm", ""),
        "fusion_confidence": _safe_float(final_result.get("fusion_confidence", final_result.get("posterior_confidence", 0.0)), 0.0),
        "scene_complexity": final_result.get("scene_complexity", packet.get("scene_complexity", 0.0)),
        "expected_hazard_depth": final_result.get("expected_hazard_depth", 0.0),
    }

    return {
        "experiment_profile": experiment_profile,
        "source_mode": source_mode,
        "profile_name": profile_name,
        "scenario_id": scenario_id,
        "frame_idx": frame_idx,
        "road_type": faulted_packet.get("road_type", packet.get("road_type", "")),
        "road_environment": faulted_packet.get("road_environment", packet.get("road_environment", "")),
        "weather": faulted_packet.get("weather", packet.get("weather", "")),
        "lighting": faulted_packet.get("lighting", packet.get("lighting", "")),
        "vehicle_speed": _safe_float(faulted_packet.get("vehicle_speed", packet.get("vehicle_speed", 0.0)), 0.0),
        "pothole_depth": _safe_float(faulted_packet.get("pothole_depth", packet.get("pothole_depth", 0.0)), 0.0),
        "water_depth": _safe_float(faulted_packet.get("water_depth", packet.get("water_depth", 0.0)), 0.0),
        "ntu": _safe_float(faulted_packet.get("ntu", packet.get("ntu", 0.0)), 0.0),
        "true_depth": _safe_float(faulted_packet.get("true_depth", packet.get("true_depth", 0.0)), 0.0),
        "true_ntu": _safe_float(faulted_packet.get("true_ntu", packet.get("true_ntu", packet.get("ntu", 0.0))), 0.0),
        "scene_complexity": _safe_float(faulted_packet.get("scene_complexity", packet.get("scene_complexity", 0.0)), 0.0),
        "lidar": _safe_float(faulted_packet.get("lidar", packet.get("lidar", 0.0)), 0.0),
        "radar": _safe_float(faulted_packet.get("radar", packet.get("radar", 0.0)), 0.0),
        "ultrasonic": _safe_float(faulted_packet.get("ultrasonic", packet.get("ultrasonic", 0.0)), 0.0),
        "imu_pitch": _safe_float(faulted_packet.get("imu_pitch", packet.get("imu_pitch", 0.0)), 0.0),
        "imu_roll": _safe_float(faulted_packet.get("imu_roll", packet.get("imu_roll", 0.0)), 0.0),
        "imu_acceleration": _safe_float(faulted_packet.get("imu_acceleration", packet.get("imu_acceleration", 0.0)), 0.0),
        "turbidity": _safe_float(faulted_packet.get("turbidity", packet.get("turbidity", 0.0)), 0.0),
        "water_contact": _safe_int(faulted_packet.get("water_contact", packet.get("water_contact", 0)), 0),
        "water_contact_state": faulted_packet.get("water_contact_state", packet.get("water_contact_state", "")),
        "fault_profile": faulted_packet.get("fault_profile", profile_name),
        "fault_count": _safe_int(faulted_packet.get("fault_count", 0), 0),
        "affected_sensors": faulted_packet.get("affected_sensors", []),
        "active_faults": faulted_packet.get("active_faults", []),
        "faulted_sensor_mask": faulted_packet.get("faulted_sensor_mask", {}),
        "fault_effects": faulted_packet.get("fault_effects", {}),
        "sensor_health": faulted_packet.get("sensor_health", packet.get("sensor_health", {})),
        "sensor_availability": faulted_packet.get("sensor_availability", packet.get("sensor_availability", {})),
        "sensor_state": faulted_packet.get("sensor_state", packet.get("sensor_state", {})),
        "sensor_metadata": faulted_packet.get("sensor_metadata", packet.get("sensor_metadata",{})),
        "r_lidar": _safe_float(reliability.get("r_lidar", 0.0), 0.0),
        "r_radar": _safe_float(reliability.get("r_radar", 0.0), 0.0),
        "r_ultrasonic": _safe_float(reliability.get("r_ultrasonic", 0.0), 0.0),
        "r_imu": _safe_float(reliability.get("r_imu", 0.0), 0.0),
        "r_turbidity": _safe_float(reliability.get("r_turbidity", 0.0), 0.0),
        "r_water": _safe_float(reliability.get("r_water", 0.0), 0.0),
        "reliability_vector": reliability.get("reliability_vector", []),
        "measurement_vector": reliability.get("measurement_vector", []),
        "measurement_spread": _safe_float(reliability.get("measurement_spread", 0.0), 0.0),
        "sensor_agreement": _safe_float(reliability.get("sensor_agreement", 0.0), 0.0),
        "reliability_spread": _safe_float(reliability.get("reliability_spread", 0.0), 0.0),
        "effective_sensor_count": _safe_float(reliability.get("effective_sensor_count", 0.0), 0.0),
        "dominance_ratio": _safe_float(reliability.get("dominance_ratio", 0.0), 0.0),
        "active_mask": reliability.get("active_mask", []),
        "weights": reliability.get("weights", {}),
        "weight_vector": reliability.get("weight_vector", []),
        "weight_sum": _safe_float(reliability.get("weight_sum", 0.0), 0.0),
        "fusion_confidence": hazard_fields["fusion_confidence"],
        "dominant_sensor": reliability.get("dominant_sensor", ""),
        "weight_entropy": _safe_float(reliability.get("weight_entropy", 0.0), 0.0),
        "weight_variance": _safe_float(reliability.get("weight_variance", 0.0), 0.0),
        "posterior_summary": final_result.get("posterior_summary", {}),
        "uncertainty": final_result.get("uncertainty", {}),
        "posterior_mass_sum": _safe_float(final_result.get("posterior_mass_sum", 0.0), 0.0),
        "posterior_confidence": _safe_float(final_result.get("posterior_confidence", 0.0), 0.0),
        "posterior_peak": _safe_float(final_result.get("posterior_peak", 0.0), 0.0),
        "posterior_entropy": _safe_float(final_result.get("posterior_entropy", 0.0), 0.0),
        "posterior_variance": _safe_float(final_result.get("posterior_variance", 0.0), 0.0),
        "posterior_std": _safe_float(final_result.get("posterior_std", 0.0), 0.0),
        "expected_depth": _safe_float(final_result.get("expected_depth", 0.0), 0.0),
        "map_depth": _safe_float(final_result.get("map_depth", 0.0), 0.0),
        "ci_lower": _safe_float(final_result.get("ci_lower", 0.0), 0.0),
        "ci_upper": _safe_float(final_result.get("ci_upper", 0.0), 0.0),
        "ci_width_95": _safe_float(final_result.get("ci_width_95", 0.0), 0.0),
        "uncertainty_confidence": _safe_float(final_result.get("uncertainty", {}).get(
            "confidence", 0.0), 0.0) if isinstance(final_result.get("uncertainty", {}), dict) else 0.0,
        "uncertainty_map_depth": _safe_float(final_result.get("uncertainty", {}).get(
            "map_depth", 0.0), 0.0) if isinstance(final_result.get("uncertainty", {}), dict) else 0.0,
        "uncertainty_expected_depth": _safe_float(final_result.get("uncertainty", {}).get(
            "expected_depth", 0.0), 0.0) if isinstance(final_result.get("uncertainty", {}), dict) else 0.0,
        "uncertainty_ci_lower": _safe_float(final_result.get("uncertainty", {}).get(
            "ci_lower", 0.0), 0.0) if isinstance(final_result.get("uncertainty", {}), dict) else 0.0,
        "uncertainty_ci_upper": _safe_float(final_result.get("uncertainty", {}).get(
            "ci_upper", 0.0), 0.0) if isinstance(final_result.get("uncertainty", {}), dict) else 0.0,
        "hazard_probability": _safe_float(hazard_fields["hazard_probability"], 0.0),
        "severity_index": _safe_float(hazard_fields["severity_index"], 0.0),
        "risk_score": _safe_float(hazard_fields["risk_score"], 0.0),
        "status": hazard_fields["status"],
        "threshold_cm": _safe_float(hazard_fields["threshold_cm"], 0.0),
        "decision_boundary": _safe_float(hazard_fields["decision_boundary"], 0.0),
        "depth_span_cm": _safe_float(hazard_fields["depth_span_cm"], 0.0),
        "expected_hazard_depth": _safe_float(hazard_fields["expected_hazard_depth"], 0.0),
        "sensor_ms": sensor_ms,
        "fault_ms": fault_ms,
        "reliability_ms": reliability_ms,
        "fusion_ms": fusion_ms,
        "hazard_ms": hazard_ms,
        "total_ms": total_ms,
        "process_rss_mb": _safe_float(process_now.get("process_rss_mb", 0.0), 0.0),
        "process_vms_mb": _safe_float(process_now.get("process_vms_mb", 0.0), 0.0),
        "process_cpu_percent": _safe_float(process_now.get("process_cpu_percent", 0.0), 0.0),
        "system_cpu_percent": _safe_float(process_now.get("system_cpu_percent", 0.0), 0.0),
        "system_memory_percent": _safe_float(process_now.get("system_memory_percent", 0.0), 0.0),
        "resource_timestamp_s": _safe_float(process_now.get("timestamp_s", 0.0), 0.0),
        "resource_label": process_now.get("label", ""),
        "metadata": metadata,
    }

def _summary_from_rows(*,
    experiment_profile: str, source_mode: str, profile_name: str, total_scenarios: int, sensor_latencies: List[float],
    fault_latencies: List[float], reliability_latencies: List[float], fusion_latencies: List[float], hazard_latencies: List[float],
    confidence_list: List[float], risk_list: List[float], hazard_list: List[int], process_before: Dict[str, Any],
    process_after: Dict[str, Any], cfg: VirtualEdgeConfig, resolved_max_scenarios: int) -> Dict[str, Any]:

    def mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "experiment_profile": experiment_profile,
        "profile_name": profile_name,
        "source_mode": source_mode,
        "use_fault_injection": bool(cfg.use_fault_injection),
        "random_seed": int(cfg.random_seed),
        "max_scenarios": int(resolved_max_scenarios),
        "total_scenarios": int(total_scenarios),
        "avg_digital_twin_ms": mean(sensor_latencies),
        "avg_fault_ms": mean(fault_latencies),
        "avg_reliability_ms": mean(reliability_latencies),
        "avg_fusion_ms": mean(fusion_latencies),
        "avg_hazard_ms": mean(hazard_latencies),
        "avg_total_ms": float((sum(sensor_latencies) + sum(fault_latencies) + sum(
            reliability_latencies) + sum(fusion_latencies) + sum(hazard_latencies)) / max(total_scenarios, 1)),
        "mean_confidence": mean(confidence_list),
        "mean_risk_score": mean(risk_list),
        "hazard_count": int(sum(hazard_list)),
        "process_rss_before_mb": _safe_float(process_before.get("process_rss_mb", 0.0), 0.0),
        "process_rss_after_mb": _safe_float(process_after.get("process_rss_mb", 0.0), 0.0),
        "process_cpu_before_percent": _safe_float(process_before.get("process_cpu_percent", 0.0), 0.0),
        "process_cpu_after_percent": _safe_float(process_after.get("process_cpu_percent", 0.0), 0.0),
        "system_cpu_before_percent": _safe_float(process_before.get("system_cpu_percent", 0.0), 0.0),
        "system_cpu_after_percent": _safe_float(process_after.get("system_cpu_percent", 0.0), 0.0),
        "system_memory_before_percent": _safe_float(process_before.get("system_memory_percent", 0.0), 0.0),
        "system_memory_after_percent": _safe_float(process_after.get("system_memory_percent", 0.0), 0.0),
        "cgroup_cpu_quota_cores": process_after.get("cgroup_cpu_quota_cores", ""),
        "cgroup_cpuset_cores": process_after.get("cgroup_cpuset_cores", ""),
        "cgroup_memory_limit_gb": process_after.get("cgroup_memory_limit_gb", ""),
    }

def run_virtual_edge(
    source_mode: Optional[str] = None,
    profile_name: Optional[str] = None,
    max_scenarios: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None) -> dict:

    cfg = VirtualEdgeConfig.from_env()
    cfg.ensure_directories()
    seed_everything(cfg.random_seed)
    monitor = ResourceMonitor("virtual_edge")

    experiment_name = profile_name or cfg.profile_name
    experiment_profile = _load_experiment_profile(cfg.repo_root, experiment_name)
    resolved_source_mode = (source_mode or experiment_profile.get("source_mode", cfg.source_mode)).strip().lower()
    resolved_fault_profile = experiment_profile.get("profile_name", "baseline")
    resolved_max_scenarios = (max_scenarios if max_scenarios is not None else int(
        experiment_profile.get("max_scenarios", cfg.max_scenarios)))
    
    profile_filters = _scene_filters(experiment_profile)
    filters = (filters if filters is not None else profile_filters)
    
    source = load_source(
        mode=resolved_source_mode,
        registry_path=cfg.scenarios_path,
        dataset_path=cfg.dataset_path,
        max_scenarios=resolved_max_scenarios,
        filters=filters)

    run_dir = cfg.output_root / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    digital_twin = DigitalTwinAdapter(seed=cfg.random_seed)
    fault_adapter = FaultAdapter(seed=cfg.random_seed, fps=20)
    reliability_adapter = ReliabilityAdapter(seed=cfg.random_seed)
    fusion_adapter = FusionAdapter(
        depth_min_cm=cfg.depth_min_cm,
        depth_max_cm=cfg.depth_max_cm,
        resolution=cfg.resolution)
    hazard_adapter = HazardAdapter()

    per_scenario_fields = [ "experiment_profile", "source_mode", "profile_name", "scenario_id", "frame_idx", "road_type",
        "road_environment", "weather", "lighting", "vehicle_speed", "pothole_depth", "water_depth", "ntu", "true_depth",
        "true_ntu","scene_complexity","lidar","radar","ultrasonic","imu_pitch","imu_roll","imu_acceleration", "turbidity",
        "water_contact", "water_contact_state", "fault_profile", "fault_count", "affected_sensors", "active_faults",
        "faulted_sensor_mask", "fault_effects", "sensor_health", "sensor_availability", "sensor_state", "sensor_metadata",
        "r_lidar", "r_radar", "r_ultrasonic", "r_imu", "r_turbidity", "r_water", "reliability_vector", "measurement_vector",
        "measurement_spread", "sensor_agreement", "reliability_spread", "effective_sensor_count", "dominance_ratio",
        "active_mask", "weights", "weight_vector", "weight_sum", "fusion_confidence", "dominant_sensor", "weight_entropy",
        "weight_variance", "posterior_summary", "uncertainty", "posterior_mass_sum", "posterior_confidence", "posterior_peak",
        "posterior_entropy", "posterior_variance", "posterior_std", "expected_depth", "map_depth", "ci_lower", "ci_upper",
        "ci_width_95", "uncertainty_confidence", "uncertainty_map_depth", "uncertainty_expected_depth",
        "uncertainty_ci_lower", "uncertainty_ci_upper", "hazard_probability", "severity_index", "risk_score", "status",
        "threshold_cm", "decision_boundary", "depth_span_cm", "expected_hazard_depth", "sensor_ms", "fault_ms",
        "reliability_ms", "fusion_ms", "hazard_ms", "total_ms", "process_rss_mb", "process_vms_mb", "process_cpu_percent",
        "system_cpu_percent", "system_memory_percent", "resource_timestamp_s", "resource_label", "metadata"]

    stage_fields = ["experiment_profile", "source_mode", "fault_profile", "scenario_id", "frame_idx", "stage", "elapsed_ms",
    "timestamp_s", "label", "process_rss_mb", "process_vms_mb", "process_cpu_percent", "system_cpu_percent",
    "system_memory_percent", "cgroup_cpu_quota_cores", "cgroup_cpuset_cores", "cgroup_memory_limit_gb"]

    per_scenario_csv = run_dir / "virtual_edge_scenarios.csv"
    stage_csv = run_dir / "virtual_edge_stage_metrics.csv"
    summary_csv = run_dir / "virtual_edge_summary.csv"
    summary_json = run_dir / "virtual_edge_summary.json"

    scenario_logger = CSVLogger(per_scenario_csv, per_scenario_fields, mode="w")
    stage_logger = CSVLogger(stage_csv, stage_fields, mode="w")
    json_logger = JSONLogger(summary_json)

    sensor_latencies: List[float] = []
    fault_latencies: List[float] = []
    reliability_latencies: List[float] = []
    fusion_latencies: List[float] = []
    hazard_latencies: List[float] = []
    confidence_list: List[float] = []
    risk_list: List[float] = []
    hazard_list: List[int] = []
    total_records = 0

    process_before = monitor.snapshot("before_run").to_dict()
    wall_start = time.perf_counter()
    frame = source.frame.reset_index(drop=True)

    if resolved_source_mode == "registry":
        for idx, row in frame.iterrows():
            scenario = row.to_dict()
            scenario = _scenario_payload_to_common(scenario, resolved_source_mode)

            with stage_timer() as tdt:
                packet = digital_twin.simulate(scenario, frame_idx=idx)
            dt_ms = float(tdt["elapsed_ms"])

            fault_profile = resolved_fault_profile if cfg.use_fault_injection else "baseline"
            with stage_timer() as tft:
                packet_faulted = fault_adapter.apply_packet(packet, profile_name=fault_profile)
            ft_ms = float(tft["elapsed_ms"])

            with stage_timer() as trl:
                reliability = reliability_adapter.compute(packet_faulted)
            rl_ms = float(trl["elapsed_ms"])

            with stage_timer() as tfs:
                fusion = fusion_adapter.fuse(packet_faulted, reliability)
            fs_ms = float(tfs["elapsed_ms"])

            with stage_timer() as thz:
                final_result = hazard_adapter.evaluate(fusion,
                    scene_complexity=_safe_float(packet_faulted.get("scene_complexity", 0.0), 0.0))
            hz_ms = float(thz["elapsed_ms"])

            process_now = monitor.snapshot(f"scenario_{idx + 1}").to_dict()
            total_ms = dt_ms + ft_ms + rl_ms + fs_ms + hz_ms

            record = _build_per_scenario_record(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                profile_name=fault_profile,
                scenario_id=packet_faulted.get("scenario_id", f"SC{idx + 1:03d}"),
                frame_idx=_safe_int(packet_faulted.get("frame_idx", idx), idx),
                packet=packet,
                faulted_packet=packet_faulted,
                reliability=reliability,
                final_result=final_result,
                sensor_ms=dt_ms,
                fault_ms=ft_ms,
                reliability_ms=rl_ms,
                fusion_ms=fs_ms,
                hazard_ms=hz_ms,
                total_ms=total_ms,
                process_now=process_now)

            scenario_logger.write_row(_row_for_csv(record, per_scenario_fields))
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="digital_twin",
                elapsed_ms=dt_ms,
                snapshot=process_now), stage_fields))
            
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="fault_injection",
                elapsed_ms=ft_ms,
                snapshot=process_now), stage_fields))
            
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="reliability",
                elapsed_ms=rl_ms,
                snapshot=process_now), stage_fields))
            
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="fusion",
                elapsed_ms=fs_ms,
                snapshot=process_now), stage_fields))
            
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="hazard",
                elapsed_ms=hz_ms,
                snapshot=process_now), stage_fields))

            sensor_latencies.append(dt_ms)
            fault_latencies.append(ft_ms)
            reliability_latencies.append(rl_ms)
            fusion_latencies.append(fs_ms)
            hazard_latencies.append(hz_ms)

            confidence_list.append(_safe_float(
                final_result.get("posterior_confidence", final_result.get("fusion_confidence", 0.0)), 0.0))
            risk_list.append(_safe_float(final_result.get("risk_score", 0.0), 0.0))
            hazard_list.append(1 if str(final_result.get("status", "")).upper() == "HAZARD" else 0)
            total_records += 1
    else:
        for idx, row in frame.iterrows():
            packet = row.to_dict()
            packet = _scenario_payload_to_common(packet, resolved_source_mode)
            packet.setdefault("scenario_id", _safe_int(packet.get("scenario_id", idx + 1), idx + 1))
            packet.setdefault("frame_idx", idx)
            fault_profile = resolved_fault_profile if cfg.use_fault_injection else "baseline"

            with stage_timer() as tft:
                packet_faulted = fault_adapter.apply_packet(packet, profile_name=fault_profile)
            ft_ms = float(tft["elapsed_ms"])

            with stage_timer() as trl:
                reliability = reliability_adapter.compute(packet_faulted)
            rl_ms = float(trl["elapsed_ms"])

            with stage_timer() as tfs:
                fusion = fusion_adapter.fuse(packet_faulted, reliability)
            fs_ms = float(tfs["elapsed_ms"])

            with stage_timer() as thz:
                final_result = hazard_adapter.evaluate(fusion,
                    scene_complexity=_safe_float(packet_faulted.get("scene_complexity", 0.0), 0.0))
            hz_ms = float(thz["elapsed_ms"])

            process_now = monitor.snapshot(f"dataset_{idx + 1}").to_dict()
            total_ms = ft_ms + rl_ms + fs_ms + hz_ms

            record = _build_per_scenario_record(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                profile_name=fault_profile,
                scenario_id=_safe_int(packet_faulted.get("scenario_id", idx + 1), idx + 1),
                frame_idx=_safe_int(packet_faulted.get("frame_idx", idx), idx),
                packet=packet,
                faulted_packet=packet_faulted,
                reliability=reliability,
                final_result=final_result,
                sensor_ms=0.0,
                fault_ms=ft_ms,
                reliability_ms=rl_ms,
                fusion_ms=fs_ms,
                hazard_ms=hz_ms,
                total_ms=total_ms,
                process_now=process_now)

            scenario_logger.write_row(_row_for_csv(record, per_scenario_fields))
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="fault_injection",
                elapsed_ms=ft_ms,
                snapshot=process_now), stage_fields))
            
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="reliability",
                elapsed_ms=rl_ms,
                snapshot=process_now), stage_fields))
            
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="fusion",
                elapsed_ms=fs_ms,
                snapshot=process_now), stage_fields))
            
            stage_logger.write_row(_row_for_csv(_timing_stage_row(
                experiment_profile=experiment_name,
                source_mode=resolved_source_mode,
                fault_profile=fault_profile,
                scenario_id=record["scenario_id"],
                frame_idx=record["frame_idx"],
                stage="hazard",
                elapsed_ms=hz_ms,
                snapshot=process_now), stage_fields))

            fault_latencies.append(ft_ms)
            reliability_latencies.append(rl_ms)
            fusion_latencies.append(fs_ms)
            hazard_latencies.append(hz_ms)

            confidence_list.append(_safe_float(
                final_result.get("posterior_confidence", final_result.get("fusion_confidence", 0.0)), 0.0))
            risk_list.append(_safe_float(final_result.get("risk_score", 0.0), 0.0))
            hazard_list.append(1 if str(final_result.get("status", "")).upper() == "HAZARD" else 0)
            total_records += 1

    wall_elapsed_s = time.perf_counter() - wall_start
    process_after = monitor.snapshot("after_run").to_dict()

    summary = _summary_from_rows(
        experiment_profile=experiment_name,
        source_mode=resolved_source_mode,
        profile_name=(resolved_fault_profile if cfg.use_fault_injection else "baseline"),
        total_scenarios=total_records,
        sensor_latencies=sensor_latencies,
        fault_latencies=fault_latencies,
        reliability_latencies=reliability_latencies,
        fusion_latencies=fusion_latencies,
        hazard_latencies=hazard_latencies,
        confidence_list=confidence_list,
        risk_list=risk_list,
        hazard_list=hazard_list,
        process_before=process_before,
        process_after=process_after, cfg=cfg,
        resolved_max_scenarios=resolved_max_scenarios)
    
    summary["wall_elapsed_s"] = float(wall_elapsed_s)
    summary["throughput_sps"] = float(total_records / max(wall_elapsed_s, 1e-12))

    summary["filters"] = filters
    summary["original_scenarios"] = source.original_count
    summary["available_after_filter"] = source.filtered_count
    summary["loaded_for_experiment"] = source.loaded_count
    summary["filtered_out"] = (source.original_count - source.filtered_count)
    summary["filter_ratio"] = (source.filtered_count / source.original_count if source.original_count else 0.0)

    summary["mean_water_depth"] = float(frame["water_depth"].mean())
    summary["mean_ntu"] = float(frame["ntu"].mean())
    summary["mean_true_depth"] = float(frame["true_depth"].mean())

    summary_fields = list(summary.keys())
    summary_logger = CSVLogger(summary_csv, summary_fields, mode="w")
    summary_logger.write_row(_row_for_csv(summary, summary_fields))
    json_logger.write(summary)

    return {
        "summary": summary,
        "experiment_profile": experiment_profile,
        "scenario_registry": str(cfg.scenarios_path),

        "paths": {
            "per_scenario_csv": str(per_scenario_csv),
            "stage_csv": str(stage_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        }, "records_processed": total_records,
    }

def main() -> None:
    result = run_virtual_edge()
    summary = result["summary"]

    print("=" * 60)
    print(f"{'FloodTwin-HIL Virtual Edge Run':^60}")
    print("=" * 60)
    
    print(f"Scenario Registry            : {result['scenario_registry']}")
    print(f"CSV (per-scenario)           : {result['paths']['per_scenario_csv']}")
    print(f"CSV (stage metrics)          : {result['paths']['stage_csv']}")
    print(f"CSV (summary)                : {result['paths']['summary_csv']}")
    print(f"JSON (summary)               : {result['paths']['summary_json']}")

    print("=" * 60)
    for key, value in summary.items(): print(f"{key:<32}: {value}")
    print("=" * 60)

if __name__ == "__main__":
    main()