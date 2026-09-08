from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from fault_injection.compound_fault import CompoundFaultEngine
from fault_injection.delay_fault import DelayInjector
from fault_injection.drift_fault import DriftInjector
from fault_injection.dropout_fault import DropoutInjector
from fault_injection.fault_effects import FaultEffects
from fault_injection.noise_fault import NoiseInjector
from fault_injection.sync_fault import SyncInjector

SCALAR_FIELDS = ("lidar", "radar", "ultrasonic", "imu_pitch", "imu_roll", "turbidity", "water_contact")

PROFILE_FIELDS = { "baseline": set(),
    "flood_severity": {"lidar", "ultrasonic", "turbidity", "water_contact"},
    "muddy_water": {"lidar", "radar", "ultrasonic", "turbidity"},
    "high_speed": {"ultrasonic", "imu_pitch", "imu_roll", "radar"},
    "complex_environment": {"lidar", "radar", "ultrasonic", "imu_pitch", "imu_roll", "turbidity"},
    "sensor_failure": {"lidar", "radar", "ultrasonic", "imu_pitch", "imu_roll"},
    "compound_fault": {"lidar", "radar", "ultrasonic"},
    "uncertainty_analysis": {"lidar", "radar", "ultrasonic", "imu_pitch", "imu_roll"},
    "stress_test": {"lidar", "radar", "ultrasonic", "imu_pitch", "imu_roll", "turbidity", "water_contact"}}

COMPOUND_ALIASES = {
    "compound_fault": "scenario_c",
    "compound_c": "scenario_c",
    "scenario_a": "scenario_a",
    "scenario_b": "scenario_b",
    "scenario_c": "scenario_c" }

def _severity_map(profile_name: str) -> dict:
    profile_name = str(profile_name).strip().lower()
    profiles = { "baseline": {},
    
        "flood_severity": {
            "lidar": [{"fault_type": "noise", "severity": "severe"}],
            "ultrasonic": [{"fault_type": "noise", "severity": "severe"}],
            "turbidity": [{"fault_type": "noise", "severity": "moderate"}]},

        "muddy_water": {
            "lidar": [{"fault_type": "noise", "severity": "severe"}],
            "radar": [{"fault_type": "noise", "severity": "mild"}],
            "ultrasonic": [{"fault_type": "noise", "severity": "severe"}],
            "turbidity": [{"fault_type": "noise", "severity": "severe"}]},

        "high_speed": {
            "ultrasonic": [{"fault_type": "delay", "severity": "high"}],
            "imu_pitch": [{"fault_type": "drift", "severity": "medium"}],
            "imu_roll": [{"fault_type": "drift", "severity": "medium"}],
            "radar": [{"fault_type": "sync", "severity": "low", "direction": "backward"}]},

        "complex_environment": {
            "lidar": [{"fault_type": "noise", "severity": "moderate"}],
            "radar": [{"fault_type": "noise", "severity": "mild"}],
            "ultrasonic": [{"fault_type": "delay", "severity": "low"}],
            "imu_pitch": [{"fault_type": "drift", "severity": "low"}],
            "imu_roll": [{"fault_type": "drift", "severity": "low"}],
            "turbidity": [{"fault_type": "noise", "severity": "moderate"}]},

        "sensor_failure": {
            "lidar": [{"fault_type": "dropout", "severity": "high"}],
            "radar": [{"fault_type": "noise", "severity": "severe"}],
            "ultrasonic": [{"fault_type": "delay", "severity": "high"}],
            "imu_pitch": [{"fault_type": "drift", "severity": "medium"}],
            "imu_roll": [{"fault_type": "drift", "severity": "medium"}]},

        "compound_fault": {
            "lidar": [{"fault_type": "dropout", "severity": "medium"}],
            "ultrasonic": [{"fault_type": "noise", "severity": "severe"}],
            "radar": [{"fault_type": "sync", "severity": "high", "direction": "backward"}]},

        "uncertainty_analysis": {
            "lidar": [{"fault_type": "noise", "severity": "mild"}],
            "radar": [{"fault_type": "noise", "severity": "mild"}],
            "ultrasonic": [{"fault_type": "noise", "severity": "mild"}],
            "imu_pitch": [{"fault_type": "drift", "severity": "low"}],
            "imu_roll": [{"fault_type": "drift", "severity": "low"}]},

        "stress_test": {
            "lidar": [{"fault_type": "noise", "severity": "severe"}],
            "radar": [{"fault_type": "noise", "severity": "moderate"}],
            "ultrasonic": [{"fault_type": "delay", "severity": "high"}],
            "imu_pitch": [{"fault_type": "drift", "severity": "high"}],
            "imu_roll": [{"fault_type": "drift", "severity": "high"}],
            "turbidity": [{"fault_type": "noise", "severity": "severe"}],
            "water_contact": [{"fault_type": "dropout", "severity": "medium"}]}}

    return profiles.get(profile_name, profiles["baseline"])

def _fault_signature(fault: Dict[str, Any]) -> tuple:
    return tuple(sorted((str(k), repr(v)) for k, v in fault.items()))

def _dedupe_faults(faults: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for fault in faults:
        sig = _fault_signature(fault)
        if sig in seen: continue
        seen.add(sig)
        out.append(dict(fault))
    return out

def _sensor_key_from_field(field: str) -> str:
    if field in {"imu_pitch", "imu_roll"}:
        return "imu"
    if field == "water_contact":
        return "water"
    return field

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)

def _clip01(value: Any, default: float = 0.0) -> float:
    return float(np.clip(_safe_float(value, default), 0.0, 1.0))

@dataclass
class FaultAdapter:

    seed: int = 42
    fps: int = 20
    history_len: int = 64

    def __post_init__(self):
        self.effects = FaultEffects(random_seed=self.seed)
        self.compound = CompoundFaultEngine(random_seed=self.seed, fps=self.fps)
        self.dropout = DropoutInjector(random_seed=self.seed)
        self.noise = NoiseInjector(random_seed=self.seed)
        self.delay = DelayInjector(fps=self.fps)
        self.drift = DriftInjector(random_seed=self.seed)
        self.sync = SyncInjector()
        self._history = defaultdict(lambda: deque(maxlen=self.history_len))
        self.frame_idx = 0

    def _apply_scalar_fault(self, values: np.ndarray, fault: dict, column: str) -> np.ndarray:
        fault_type = str(fault.get("fault_type", "")).strip().lower()
        severity = str(fault.get("severity", "medium")).strip().lower()

        if fault_type == "dropout":
            return self.dropout.inject(values, severity=severity)

        if fault_type == "noise":
            clip_min = 0.0 if column in {"lidar", "radar", "ultrasonic", "turbidity", "water_contact"} else None
            return self.noise.inject(values, severity=severity, clip_min=clip_min)

        if fault_type == "delay":
            return self.delay.inject(values, severity=severity, hold_first=True)

        if fault_type == "drift":
            return self.drift.inject(
                values, severity=severity,
                direction=float(fault.get("direction", 1.0)),
                walk_sigma=float(fault.get("walk_sigma", 0.0)),
                start_bias=float(fault.get("start_bias", 0.0)))

        if fault_type == "sync":
            direction = str(fault.get("direction", "backward"))
            return self.sync.inject(values, severity=severity, direction=direction)
        return values

    def _apply_faults_to_series(self, values: np.ndarray, faults: Sequence[Dict[str, Any]], column: str) -> np.ndarray:
        result = np.asarray(values, dtype=float).copy()
        for fault in _dedupe_faults(faults):
            result = self._apply_scalar_fault(result, fault, column)
        return result

    def _sensor_state_from_packet(self, packet: Dict[str, Any], sensor: str) -> Dict[str, float]:
        sensor_health = packet.get("sensor_health", {}) if isinstance(packet.get("sensor_health"), dict) else {}
        sensor_availability = packet.get("sensor_availability", {}) if isinstance(packet.get("sensor_availability"), dict) else {}

        reliability_key = {
            "lidar": "lidar_reliability_prior",
            "radar": "radar_reliability_prior",
            "ultrasonic": "ultrasonic_reliability_prior",
            "imu": "imu_reliability_prior",
            "turbidity": "turbidity_reliability_prior",
            "water": "water_contact_reliability_prior"}[sensor]

        sigma_keys = {
            "lidar": ["lidar_noise_sigma"],
            "radar": ["radar_noise_sigma"],
            "ultrasonic": ["ultrasonic_sigma", "ultrasonic_surface_sigma", "ultrasonic_bottom_sigma"],
            "imu": ["imu_sigma"],
            "turbidity": ["turbidity_sigma", "turbidity_noise_sigma"],
            "water": ["water_contact_sigma", "water_contact_noise_sigma"]}[sensor]

        if sensor == "ultrasonic":
            sigma_values = [
                _safe_float(packet.get("ultrasonic_sigma", np.nan), np.nan),
                _safe_float(packet.get("ultrasonic_surface_sigma", np.nan), np.nan),
                _safe_float(packet.get("ultrasonic_bottom_sigma", np.nan), np.nan)]
            sigma_finite = [v for v in sigma_values if np.isfinite(v)]
            sigma = float(np.mean(sigma_finite)) if sigma_finite else 1.0
        else:
            sigma_candidates = [_safe_float(packet.get(k, np.nan), np.nan) for k in sigma_keys]
            sigma_finite = [v for v in sigma_candidates if np.isfinite(v)]
            sigma = float(np.mean(sigma_finite)) if sigma_finite else 1.0

        packet_sensor_key = "water_contact" if sensor == "water" else sensor

        return {
            "reliability": _clip01(packet.get(reliability_key, 1.0), 1.0),
            "sigma": float(max(sigma, 1e-6)),
            "availability": _clip01(sensor_availability.get(packet_sensor_key, 1.0), 1.0),
            "health": _clip01(sensor_health.get(packet_sensor_key, 1.0), 1.0),
        }

    def _update_sensor_state(
        self, sensor: str, fault_sequence: Sequence[Dict[str, Any]], packet: Dict[str, Any]) -> Dict[str, Any]:
        state = self._sensor_state_from_packet(packet, sensor)
        before = dict(state)

        if fault_sequence:
            effects_sensor = "water" if sensor in {"turbidity", "water"} else sensor
            updated = self.effects.apply_fault_sequence(
                sensor=effects_sensor,
                fault_sequence=fault_sequence,
                reliability=state["reliability"],
                sigma=state["sigma"],
                availability=state["availability"])
            
            state["reliability"] = _clip01(updated.get("reliability", state["reliability"]), state["reliability"])
            state["sigma"] = float(max(_safe_float(updated.get("sigma", state["sigma"]), state["sigma"]), 1e-6))
            state["availability"] = _clip01(updated.get("availability", state["availability"]), state["availability"])

        state["health"] = float(np.clip(0.5 * state["reliability"] + 0.5 * state["availability"], 0.0, 1.0))

        return {
            "sensor": sensor,
            "fault_types": [str(f.get("fault_type", "none")) for f in fault_sequence] if fault_sequence else ["none"],
            "severities": [str(f.get("severity", "none")) for f in fault_sequence] if fault_sequence else ["none"],
            "fault_count": len(fault_sequence),
            "before": before,
            "after": state,
            "delta_reliability": float(state["reliability"] - before["reliability"]),
            "delta_sigma": float(state["sigma"] - before["sigma"]),
            "delta_availability": float(state["availability"] - before["availability"]),
            "delta_health": float(state["health"] - before["health"])}

    def _update_packet_sensor_state_fields(self, out: Dict[str, Any], sensor: str, new_state: Dict[str, Any]) -> None:
        sensor_health = dict(out.get("sensor_health", {})) if isinstance(out.get("sensor_health"), dict) else {}
        sensor_availability = dict(out.get("sensor_availability", {})) if isinstance(out.get("sensor_availability"), dict) else {}
        sensor_state = dict(out.get("sensor_state", {})) if isinstance(out.get("sensor_state"), dict) else {}
        packet_sensor_key = "water_contact" if sensor == "water" else sensor
        sensor_health[packet_sensor_key] = float(new_state["health"])
        sensor_availability[packet_sensor_key] = float(new_state["availability"])

        sensor_state[sensor] = {
            "reliability": float(new_state["reliability"]),
            "sigma": float(new_state["sigma"]),
            "availability": float(new_state["availability"]),
            "health": float(new_state["health"])}

        out["sensor_health"] = sensor_health
        out["sensor_availability"] = sensor_availability
        out["sensor_state"] = sensor_state

        if sensor == "lidar":
            out["lidar_reliability_prior"] = float(new_state["reliability"])
            out["lidar_noise_sigma"] = float(new_state["sigma"])
            out["lidar_health"] = float(new_state["health"])
        elif sensor == "radar":
            out["radar_reliability_prior"] = float(new_state["reliability"])
            out["radar_noise_sigma"] = float(new_state["sigma"])
            out["radar_health"] = float(new_state["health"])
        elif sensor == "ultrasonic":
            out["ultrasonic_reliability_prior"] = float(new_state["reliability"])
            out["ultrasonic_sigma"] = float(new_state["sigma"])
            out["ultrasonic_surface_sigma"] = float(new_state["sigma"])
            out["ultrasonic_bottom_sigma"] = float(new_state["sigma"])
            out["ultrasonic_health"] = float(new_state["health"])
        elif sensor == "imu":
            out["imu_reliability_prior"] = float(new_state["reliability"])
            out["imu_sigma"] = float(new_state["sigma"])
            out["imu_health"] = float(new_state["health"])
        elif sensor == "turbidity":
            out["turbidity_reliability_prior"] = float(new_state["reliability"])
            out["turbidity_sigma"] = float(new_state["sigma"])
            out["turbidity_health"] = float(new_state["health"])
        elif sensor == "water":
            out["water_contact_reliability_prior"] = float(new_state["reliability"])
            out["water_contact_sigma"] = float(new_state["sigma"])
            out["water_contact_health"] = float(new_state["health"])

    def apply_batch_profile(self, frame: pd.DataFrame, profile_name: str) -> pd.DataFrame:
        profile_name = str(profile_name).strip().lower()
        profile = _severity_map(profile_name)

        out = frame.copy()
        compound_profile = COMPOUND_ALIASES.get(profile_name)
        compound_applied_fields = set()

        if compound_profile and {"lidar", "radar", "ultrasonic", "imu_pitch", "imu_roll"
        }.issubset(set(out.columns)):
            out = self.compound.apply_compound_profile(out, profile_name=compound_profile)
            compound_applied_fields = {field for field in profile if field in out.columns}

        for field in ("lidar", "radar", "ultrasonic", "imu_pitch",
            "imu_roll","turbidity", "water_contact"):
            if field in compound_applied_fields: continue
            if field not in out.columns or field not in profile: continue
            values = out[field].to_numpy(dtype=float, copy=True)
            out[field] = self._apply_faults_to_series(values, profile[field], field)

        out["fault_profile"] = profile_name
        return out

    def apply_packet(self, packet: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
        profile_name = str(profile_name).strip().lower()
        profile = _severity_map(profile_name)
        out = dict(packet or {})
        self.frame_idx += 1

        active_faults: List[Dict[str, Any]] = []
        affected_sensors: List[str] = []
        computed_sensor_states: Dict[str, Dict[str, Any]] = {}
        faulted_sensor_mask = {}

        # Faulted signal fields are tracked causally using history buffers
        for field in SCALAR_FIELDS:
            if field not in out: continue
            faults = profile.get(field, [])
            if not faults:
                value = out[field]
                if value is not None:
                    self._history[field].append(_safe_float(value, np.nan))
                continue

            sensor = _sensor_key_from_field(field)
            affected_sensors.append(sensor)

            # Build a causal window so delay / sync faults have visible temporal effect
            value = out[field]
            if value is None: continue
            series = np.asarray(list(self._history[field]) + [_safe_float(value, np.nan)], dtype=float)
            series = np.nan_to_num(series, nan=np.nanmedian(series) if np.any(np.isfinite(series)) else 0.0)

            transformed = self._apply_faults_to_series(series, faults, field)
            new_value = float(transformed[-1]) if np.isfinite(transformed[-1]) else np.nan
            out[field] = new_value
            self._history[field].append(new_value)

            active_faults.extend([{
                "sensor": sensor, "field": field,
                "fault_type": str(f.get("fault_type", "none")),
                "severity": str(f.get("severity", "none")),
                "direction": f.get("direction", None),
            } for f in faults])
            faulted_sensor_mask[sensor] = True

        # IMU faults are attached to both pitch and roll, but the sensor state should be updated once
        if "imu_pitch" in profile or "imu_roll" in profile:
            imu_faults = _dedupe_faults((profile.get("imu_pitch", []) or []) + (profile.get("imu_roll", []) or []))
            imu_state = self._update_sensor_state("imu", imu_faults, out)
            computed_sensor_states["imu"] = imu_state
            self._update_packet_sensor_state_fields(out, "imu", imu_state["after"])
        else:
            # Ensure state keys exist even when no fault is applied
            if "imu_reliability_prior" in out or isinstance(out.get("sensor_state"), dict):
                imu_state = self._update_sensor_state("imu", [], out)
                computed_sensor_states["imu"] = imu_state
                self._update_packet_sensor_state_fields(out, "imu", imu_state["after"])

        # Update per-sensor states for single-sensor fault groups
        for sensor in ("lidar", "radar", "ultrasonic", "turbidity", "water"):
            if sensor == "water":
                faults = profile.get("water_contact", [])
            else:
                if sensor not in profile: continue
                faults = profile.get(sensor, [])

            sensor_state = self._update_sensor_state(sensor, faults, out)
            computed_sensor_states[sensor] = sensor_state
            self._update_packet_sensor_state_fields(out, sensor, sensor_state["after"])
            if sensor != "water":
                faulted_sensor_mask[sensor] = bool(faults)
            
        sensor_fault_summaries = computed_sensor_states

        # ---------------- Refresh Observable Context ----------------
        if "turbidity" in out:
            ntu_observed = _safe_float(out.get("turbidity"), 0.0)
            out["ntu_observed"] = ntu_observed
            out["ntu_available"] = int(np.isfinite(ntu_observed) and out.get(
                "sensor_availability", {}).get("turbidity", 1.0) > 0.0)

        if "water_contact" in out:
            water_context = _clip01(out.get("water_contact"), 0.0)
            out["water_context"] = water_context

        if "ultrasonic" in out:
            ultrasonic_value = _safe_float(out.get("ultrasonic"), np.nan)
            if np.isfinite(ultrasonic_value):
                out["water_context_available"] = int(out.get(
                    "sensor_availability", {}).get("ultrasonic", 1.0) > 0.0)

        out["fault_profile"] = profile_name
        out["fault_count"] = len(active_faults)
        out["affected_sensors"] = sorted(set(affected_sensors))
        out["faulted_sensor_mask"] = faulted_sensor_mask
        out["active_faults"] = active_faults
        out["fault_effects"] = sensor_fault_summaries

        out["metadata"] = {
            "fault_type": "fault_effect_mapping",
            "model": "Fault-aware reliability and uncertainty propagation",
            "seed": self.seed,
            "fps": self.fps,
            "history_len": self.history_len }
        return out