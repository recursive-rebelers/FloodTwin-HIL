import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fault_injection.dropout_fault import DropoutInjector
from fault_injection.noise_fault import NoiseInjector
from fault_injection.delay_fault import DelayInjector
from fault_injection.drift_fault import DriftInjector
from fault_injection.sync_fault import SyncInjector

class CompoundFaultEngine:

    def __init__(self, random_seed=42, fps=20):
        self.seed = random_seed
        self.dropout = DropoutInjector(random_seed=random_seed)
        self.noise = NoiseInjector(random_seed=random_seed)
        self.delay = DelayInjector(fps=fps)
        self.drift = DriftInjector(random_seed=random_seed)
        self.sync = SyncInjector()

    def _apply_if_present(self, df_faulty, column, transform_fn, *args, **kwargs):
        if column in df_faulty.columns:
            df_faulty[column] = transform_fn(df_faulty[column].to_numpy(), *args, **kwargs)
        return df_faulty

    def apply_scenario_a(self, df, lidar_dropout="high", radar_noise="severe"):
        df_faulty = df.copy()
        df_faulty = self._apply_if_present(df_faulty, "lidar", self.dropout.inject, severity=lidar_dropout)
        df_faulty = self._apply_if_present(df_faulty, "radar", self.noise.inject, severity=radar_noise, clip_min=0.0)
        return df_faulty

    def apply_scenario_b(self, df, imu_drift="medium", ultrasonic_delay="medium"):
        df_faulty = df.copy()
        df_faulty = self._apply_if_present(df_faulty, "imu_pitch", self.drift.inject, severity=imu_drift)
        df_faulty = self._apply_if_present(df_faulty, "imu_roll", self.drift.inject, severity=imu_drift)
        df_faulty = self._apply_if_present(df_faulty, "ultrasonic", self.delay.inject, severity=ultrasonic_delay)
        return df_faulty

    def apply_scenario_c(self, df, lidar_dropout="medium", ultrasonic_noise="severe", radar_sync="high", 
        sync_direction="backward"):
        df_faulty = df.copy()
        df_faulty = self._apply_if_present(df_faulty, "lidar", self.dropout.inject, severity=lidar_dropout)
        df_faulty = self._apply_if_present(df_faulty, "ultrasonic", self.noise.inject, severity=ultrasonic_noise, clip_min=0.0)
        df_faulty = self._apply_if_present(df_faulty, "radar", self.sync.inject, severity=radar_sync, direction=sync_direction)
        return df_faulty

    def apply_compound_profile(self, df, profile_name="scenario_c"):
        profile_name = profile_name.lower().strip()
        
        if profile_name in {"scenario_a", "a"}:
            return self.apply_scenario_a(df)
        if profile_name in {"scenario_b", "b"}:
            return self.apply_scenario_b(df)
        if profile_name in {"scenario_c", "c"}:
            return self.apply_scenario_c(df)

        raise ValueError("profile_name must be one of: scenario_a, scenario_b, scenario_c")

    def metadata(self):
        return {
            "fault_type": "compound_fault",
            "model": "Multi-fault orchestration for robustness evaluation",
            "seed": self.seed,
            "scenarios": {

                "scenario_a": {
                    "lidar": "dropout",
                    "radar": "gaussian_noise"
                },

                "scenario_b": {
                    "imu_pitch": "drift",
                    "imu_roll": "drift",
                    "ultrasonic": "delay"
                },

                "scenario_c": {
                    "lidar": "dropout",
                    "ultrasonic": "gaussian_noise",
                    "radar": "synchronization_error",
                }
            }
        }