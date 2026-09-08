import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import yaml

_ALLOWED_SOURCE_MODES = {"registry", "dataset", "live"}

@dataclass(frozen=True, slots=True)
class VirtualEdgeConfig:

    repo_root: Path
    virtual_edge_root: Path

    scenarios_path: Path
    dataset_path: Path

    output_root: Path
    figures_root: Path
    reports_root: Path

    source_mode: str
    profile_name: str

    random_seed: int
    max_scenarios: Optional[int]

    log_every_n: int
    use_fault_injection: bool

    depth_min_cm: float
    depth_max_cm: float
    resolution: int

    @classmethod
    def from_env(cls) -> "VirtualEdgeConfig":
        repo_root = Path(__file__).resolve().parents[2]
        virtual_edge_root = repo_root / "virtual_edge"
        config_path = (virtual_edge_root / "config" / "edge_config.yaml")
        cfg = cls.from_yaml(config_path)

        runtime = cfg["runtime"]
        paths = cfg["paths"]
        fusion = cfg["fusion"]
        source_mode = os.getenv("FLOODTWIN_SOURCE_MODE", runtime["source_mode"],).strip().lower()

        if source_mode not in _ALLOWED_SOURCE_MODES:
            raise ValueError(
                f"Unsupported source mode: '{source_mode}' "
                f"Expected one of {_ALLOWED_SOURCE_MODES}")

        depth_min = float(fusion["depth_min_cm"])
        depth_max = float(fusion["depth_max_cm"])
        resolution = int(fusion["resolution"])

        if depth_min >= depth_max:
            raise ValueError("depth_min_cm must be smaller than depth_max_cm")
        if resolution <= 0:
            raise ValueError("Fusion resolution must be greater than zero")

        return cls(
            repo_root=repo_root,
            virtual_edge_root=virtual_edge_root,

            scenarios_path=repo_root / paths["scenarios_csv"],
            dataset_path=repo_root / paths["dataset_csv"],

            output_root=repo_root / paths["output_root"],
            figures_root=repo_root / paths["figures_root"],
            reports_root=repo_root / paths["reports_root"],
            source_mode=source_mode,

            profile_name=os.getenv("FLOODTWIN_PROFILE_NAME", runtime["profile_name"]).strip(),
            random_seed=cls._env_int("FLOODTWIN_RANDOM_SEED", runtime["random_seed"]),
            max_scenarios=cls._env_optional_int("FLOODTWIN_MAX_SCENARIOS", runtime["max_scenarios"]),
            log_every_n=max(1, cls._env_int("FLOODTWIN_LOG_EVERY_N", runtime["log_every_n"])),
            use_fault_injection=cls._env_bool("FLOODTWIN_USE_FAULT_INJECTION", runtime["use_fault_injection"]),

            depth_min_cm=depth_min,
            depth_max_cm=depth_max,
            resolution=resolution)

    @staticmethod
    def from_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found:\n{path}")
        with path.open("r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
        if not isinstance(cfg, dict):
            raise ValueError("Configuration YAML is invalid")
        return cfg

    @staticmethod
    def _env_int(name: str, default: Any) -> int:
        value = os.getenv(name)
        if value is None:
            return int(default)
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable '{name}' "
                f"must be an integer") from exc

    @staticmethod
    def _env_optional_int(name: str, default: Any) -> Optional[int]:
        value = os.getenv(name)
        if value is None:
            return None if default is None else int(default)
        value = value.strip()
        if value == "":
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable '{name}' "
                f"must be an integer") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return parsed

    @staticmethod
    def _env_bool(name: str, default: Any) -> bool:
        value = str(os.getenv(name, str(default))).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value for '{name}': {value}")

    def ensure_directories(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.figures_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)