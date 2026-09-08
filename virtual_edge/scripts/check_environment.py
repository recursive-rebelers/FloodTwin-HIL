import importlib
import os
import platform
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from virtual_edge.scripts.performance_monitor import (
    get_cgroup_cpu_quota_cores, get_cgroup_cpuset_cores, get_cgroup_memory_limit_gb)

REQUIRED_FOLDERS = (
    "digital_twin", "reliability_engine", "fusion_engine", "fault_injection", "datasets", "results")

REQUIRED_MODULES = (
    "numpy", "pandas", "scipy", "matplotlib", "psutil", "sklearn", "networkx", "yaml")

def _print_header() -> None:
    print("=" * 60)
    print(f"{'FloodTwin-HIL Virtual Edge Environment Validation':^60}")
    print("=" * 60)

def _print_system_info(repo_root: Path) -> None:
    print(f"Hostname                 : {platform.node()}")
    print(f"Platform                 : {platform.platform()}")
    print(f"Architecture             : {platform.machine()}")
    print(f"Python Version           : {platform.python_version()}")
    print(f"Python Executable        : {sys.executable}")

    exmo = ("Docker" if Path("/.dockerenv").exists() else "Host")
    print(f"Execution Model          : {exmo}")
    print(f"Visible CPUs (Host)      : {os.cpu_count()}")
    print(f"CPU Quota (cgroup)       : {get_cgroup_cpu_quota_cores():.2f} cores")
    print(f"CPU Set (cgroup)         : {get_cgroup_cpuset_cores()} cores")
    print(f"Memory Limit (cgroup)    : {get_cgroup_memory_limit_gb():.2f} GB")
    print(f"Workspace Exists         : {repo_root.exists()}")

def _print_workspace(repo_root: Path) -> None:
    print("-" * 60)
    print("Workspace Validation")
    print("-" * 60)

    for folder in REQUIRED_FOLDERS:
        path = repo_root / folder
        status = "✓" if path.exists() else "✗"
        print(f"{folder:<30}{status}")

def _print_dependencies() -> None:
    print("-" * 60)
    print("Python Dependencies")
    print("-" * 60)

    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "Unknown")
            print(f"{module_name:<20} OK ({version})")
        except Exception as exc:
            print(f"{module_name:<20} MISSING ({exc})")

def _print_configuration(repo_root: Path, cfg: dict) -> None:
    print("-" * 60)
    print("Runtime Configuration")
    print("-" * 60)

    runtime = cfg.get("runtime", {})
    paths = cfg.get("paths", {})
    registry = repo_root / paths.get("scenarios_csv", "")
    dataset = repo_root / paths.get("dataset_csv", "")

    print(f"Source Mode              : {runtime.get('source_mode', 'Unknown')}")
    print(f"Profile Name             : {runtime.get('profile_name', 'Unknown')}")
    print(f"Scenario Registry        : {registry}")
    print(f"Dataset                  : {dataset}")

    print(f"Scenario Registry Exists : {'✓' if registry.exists() else '✗'}")
    print(f"Dataset Exists           : {'✓' if dataset.exists() else '✗'}")

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / "virtual_edge" / "config" / "edge_config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found:\n{config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    _print_header()
    _print_system_info(repo_root)
    _print_workspace(repo_root)
    _print_dependencies()
    _print_configuration(repo_root, cfg)

    print("=" * 60)
    print(f"{'Environment Validation Completed!':^60}")
    print("=" * 60)

if __name__ == "__main__":
    main()