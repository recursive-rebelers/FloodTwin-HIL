import argparse
import sys
from pathlib import Path
from typing import Any
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from virtual_edge.scripts.virtual_edge_runner import run_virtual_edge

def load_experiments(repo_root: Path) -> dict[str, dict[str, Any]]:
    config_path = (repo_root / "virtual_edge" / "config" / "experiment_profiles.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment profile configuration not found:\n"
            f"{config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError("experiment_profiles.yaml is invalid")
    experiments = data.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("'experiments' section must be a dictionary")
    return experiments

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FloodTwin-HIL Virtual Edge Experiments")
    parser.add_argument("--profile", default=None,
        help="Experiment profile name from experiment_profiles.yaml")
    parser.add_argument("--list", action="store_true",
        help="List available experiment profiles")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    experiments = load_experiments(repo_root)

    if args.list:
        if not experiments:
            print("No experiment profiles found")
            return
        print("Available experiment profiles:\n")

        for name, profile in experiments.items():
            description = profile.get("description", "")
            print(f"  {name:<20} {description}")
        return

    profile_name = args.profile or "baseline_registry"
    if profile_name not in experiments:
        raise SystemExit(
            f"Unknown experiment profile '{profile_name}'.\n"
            f"Use '--list' to view available profiles")

    profile = experiments[profile_name]
    result = run_virtual_edge(
        source_mode=profile.get("source_mode"),
        profile_name=profile.get("profile_name", profile_name),
        max_scenarios=profile.get("max_scenarios"),
        filters=profile.get("filters", {}))

    summary = result.get("summary")
    if summary is not None:
        print("=" * 60)
        print(f"{'FloodTwin-HIL Virtual Edge Experimental Run':^60}")
        print("=" * 60)
        for key, value in summary.items(): print(f"{key:<32}: {value}")
        print("=" * 60)
    else:
        print("Experiment Completed Successfully!")

if __name__ == "__main__":
    main()