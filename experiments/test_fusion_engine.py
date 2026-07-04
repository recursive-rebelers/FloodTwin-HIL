import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fusion_engine.bayesian_fusion import BayesianFusionCore
from fusion_engine.fixed_fusion import FixedFusionEngine
from fusion_engine.adaptive_fusion import AdaptiveFusionEngine
from fusion_engine.uncertainty import UncertaintyQuantification
from fusion_engine.hazard_assessment import HazardAssessment

DATASET_FILE_IN = BASE_DIR / "datasets" / "dataset_v2.csv"
RESULTS_DIR = BASE_DIR / "results" / "fusion_engine"
DATASET_FILE_OUT = RESULTS_DIR / "fusion_results.csv"
SUMMARY_FILE = RESULTS_DIR / "fusion_summary.csv"

def derive_fixed_sigmas(df):

    mean_r_lidar = float(df["R_lidar"].mean())
    mean_r_ultra = float(df["R_ultrasonic"].mean())
    mean_r_radar = float(df["R_radar"].mean())

    sigma_lidar = np.clip(1.55 / (mean_r_lidar + 0.10), 1.80, 3.30)
    sigma_ultra = np.clip(1.55 / (mean_r_ultra + 0.10), 1.70, 3.20)
    sigma_radar = np.clip(1.45 / (mean_r_radar + 0.10), 1.40, 2.60)

    return float(sigma_lidar), float(sigma_ultra), float(sigma_radar)

def run_fusion_pipeline():

    df = pd.read_csv(DATASET_FILE_IN)
    dataset = []

    core = BayesianFusionCore(depth_min=0.0, depth_max=30.0, resolution=300)

    # Lower gamma reduces over-dominance of the best sensor and typically
    # improves adaptive-vs-fixed performance on mixed-quality scenes
    adaptive_engine = AdaptiveFusionEngine(core)
    fixed_engine = FixedFusionEngine(core)
    uq = UncertaintyQuantification()
    hazard_assessor = HazardAssessment(threshold_cm=10.0, decision_boundary=0.80)
    fixed_sigmas = derive_fixed_sigmas(df)
    np.random.seed(42)

    # Latent pothole depth for fusion benchmarking
    df["true_depth"] = np.random.uniform(2.0, 25.0, len(df))

    # Observation model: sensor noise scales inversely with reliability
    df["z_lidar"] = (df["true_depth"] + np.random.normal(0, 1.425 / (df["R_lidar"] + 0.1), len(df)))
    df["z_ultra"] = (df["true_depth"] + np.random.normal(0, 1.425 / (df["R_ultrasonic"] + 0.1), len(df)))
    df["z_radar"] = (df["true_depth"] + np.random.normal(0, 1.425 / (df["R_radar"] + 0.1), len(df)))

    for _, row in df.iterrows():
        true_depth = float(row["true_depth"])

        # 1) Single-sensor baseline: LiDAR only
        single_est = float(row["z_lidar"])
        single_err = abs(single_est - true_depth)

        # 2) Fixed fusion baseline: static uncertainty, equal weights
        fixed = fixed_engine.estimate(row["z_lidar"], row["z_ultra"], row["z_radar"], sigmas=fixed_sigmas)
        fixed_metrics = uq.compute(core.depths, fixed["posterior"], core.delta_x)

        fixed_est = float(fixed_metrics["map_depth"])
        fixed_err = abs(fixed_est - true_depth)

        # 3) Adaptive fusion: reliability-aware weights and scenario-aware sigmas
        adaptive = adaptive_engine.fuse(
            row["z_lidar"],
            row["z_ultra"],
            row["z_radar"],
            row["R_lidar"],
            row["R_ultrasonic"],
            row["R_radar"],
            sigmas=None
        )

        adaptive_metrics = uq.compute(core.depths, adaptive["posterior"], core.delta_x)
        adapt_est = float(adaptive_metrics["map_depth"])
        adapt_err = abs(adapt_est - true_depth)

        # 4) Hazard assessment on the adaptive posterior
        hazard = hazard_assessor.evaluate(core.depths, adaptive["posterior"], core.delta_x)

        dataset.append({
            "scenario_id": row["scenario_id"],
            "water_depth": float(row["water_depth"]),
            "ntu": float(row["ntu"]),
            "true_depth": true_depth,

            "single_sensor_est": round(single_est, 3),
            "single_mae": round(single_err, 3),

            "fixed_fusion_est": round(fixed_est, 3),
            "fixed_mae": round(fixed_err, 3),

            "adaptive_fusion_est": round(adapt_est, 3),
            "adaptive_mae": round(adapt_err, 3),
            "adaptive_depth_mean": round(float(adaptive_metrics["expected_depth"]), 3),

            "fixed_confidence": round(float(fixed_metrics["confidence"]), 5),
            "adaptive_confidence": round(float(adaptive_metrics["confidence"]), 5),

            "fixed_entropy": round(float(fixed_metrics["entropy"]), 5),
            "adaptive_entropy": round(float(adaptive_metrics["entropy"]), 5),

            "fixed_variance": round(float(fixed_metrics["variance"]), 5),
            "adaptive_variance": round(float(adaptive_metrics["variance"]), 5),

            "entropy_score": round(adaptive_metrics["entropy_score"], 5),
            "peak_score":round(adaptive_metrics["peak_score"], 5),
            "interval_score": round(adaptive_metrics["interval_score"], 5),
            "posterior_peak": float(adaptive_metrics["posterior_peak"]),

            "ci_lower_95": round(float(adaptive_metrics["ci_lower"]), 3),
            "ci_upper_95": round(float(adaptive_metrics["ci_upper"]), 3),

            "adaptive_w_lidar": round(float(adaptive["weights"]["lidar"]), 3),
            "adaptive_w_ultra": round(float(adaptive["weights"]["ultrasonic"]), 3),
            "adaptive_w_radar": round(float(adaptive["weights"]["radar"]), 3),

            "adaptive_sigma_lidar": round(float(adaptive["sigmas"]["lidar"]), 3),
            "adaptive_sigma_ultra": round(float(adaptive["sigmas"]["ultrasonic"]), 3),
            "adaptive_sigma_radar": round(float(adaptive["sigmas"]["radar"]), 3),

            "fixed_sigma_lidar": round(float(fixed_sigmas[0]), 3),
            "fixed_sigma_ultra": round(float(fixed_sigmas[1]), 3),
            "fixed_sigma_radar": round(float(fixed_sigmas[2]), 3),

            "hazard_probability": round(float(hazard["hazard_probability"]), 5),
            "risk_score": round(float(hazard["risk_score"]), 2),
            "hazard_status": hazard["status"]
        })

    df_out = pd.DataFrame(dataset)

    corr_conf_mae = (df_out["adaptive_confidence"].corr(df_out["adaptive_mae"]))
    corr_hazard_depth = (df_out["hazard_probability"].corr(df_out["true_depth"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(DATASET_FILE_OUT, index=False)

    rmse_single = float(np.sqrt((df_out["single_mae"] ** 2).mean()))
    rmse_fixed = float(np.sqrt((df_out["fixed_mae"] ** 2).mean()))
    rmse_adapt = float(np.sqrt((df_out["adaptive_mae"] ** 2).mean()))

    rmse_gain_percent = ((rmse_fixed - rmse_adapt) / rmse_fixed * 100.0
        if rmse_fixed > 1e-12
        else 0.0
    )

    hazard_counts = df_out["hazard_status"].value_counts()
    hazard_rate_percent = 100.0 * hazard_counts.get("HAZARD", 0) / len(df_out)
    caution_rate_percent = 100.0 * hazard_counts.get("CAUTION", 0) / len(df_out)
    safe_rate_percent = 100.0 * hazard_counts.get("SAFE", 0) / len(df_out)
    confidence_gain_percent = float((df_out["adaptive_confidence"].mean() - df_out["fixed_confidence"].mean()) * 100.0)
    adaptive_win_rate = (df_out["adaptive_mae"] < df_out["fixed_mae"]).mean()

    summary = {
        "samples": len(df_out),
        "rmse_single_sensor": rmse_single,
        "rmse_fixed_fusion": rmse_fixed,
        "rmse_adaptive_fusion": rmse_adapt,
        "rmse_gain_percent": rmse_gain_percent,

        "avg_single_mae": float(df_out["single_mae"].mean()),
        "avg_fixed_mae": float(df_out["fixed_mae"].mean()),
        "avg_adaptive_mae": float(df_out["adaptive_mae"].mean()),
        "adaptive_win_rate": adaptive_win_rate,

        "avg_fixed_confidence": float(df_out["fixed_confidence"].mean()),
        "avg_confidence": float(df_out["adaptive_confidence"].mean()),
        "std_confidence": float(df_out["adaptive_confidence"].std()),
        "confidence_gain_percent": confidence_gain_percent,

        "avg_fixed_entropy": float(df_out["fixed_entropy"].mean()),
        "avg_entropy": float(df_out["adaptive_entropy"].mean()),
        "std_entropy": float(df_out["adaptive_entropy"].std()),

        "avg_fixed_variance": float(df_out["fixed_variance"].mean()),
        "avg_variance": float(df_out["adaptive_variance"].mean()),

        "avg_adaptive_w_lidar": float(df_out["adaptive_w_lidar"].mean()),
        "avg_adaptive_w_ultra": float(df_out["adaptive_w_ultra"].mean()),
        "avg_adaptive_w_radar": float(df_out["adaptive_w_radar"].mean()),

        "avg_adaptive_sigma_lidar": float(df_out["adaptive_sigma_lidar"].mean()),
        "avg_adaptive_sigma_ultra": float(df_out["adaptive_sigma_ultra"].mean()),
        "avg_adaptive_sigma_radar": float(df_out["adaptive_sigma_radar"].mean()),

        "corr_confidence_mae": float(corr_conf_mae),
        "corr_hazard_depth": float(corr_hazard_depth),
        
        "avg_hazard_probability": float(df_out["hazard_probability"].mean()),
        "hazards_detected": int(hazard_counts.get("HAZARD", 0)),
        "hazard_rate_percent": hazard_rate_percent,
        "caution_flags": int(hazard_counts.get("CAUTION", 0)),
        "caution_rate_percent": caution_rate_percent,
        "safe_cases": int(hazard_counts.get("SAFE", 0)),
        "safe_rate_percent": safe_rate_percent,

        "mean_entropy_score": df_out["entropy_score"].mean(),
        "std_entropy_score": df_out["entropy_score"].std(),
        "mean_peak_score": df_out["peak_score"].mean(),
        "std_peak_score": df_out["peak_score"].std(),
        "mean_interval_score": df_out["interval_score"].mean(),
        "std_interval_score": df_out["interval_score"].std()
    }

    summary_df = pd.DataFrame([summary])
    summary_df = summary_df.round(5)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print("=" * 60)
    print()
    print(f"Samples : {len(df_out)}")
    print(f"Dataset Saved : {DATASET_FILE_OUT}")
    print(f"Summary Saved : {SUMMARY_FILE}")
    print()
    print(summary_df.T.to_string(header=False))
    print()
    print("Hazard State Distribution")
    print(df_out["hazard_status"].value_counts())
    print()
    print("=" * 60)

if __name__ == "__main__":
    run_fusion_pipeline()