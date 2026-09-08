import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ValidationEvidenceBands:

    excellent: float = 0.85
    strong: float = 0.70
    moderate: float = 0.50

class ValidationEvidenceScorer:

    DEFAULT_TARGET_SAMPLES = 30000
    DEPTH_RANGE_CM = (0.0, 30.0)
    WATER_PROXY_RANGE_CM = (0.0, 20.0)
    NTU_RANGE = (0.0, 500.0)
    SCENE_RANGE = (0.0, 1.0)
    CLUTTER_RANGE = (0.0, 0.30)

    EXPECTED_FAULT_SCENARIOS = (
        "baseline",
        "lidar_low_dropout",
        "lidar_high_dropout",
        "ultrasonic_high_dropout",
        "radar_high_dropout",
        "radar_severe_noise",
        "ultrasonic_high_delay",
        "radar_high_sync_error",
        "compound_scenario_a",
        "compound_scenario_c",
    )

    DEFAULT_WEIGHTS = {
        "dataset_coverage": 0.10,
        "environment_coverage": 0.10,
        "fault_coverage": 0.12,
        "benchmark_breadth": 0.08,
        "predictive_accuracy": 0.16,
        "uncertainty_calibration": 0.14,
        "statistical_evidence": 0.10,
        "validation_stability": 0.10,
        "hazard_validation": 0.10,
    }

    @staticmethod
    def _to_frame(data: Optional[Union[pd.DataFrame, Dict[str, Any]]]) -> pd.DataFrame:
        if data is None: return pd.DataFrame()
        if isinstance(data, pd.DataFrame): return data.copy()
        if isinstance(data, dict):
            if all(not isinstance(v, (list, tuple, np.ndarray, pd.Series))
                   for v in data.values()):
                return pd.DataFrame([data])
            return pd.DataFrame(data)
        raise TypeError("summary_df must be a pandas DataFrame, dict, or None")

    @staticmethod
    def _as_numeric(values: Any) -> np.ndarray:
        if values is None: return np.empty(0, dtype=float)
        arr = pd.to_numeric(values, errors="coerce")
        if isinstance(arr, pd.Series):
            arr = arr.to_numpy(dtype=float)
        else:
            arr = np.asarray(arr, dtype=float)
        arr = arr.reshape(-1)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _safe_numeric_series(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]:
        for col in candidates:
            if col in df.columns: return pd.to_numeric(df[col], errors="coerce")
        return None

    @staticmethod
    def _normalize_fraction(value: Any) -> Optional[float]:
        try: x = float(value)
        except (TypeError, ValueError): return None
        if not np.isfinite(x): return None
        if x > 1.0: x /= 100.0
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _clip01(value: Any, default: float = 0.0) -> float:
        try: x = float(value)
        except (TypeError, ValueError): x = float(default)
        if not np.isfinite(x): x = float(default)
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _summary_lookup_raw(summary: pd.DataFrame, names: Sequence[str]) -> Any:
        if summary.empty: return None
        if {"metric", "value"}.issubset(summary.columns):
            metric_map: Dict[str, Any] = {}
            for metric, value in zip(summary["metric"], summary["value"]):
                if pd.notna(metric):
                    metric_map[str(metric).strip().lower()] = value
            for name in names:
                key = str(name).strip().lower()
                if key in metric_map:
                    value = metric_map[key]
                    return value if not (isinstance(value, float) and np.isnan(value)) else None

        for name in names:
            if name in summary.columns and len(summary):
                value = summary[name].iloc[0]
                if isinstance(value, (tuple, list, np.ndarray, pd.Series)): return value
                try:
                    if pd.notna(value): return value
                except (TypeError, ValueError): return value

        if not isinstance(summary.index, pd.RangeIndex):
            lower_index = {str(i).strip().lower(): i for i in summary.index}
            for name in names:
                idx = lower_index.get(str(name).strip().lower())
                if idx is not None:
                    row = summary.loc[idx]
                    if isinstance(row, pd.Series):
                        for value in row.tolist():
                            if isinstance(value, (tuple, list, np.ndarray)): return value
                            try:
                                if pd.notna(value): return value
                            except (TypeError, ValueError): return value
        return None

    @staticmethod
    def _summary_lookup(summary: pd.DataFrame, names: Sequence[str]) -> Optional[float]:
        value = ValidationEvidenceScorer._summary_lookup_raw(summary, names)
        if value is None or isinstance(value, (tuple, list, np.ndarray, pd.Series, pd.DataFrame)): return None
        try: value = float(value)
        except (TypeError, ValueError): return None
        return value if np.isfinite(value) else None

    @staticmethod
    def _summary_lookup_text(summary: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        if summary.empty: return None
        if {"metric", "value"}.issubset(summary.columns):
            metric_map = {str(m).strip().lower(): v
                for m, v in zip(summary["metric"], summary["value"]) if pd.notna(m)}
            for name in names:
                key = str(name).strip().lower()
                if key in metric_map and pd.notna(metric_map[key]): return str(metric_map[key])
        for name in names:
            if name in summary.columns and len(summary):
                value = summary[name].iloc[0]
                if pd.notna(value): return str(value)
        return None

    @staticmethod
    def _evaluation_frame(df: pd.DataFrame) -> pd.DataFrame:
        for col in ("split", "dataset_split", "partition"):
            if col in df.columns:
                labels = df[col].astype(str).str.strip().str.lower()
                mask = labels.isin({"evaluation", "eval", "test", "testing", "holdout", "held-out"})
                if mask.any(): return df.loc[mask].copy()
        return df.copy()

    @staticmethod
    def _range_coverage(values: Any, lower: float, upper: float) -> Optional[float]:
        arr = ValidationEvidenceScorer._as_numeric(values)
        if arr.size == 0: return None
        q05, q95 = np.quantile(arr, [0.05, 0.95])
        target_span = max(float(upper - lower), 1e-12)
        observed_span = max(0.0, float(q95 - q05))
        return float(np.clip(observed_span / target_span, 0.0, 1.0))

    @staticmethod
    def _bin_coverage(values: Any, lower: float, upper: float, bin_width: float) -> Optional[float]:
        arr = ValidationEvidenceScorer._as_numeric(values)
        if arr.size == 0: return None
        arr = np.clip(arr, lower, upper)
        n_bins = max(int(math.ceil((upper - lower) / bin_width)), 1)
        bins = np.floor((arr - lower) / max(bin_width, 1e-12)).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        return float(np.clip(np.unique(bins).size / n_bins, 0.0, 1.0))

    @staticmethod
    def _presence_average(scores: Sequence[Optional[float]]) -> float:
        if not scores: return 0.0
        numeric = [0.0 if s is None else float(np.clip(s, 0.0, 1.0)) for s in scores]
        return float(np.mean(numeric))

    @staticmethod
    def _extract_best_metric(
        df: pd.DataFrame, summary: pd.DataFrame, candidates: Sequence[str]) -> Optional[float]:
        value = ValidationEvidenceScorer._summary_lookup(summary, candidates)
        if value is not None: return value
        series = ValidationEvidenceScorer._safe_numeric_series(df, candidates)
        if series is not None:
            finite = series[np.isfinite(series)]
            if len(finite): return float(finite.mean())
        return None

    @staticmethod
    def _relative_improvement_score(
        baseline: Optional[float], proposed: Optional[float]) -> Optional[float]:
        if baseline is None or proposed is None: return None
        if not np.isfinite(baseline) or not np.isfinite(proposed): return None
        if abs(float(baseline)) <= 1e-12: return None
        improvement = (float(baseline) - float(proposed)) / abs(float(baseline))
        return float(np.clip(0.5 + 0.5 * improvement, 0.0, 1.0))

    @staticmethod
    def _error_quality_score(error_cm: Optional[float]) -> Optional[float]:
        if error_cm is None or not np.isfinite(error_cm): return None
        span = ValidationEvidenceScorer.DEPTH_RANGE_CM[1] - ValidationEvidenceScorer.DEPTH_RANGE_CM[0]
        return float(np.clip(1.0 - max(float(error_cm), 0.0) / span, 0.0, 1.0))

    @staticmethod
    def _group_dispersion_score(
        df: pd.DataFrame, value_candidates: Sequence[str],
        group_candidates: Sequence[str]) -> Optional[float]:
        values = ValidationEvidenceScorer._safe_numeric_series(df, value_candidates)
        if values is None: return None
        group_col = next((c for c in group_candidates if c in df.columns), None)
        if group_col is None: return None
        temp = pd.DataFrame({"value": values, "group": df[group_col].astype(str)
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if temp.empty: return None
        grouped = temp.groupby("group")["value"].mean()
        grouped = grouped[np.isfinite(grouped.to_numpy())]

        if len(grouped) < 2: return None
        mean = float(grouped.mean())
        std = float(grouped.std(ddof=1))

        if mean <= 1e-12: return 1.0 if std <= 1e-12 else 0.0
        cv = std / mean
        return float(np.clip(1.0 - (cv - 0.10) / 0.90, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Component calculators
    # ------------------------------------------------------------------
    @classmethod
    def _component_dataset_coverage(
        cls, df: pd.DataFrame, summary: pd.DataFrame,
        target_samples: int) -> Tuple[float, Dict[str, float]]:
        reference = cls._safe_numeric_series(
            df, ("true_depth", "reference_depth", "depth", "pothole_depth"))

        range_cov = (
           cls._range_coverage(reference, *cls.DEPTH_RANGE_CM) if reference is not None else None)
        bin_cov = (
           cls._bin_coverage(reference, *cls.DEPTH_RANGE_CM, 1.0) if reference is not None else None)

        n_samples = len(df)
        if n_samples == 0:
            summary_samples = cls._summary_lookup(summary, ("samples", "n_samples", "total_samples"))
            n_samples = int(summary_samples or 0)
        sample_cov = float(np.clip(n_samples / max(int(target_samples), 1), 0.0, 1.0))

        score = (0.45 * (0.0 if range_cov is None else range_cov)
            + 0.35 * (0.0 if bin_cov is None else bin_cov)
            + 0.20 * sample_cov)

        return float(np.clip(score, 0.0, 1.0)), {
            "depth_range_coverage": 0.0 if range_cov is None else range_cov,
            "depth_bin_coverage": 0.0 if bin_cov is None else bin_cov,
            "sample_coverage": sample_cov,
            "sample_count": float(n_samples),
        }

    @classmethod
    def _component_environment_coverage(cls, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        water_proxy = cls._safe_numeric_series(df, ("water_context_depth_proxy",
            "observable_water_depth", "water_proxy"))
        ntu = cls._safe_numeric_series(df, ( "ntu_observed", "observed_ntu", "ntu_sensor",
            "ntu_proxy", "observed_turbidity", "turbidity_observed"))
        scene = cls._safe_numeric_series(df, ("scene_complexity", "scene_difficulty"))
        clutter = cls._safe_numeric_series(df, ("clutter_probability", "clutter", "radar_clutter"))

        scores = {
            "water_context_coverage": cls._range_coverage(
                water_proxy, *cls.WATER_PROXY_RANGE_CM) if water_proxy is not None else None,
            "ntu_coverage": cls._range_coverage(
                ntu, *cls.NTU_RANGE ) if ntu is not None else None,
            "scene_coverage": cls._range_coverage(
                scene, *cls.SCENE_RANGE ) if scene is not None else None,
            "clutter_coverage": cls._range_coverage(
                clutter, *cls.CLUTTER_RANGE ) if clutter is not None else None,
        }

        score = cls._presence_average(tuple(scores.values()))
        details = {key: 0.0 if value is None else float(value) for key, value in scores.items()}
        details["observable_environment_dimensions_present"] = float(
            sum(value is not None for value in scores.values()))
        return float(np.clip(score, 0.0, 1.0)), details

    @classmethod
    def _component_fault_coverage(
        cls, df: pd.DataFrame, summary: pd.DataFrame,
        fault_df: Optional[pd.DataFrame] = None) -> Tuple[float, Dict[str, float]]:
        label_cols = ("fault_scenario", "fault_type", "experiment",
            "Experiment", "benchmark_category", "category", "scenario")

        source = fault_df if fault_df is not None and not fault_df.empty else df
        labels = pd.Series(dtype=str)
        for col in label_cols:
            if col in source.columns:
                labels = source[col].dropna().astype(str).str.strip().str.lower()
                if len(labels): break

        def canonical(text: str) -> str:
            value = str(text).strip().lower()
            value = value.replace("%", "pct")
            value = value.replace("&", "and")
            value = value.replace("-", "_").replace(" ", "_")
            value = value.replace("/", "_")
            while "__" in value: value = value.replace("__", "_")
            return value

        observed = {canonical(x) for x in labels.tolist() if x}
        expected = set(cls.EXPECTED_FAULT_SCENARIOS)

        matched = set()
        for expected_name in expected:
            tokens = [t for t in expected_name.split("_") if t]
            for label in observed:
                if all(token in label.replace("(", "_").replace(")", "_")
                       for token in tokens):
                    matched.add(expected_name)
                    break

        summary_faults = cls._summary_lookup(
            summary, ("fault_experiments", "fault_scenarios", "fault_cases"))
        summary_fault_score = (
            float(np.clip(summary_faults / max(len(expected), 1), 0.0, 1.0))
            if summary_faults is not None else 0.0)

        scenario_score = len(matched) / max(len(expected), 1)
        if len(labels) == 0: scenario_score = summary_fault_score
        fault_cases_present = sum(name != "baseline" and name in matched for name in expected)
        baseline_present = float("baseline" in matched)
        baseline_fault_balance = 0.10 if baseline_present and fault_cases_present else 0.0
        score = min(1.0, 0.90 * scenario_score + baseline_fault_balance)

        return float(np.clip(score, 0.0, 1.0)), {
            "expected_fault_scenarios": float(len(expected)),
            "matched_fault_scenarios": float(len(matched)),
            "fault_cases_present": float(fault_cases_present),
            "baseline_present": baseline_present,
            "scenario_coverage": float(np.clip(scenario_score, 0.0, 1.0)),
            "summary_fault_count_score": summary_fault_score,
        }

    @classmethod
    def _component_benchmark_breadth(
        cls, df: pd.DataFrame, summary: pd.DataFrame,
        target_samples: int) -> Tuple[float, Dict[str, float]]:
        n_samples = len(df)
        if n_samples == 0:
            n_samples = int(cls._summary_lookup(summary, (
                "samples", "n_samples", "total_samples")) or 0)

        sample_score = float(np.clip(n_samples / max(int(target_samples), 1), 0.0, 1.0))
        scenario_col = next((c for c in (
            "scenario_id", "scenario", "case_id") if c in df.columns), None)
        scenario_count = (
            int(df[scenario_col].nunique(dropna=True)) if scenario_col else 0)
        scenario_score = float(np.clip(scenario_count / 10.0, 0.0, 1.0))

        required_evidence = ("true_depth", "adaptive_error", "fixed_error")
        present = sum(col in df.columns for col in required_evidence)
        metric_score = present / len(required_evidence)

        if metric_score < 1.0:
            alternative_sets = (
                {"true_depth", "adaptive_abs_error", "fixed_abs_error"},
                {"true_depth", "abs_error_adaptive", "abs_error_fixed"},
                {"reference_depth", "adaptive_error_cm", "fixed_error_cm"})
            metric_score = max(metric_score, max((sum(
                c in df.columns for c in s) / 3.0) for s in alternative_sets))

        breadth = (0.45 * sample_score + 0.35 * scenario_score + 0.20 * metric_score)
        return float(np.clip(breadth, 0.0, 1.0)), {
            "sample_breadth": sample_score,
            "scenario_breadth": scenario_score,
            "evaluation_metric_completeness": float(np.clip(metric_score, 0.0, 1.0)),
            "scenario_count": float(scenario_count),
        }

    @classmethod
    def _component_predictive_accuracy(
        cls, df: pd.DataFrame, summary: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        evaluation = cls._evaluation_frame(df)
        rmse = cls._extract_best_metric(evaluation, summary,
            ("adaptive_rmse_cm", "rmse_adaptive_fusion", "adaptive_rmse", "Mean_Adaptive_RMSE_cm"))
        mae = cls._extract_best_metric(evaluation, summary,
            ("adaptive_mae_cm", "mae_adaptive_fusion", "adaptive_mae", "Mean_Adaptive_MAE_cm"))
        fixed_rmse = cls._extract_best_metric(evaluation, summary,
            ("fixed_rmse_cm", "rmse_fixed_fusion", "fixed_rmse", "Mean_Fixed_RMSE_cm"))
        fixed_mae = cls._extract_best_metric(evaluation, summary,
            ("fixed_mae_cm", "mae_fixed_fusion", "fixed_mae", "Mean_Fixed_MAE_cm"))

        rmse_quality = cls._error_quality_score(rmse)
        mae_quality = cls._error_quality_score(mae)
        rmse_gain = cls._relative_improvement_score(fixed_rmse, rmse)
        mae_gain = cls._relative_improvement_score(fixed_mae, mae)

        absolute_scores = [s for s in (rmse_quality, mae_quality) if s is not None]
        comparative_scores = [s for s in (rmse_gain, mae_gain) if s is not None]

        groups = []
        if absolute_scores: groups.append(float(np.mean(absolute_scores)))
        if comparative_scores: groups.append(float(np.mean(comparative_scores)))
        score = float(np.mean(groups)) if groups else 0.5

        return float(np.clip(score, 0.0, 1.0)), {
            "rmse_cm": 0.0 if rmse is None else float(rmse),
            "mae_cm": 0.0 if mae is None else float(mae),
            "absolute_error_quality": (0.5 if not absolute_scores
                else float(np.mean(absolute_scores))),
            "rmse_improvement_score": 0.5 if rmse_gain is None else rmse_gain,
            "mae_improvement_score": 0.5 if mae_gain is None else mae_gain,
            "comparative_evidence_available": float(bool(comparative_scores)),
        }

    @classmethod
    def _component_uncertainty_calibration(
        cls, df: pd.DataFrame, summary: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        evaluation = cls._evaluation_frame(df)
        coverage = cls._extract_best_metric(evaluation, summary,
            ("adaptive_ci_coverage", "adaptive_ci_coverage_95",
            "ci_coverage_95", "Mean_Adaptive_CI_Coverage"))

        if coverage is not None and coverage > 1.0: coverage /= 100.0
        coverage_score = (
            float(np.clip(1.0 - abs(float(coverage) - 0.95) / 0.95, 0.0, 1.0))
            if coverage is not None else None)

        nll = cls._extract_best_metric(evaluation, summary,
            ("adaptive_nll", "mean_adaptive_nll", "Mean_Adaptive_NLL"))
        fixed_nll = cls._extract_best_metric(evaluation, summary,
            ("fixed_nll", "mean_fixed_nll", "Mean_Fixed_NLL"))
        crps = cls._extract_best_metric(evaluation, summary,
            ("adaptive_crps", "mean_adaptive_crps", "Mean_Adaptive_CRPS"))
        fixed_crps = cls._extract_best_metric(evaluation, summary,
            ("fixed_crps", "mean_fixed_crps", "Mean_Fixed_CRPS"))

        nll_gain = cls._relative_improvement_score(fixed_nll, nll)
        crps_gain = cls._relative_improvement_score(fixed_crps, crps)

        available = []
        if coverage_score is not None: available.append(0.60 * coverage_score)
        if nll_gain is not None: available.append(0.20 * nll_gain)
        if crps_gain is not None: available.append(0.20 * crps_gain)

        if not available:
            score = 0.5
        elif coverage_score is None:
            comparative = [x for x in (nll_gain, crps_gain) if x is not None]
            score = float(np.mean(comparative)) if comparative else 0.5
        else:
            score = (0.60 * coverage_score
                + 0.20 * (0.5 if nll_gain is None else nll_gain)
                + 0.20 * (0.5 if crps_gain is None else crps_gain))

        return float(np.clip(score, 0.0, 1.0)), {
            "ci_coverage_95": 0.0 if coverage is None else float(coverage),
            "ci_coverage_calibration_score": (0.5 if coverage_score is None else coverage_score),
            "nll_improvement_score": 0.5 if nll_gain is None else nll_gain,
            "crps_improvement_score": 0.5 if crps_gain is None else crps_gain,
            "nll_available": float(nll is not None),
            "crps_available": float(crps is not None),
        }

    @classmethod
    def _component_statistical_evidence(
        cls, df: pd.DataFrame, summary: pd.DataFrame,
        stats_df: Optional[pd.DataFrame] = None) -> Tuple[float, Dict[str, float]]:

        source = stats_df if stats_df is not None and not stats_df.empty else None
        if source is not None:
            row_scores = []
            p_scores = []
            effect_scores = []
            direction_scores = []
            ci_direction_scores = []

            def row_value(row: pd.Series, candidates: Sequence[str]) -> Any:
                for name in candidates:
                    if name in row.index:
                        value = row[name]
                        if value is None: continue
                        try:
                            if pd.isna(value): continue
                        except (TypeError, ValueError): pass
                        return value
                return None

            for _, row in source.iterrows():
                p_value = row_value(row, ("holm_adjusted_p_value", "adjusted_p_value",
                    "p_value", "paired_t_p_value", "statistical_p_value"))
                hedges_g = row_value(row, ("hedges_g", "effect_size", "paired_effect_size"))
                mean_difference = row_value(row, ("mean_difference", "improvement_difference",
                    "paired_mean_difference", "mean_error_difference"))
                ci = row_value(row, ("ci95_mean_difference", "confidence_interval_95",
                    "bootstrap_ci95_mean_difference"))

                try: p = float(p_value) if p_value is not None else np.nan
                except (TypeError, ValueError): p = np.nan
                p_score = (float(np.clip(-math.log10(max(min(p, 1.0), 1e-300)) / 4.0, 0.0, 1.0))
                    if np.isfinite(p) else 0.5)

                try: g = float(hedges_g) if hedges_g is not None else np.nan
                except (TypeError, ValueError): g = np.nan
                if np.isfinite(g):
                    magnitude = float(np.clip(abs(g) / 0.8, 0.0, 1.0))
                    effect_score = magnitude if g > 0.0 else 0.0
                else: effect_score = 0.5

                try: diff = float(mean_difference) if mean_difference is not None else np.nan
                except (TypeError, ValueError): diff = np.nan
                direction_score = 1.0 if np.isfinite(diff) and diff > 0.0 else (
                    0.0 if np.isfinite(diff) else 0.5)

                ci_direction_score = 0.5
                if isinstance(ci, (tuple, list, np.ndarray, pd.Series)) and len(ci) == 2:
                    try:
                        low, high = float(ci[0]), float(ci[1])
                        if np.isfinite(low) and np.isfinite(high):
                            if low > 0.0: ci_direction_score = 1.0
                            elif high < 0.0: ci_direction_score = 0.0
                    except (TypeError, ValueError): pass

                row_score = (0.35 * p_score
                    + 0.35 * effect_score
                    + 0.20 * direction_score
                    + 0.10 * ci_direction_score)

                row_scores.append(row_score)
                p_scores.append(p_score)
                effect_scores.append(effect_score)
                direction_scores.append(direction_score)
                ci_direction_scores.append(ci_direction_score)

            if row_scores:
                return float(np.clip(np.mean(row_scores), 0.0, 1.0)), {
                    "p_value_score": float(np.mean(p_scores)),
                    "effect_size_score": float(np.mean(effect_scores)),
                    "improvement_direction_score": float(np.mean(direction_scores)),
                    "ci_direction_score": float(np.mean(ci_direction_scores)),
                    "p_value_available": float(any(col in source.columns for col in (
                        "holm_adjusted_p_value", "adjusted_p_value", "p_value",
                        "paired_t_p_value", "statistical_p_value"))),
                    "effect_size_available": float(any(col in source.columns for col in (
                        "hedges_g", "effect_size", "paired_effect_size"))),
                    "pairwise_comparisons_evaluated": float(len(row_scores)),
                    "holm_adjusted_p_values_used": float("holm_adjusted_p_value" in source.columns),
                    "statistical_source_rows": float(len(source)),
                }

        p_value = cls._extract_best_metric(df, summary, ("holm_adjusted_p_value", "adjusted_p_value",
            "p_value", "paired_t_p_value", "statistical_p_value"))
        hedges_g = cls._extract_best_metric(df, summary, ("hedges_g", "effect_size",
            "paired_effect_size"))
        mean_difference = cls._extract_best_metric(df, summary, ("mean_error_difference",
            "paired_mean_difference", "mean_difference"))
        ci = cls._summary_lookup_raw(summary, ("ci95_mean_difference",
            "bootstrap_ci95_mean_difference", "confidence_interval_95"))

        p_score = (float(np.clip(-math.log10(max(min(
            float(p_value), 1.0), 1e-300)) / 4.0, 0.0, 1.0)) if p_value is not None else 0.5)

        if hedges_g is None:
            effect_score = 0.5
        else:
            magnitude = float(np.clip(abs(float(hedges_g)) / 0.8, 0.0, 1.0))
            effect_score = magnitude if hedges_g > 0.0 else 0.0

        direction_score = 0.5
        if mean_difference is not None:
            direction_score = 1.0 if mean_difference > 0.0 else 0.0

        ci_direction_score = 0.5
        if isinstance(ci, (tuple, list, np.ndarray, pd.Series)) and len(ci) == 2:
            try:
                low, high = float(ci[0]), float(ci[1])
                if np.isfinite(low) and np.isfinite(high):
                    if low > 0.0: ci_direction_score = 1.0
                    elif high < 0.0: ci_direction_score = 0.0
            except (TypeError, ValueError): pass

        score = (0.35 * p_score
            + 0.35 * effect_score
            + 0.20 * direction_score
            + 0.10 * ci_direction_score)

        return float(np.clip(score, 0.0, 1.0)), {
            "p_value_score": p_score,
            "effect_size_score": effect_score,
            "improvement_direction_score": direction_score,
            "ci_direction_score": ci_direction_score,
            "p_value_available": float(p_value is not None),
            "effect_size_available": float(hedges_g is not None),
            "pairwise_comparisons_evaluated": 0.0,
            "holm_adjusted_p_values_used": 0.0,
            "statistical_source_rows": 0.0,
        }

    @classmethod
    def _component_validation_stability(
        cls, df: pd.DataFrame, summary: pd.DataFrame,
        fault_df: Optional[pd.DataFrame] = None) -> Tuple[float, Dict[str, float]]:

        group_candidates = (
            "fault_scenario", "fault_type", "experiment",
            "Experiment", "benchmark_category", "category")
        value_candidates = (
            "adaptive_abs_error", "abs_error_adaptive", "adaptive_error",
            "adaptive_error_cm", "Adaptive_RMSE_cm", "adaptive_rmse_cm", "Adaptive_RMSE")

        scores = []
        source_labels = []
        for name, frame in (("fault", fault_df), ("benchmark", df)):
            if frame is None or frame.empty: continue
            evaluation = cls._evaluation_frame(frame)
            score = cls._group_dispersion_score(evaluation, value_candidates, group_candidates)
            if score is not None:
                scores.append(float(score))
                source_labels.append(name)
        stability = float(np.mean(scores)) if scores else 0.5

        return float(np.clip(stability, 0.0, 1.0)), {
            "cross_condition_error_stability": stability,
            "stability_evidence_available": float(bool(scores)),
            "stability_source_count": float(len(scores)),
            "fault_stability_available": float("fault" in source_labels),
            "benchmark_stability_available": float("benchmark" in source_labels),
        }

    @classmethod
    def _component_hazard_validation(
        cls, df: pd.DataFrame, summary: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        evaluation = cls._evaluation_frame(df)

        brier = cls._extract_best_metric(evaluation, summary,
            ("adaptive_hazard_brier", "hazard_brier", "brier_score", "adaptive_brier_score"))
        accuracy = cls._extract_best_metric(evaluation, summary,
            ("adaptive_hazard_accuracy", "hazard_accuracy",
            "accuracy_hazard", "hazard_accuracy_at_80"))
        ece = cls._extract_best_metric(evaluation, summary,
            ("adaptive_hazard_ece", "hazard_ece", "expected_calibration_error"))

        brier_score = (float(np.clip(1.0 - brier / 0.25, 0.0, 1.0)) if brier is not None else None)
        accuracy_fraction = cls._normalize_fraction(accuracy)
        accuracy_score = accuracy_fraction if accuracy_fraction is not None else None
        ece_score = (float(np.clip(1.0 - ece, 0.0, 1.0)) if ece is not None else None)

        scores = []
        if brier_score is not None: scores.append(("brier", 0.50, brier_score))
        if accuracy_score is not None: scores.append(("accuracy", 0.30, accuracy_score))
        if ece_score is not None: scores.append(("ece", 0.20, ece_score))

        if scores:
            total_weight = sum(weight for _, weight, _ in scores)
            score = sum(weight * value for _, weight, value in scores) / total_weight
        else: score = 0.5

        return float(np.clip(score, 0.0, 1.0)), {
            "brier_score_quality": 0.5 if brier_score is None else brier_score,
            "hazard_accuracy_score": 0.5 if accuracy_score is None else accuracy_score,
            "hazard_ece_score": 0.5 if ece_score is None else ece_score,
            "brier_available": float(brier is not None),
            "accuracy_available": float(accuracy_score is not None),
            "ece_available": float(ece is not None),
        }

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------
    @classmethod
    def compute(
        cls, df: pd.DataFrame,
        summary_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        target_samples: int = DEFAULT_TARGET_SAMPLES,
        return_components: bool = False,
        weights: Optional[Dict[str, float]] = None,
        stats_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        fault_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None) -> Union[float, Dict[str, Any]]:

        if not isinstance(df, pd.DataFrame): raise TypeError("df must be a pandas DataFrame")
        if target_samples <= 0: raise ValueError("target_samples must be greater than zero")

        summary = cls._to_frame(summary_df)
        stats = cls._to_frame(stats_df)
        faults = cls._to_frame(fault_df)

        components = {}
        component_details = {}

        calculators = (
            ("dataset_coverage", cls._component_dataset_coverage),
            ("environment_coverage", cls._component_environment_coverage),
            ("fault_coverage", cls._component_fault_coverage),
            ("benchmark_breadth", cls._component_benchmark_breadth),
            ("predictive_accuracy", cls._component_predictive_accuracy),
            ("uncertainty_calibration", cls._component_uncertainty_calibration),
            ("statistical_evidence", cls._component_statistical_evidence),
            ("validation_stability", cls._component_validation_stability),
            ("hazard_validation", cls._component_hazard_validation))

        for name, calculator in calculators:
            if name in {"dataset_coverage", "benchmark_breadth"}:
                score, details = calculator(df, summary, target_samples)
            elif name in {"environment_coverage"}:
                score, details = calculator(df)
            elif name == "fault_coverage":
                score, details = calculator(df, summary, faults)
            elif name == "statistical_evidence":
                score, details = calculator(df, summary, stats)
            elif name == "validation_stability":
                score, details = calculator(df, summary, faults)
            else:
                score, details = calculator(df, summary)
            components[name] = float(np.clip(score, 0.0, 1.0))
            component_details[name] = details

        final_weights = dict(cls.DEFAULT_WEIGHTS)
        if weights is not None:
            unknown = set(weights) - set(final_weights)
            if unknown: raise ValueError(f"Unknown weight keys: {sorted(unknown)}")
            final_weights.update({key: float(value) for key, value in weights.items()})

        if any(value < 0.0 for value in final_weights.values()):
            raise ValueError("Weights must be non-negative")
        weight_sum = float(sum(final_weights.values()))
        if weight_sum <= 0.0: raise ValueError("At least one score weight must be positive")

        final_weights = {key: value / weight_sum for key, value in final_weights.items()}
        score = float(sum(components[key] * final_weights[key] for key in components))
        score = float(np.clip(score, 0.0, 1.0))
        bands = ValidationEvidenceBands()
        if score >= bands.excellent: label = "Excellent"
        elif score >= bands.strong: label = "Strong"
        elif score >= bands.moderate: label = "Moderate"
        else: label = "Weak"

        detail = {
            "score": score,
            "label": label,
            "components": components,
            "component_details": component_details,
            "weights": final_weights,
            "evidence_sources": {
                "primary_validation_rows": float(len(df)),
                "statistical_rows": float(len(stats)),
                "fault_rows": float(len(faults)),
                "dedicated_statistical_evidence_supplied": bool(not stats.empty),
                "dedicated_fault_evidence_supplied": bool(not faults.empty),
            },

            "interpretation": (
                "Composite Validation Evidence Score. "
                "This is an engineering-defined summary of benchmark "
                "validation breadth and evidence quality; it is not a "
                "probability, statistical confidence level, or probability "
                "that the research is correct."
            )}

        return detail if return_components else score

    @classmethod
    def score_dataframe(
        cls, df: pd.DataFrame,
        summary_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        target_samples: int = DEFAULT_TARGET_SAMPLES,
        weights: Optional[Dict[str, float]] = None,
        stats_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        fault_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None) -> Dict[str, Any]:

        return cls.compute(
            df, summary_df=summary_df,
            target_samples=target_samples,
            return_components=True,
            weights=weights,
            stats_df=stats_df,
            fault_df=fault_df)

    @classmethod
    def export_report(
        cls, df: pd.DataFrame,
        output_path: Union[str, Path],
        summary_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        target_samples: int = DEFAULT_TARGET_SAMPLES,
        weights: Optional[Dict[str, float]] = None,
        stats_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        fault_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None) -> pd.DataFrame:

        report = cls.score_dataframe(
            df, summary_df=summary_df,
            target_samples=target_samples,
            weights=weights,
            stats_df=stats_df,
            fault_df=fault_df)

        row: Dict[str, Any] = {
            "validation_evidence_score": report["score"],
            "validation_evidence_label": report["label"],
        }

        for name, value in report["components"].items(): row[name] = value
        out = pd.DataFrame([row])
        out.to_csv(str(output_path), index=False)
        return out

__all__ = ["ValidationEvidenceBands", "ValidationEvidenceScorer"]