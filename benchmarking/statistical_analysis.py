from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

@dataclass(frozen=True)
class StatisticalResult:

    n_samples: int
    t_statistic: float
    p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float

    normality_test: str
    normality_statistic: float
    normality_p_value: float

    mean_difference: float
    median_difference: float
    mean_baseline: float
    mean_proposed: float
    improvement_percent: float

    effect_size: float
    effect_size_uncorrected: float
    effect_category: str

    confidence_interval_95: Tuple[float, float]
    bootstrap_mean_difference: float
    bootstrap_ci_95: Tuple[float, float]
    bootstrap_improvement_percent: float
    bootstrap_improvement_ci_95: Tuple[float, float]

    post_hoc_power: float
    evidence_strength: str

class StatisticalEvaluator:

    def __init__(
        self, alpha: float = 0.05,
        bootstrap_samples: int = 5000,
        random_seed: int = 42,
        normality_alpha: float = 0.05,
        confidence_level: float = 0.95) -> None:

        if not 0.0 < alpha < 1.0: raise ValueError("alpha must lie strictly between 0 and 1")
        if not 0.0 < normality_alpha < 1.0:
            raise ValueError("normality_alpha must lie strictly between 0 and 1")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")

        self.alpha = float(alpha)
        self.bootstrap_samples = int(max(500, bootstrap_samples))
        self.random_seed = int(random_seed)
        self.normality_alpha = float(normality_alpha)
        self.confidence_level = float(confidence_level)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float_array(values: Iterable[Any]) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr

    @classmethod
    def _paired_finite_arrays(
        cls, error_baseline: Iterable[Any],
        error_proposed: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        baseline = cls._as_float_array(error_baseline)
        proposed = cls._as_float_array(error_proposed)
        if baseline.shape != proposed.shape:
            raise ValueError("error_baseline and error_proposed must have the same length")
        mask = np.isfinite(baseline) & np.isfinite(proposed)
        baseline_f = baseline[mask]
        proposed_f = proposed[mask]
        if baseline_f.size < 2: raise ValueError("At least two paired finite samples are required")
        return baseline_f, proposed_f, mask

    @staticmethod
    def _safe_mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float("nan")

    @staticmethod
    def _safe_median(x: np.ndarray) -> float:
        return float(np.median(x)) if x.size else float("nan")

    # ------------------------------------------------------------------
    # Effect size
    # ------------------------------------------------------------------
    @staticmethod
    def _cohens_d_paired(diff: np.ndarray) -> float:
        if diff.size < 2: return float("nan")
        sd = float(np.std(diff, ddof=1))
        if not np.isfinite(sd): return float("nan")
        if sd <= 1e-12:
            if abs(float(np.mean(diff))) <= 1e-12: return 0.0
            return float("inf") if np.mean(diff) > 0 else float("-inf")
        return float(np.mean(diff) / sd)

    @staticmethod
    def _hedges_correction(n: int) -> float:
        if n <= 2: return 1.0
        df = n - 1
        return float(1.0 - 3.0 / max(4.0 * df - 1.0, 1.0))

    @classmethod
    def _hedges_g(cls, diff: np.ndarray) -> Tuple[float, float]:
        d = cls._cohens_d_paired(diff)
        correction = cls._hedges_correction(diff.size)
        if not np.isfinite(d):
            if np.isinf(d): return float(d), float(d)
            return float("nan"), float("nan")
        return float(d * correction), float(d)

    @classmethod
    def _effect_size_category(cls, g: float) -> str:
        if not np.isfinite(g): return "Undefined" if np.isnan(g) else "Very Large"
        ag = abs(float(g))
        if ag < 0.2: return "Negligible"
        if ag < 0.5: return "Small"
        if ag < 0.8: return "Medium"
        return "Large"

    # ------------------------------------------------------------------
    # Descriptive evidence classification
    # ------------------------------------------------------------------
    @staticmethod
    def _evidence_strength(p_value: float, effect_size: float) -> str:
        if not np.isfinite(p_value) or not np.isfinite(effect_size): return "Inconclusive"
        p = float(p_value)
        g = abs(float(effect_size))
        if p < 1e-4 and g >= 0.8: return "Very Strong"
        if p < 1e-3 and g >= 0.5: return "Strong"
        if p < 0.01 and g >= 0.2: return "Moderate"
        if p < 0.05: return "Weak"
        return "Inconclusive"

    # ------------------------------------------------------------------
    # Normality diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _normality_test(diff: np.ndarray) -> Tuple[str, float, float]:
        n = int(diff.size)
        if n < 3: return "insufficient_samples", float("nan"), float("nan")
        if n <= 5000:
            try:
                stat, p = stats.shapiro(diff)
                return "shapiro_wilk", float(stat), float(p)
            except Exception: return "shapiro_wilk_failed", float("nan"), float("nan")
        if n >= 8:
            try:
                stat, p = stats.normaltest(diff)
                return "dagostino_pearson", float(stat), float(p)
            except Exception: return "dagostino_pearson_failed", float("nan"), float("nan")
        return "insufficient_samples", float("nan"), float("nan")

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------
    def _t_ci_mean_difference(self, diff: np.ndarray) -> Tuple[float, float]:
        n = diff.size
        mean_diff = float(np.mean(diff))
        if n < 2: return mean_diff, mean_diff
        sd = float(np.std(diff, ddof=1))
        if not np.isfinite(sd): return mean_diff, mean_diff
        sem = sd / np.sqrt(n)
        if sem <= 1e-12: return mean_diff, mean_diff
        alpha_tail = (1.0 - self.confidence_level) / 2.0
        critical = stats.t.ppf(1.0 - alpha_tail, df=n - 1)
        if not np.isfinite(critical): return mean_diff, mean_diff
        margin = float(critical * sem)
        return float(mean_diff - margin), float(mean_diff + margin)

    # ------------------------------------------------------------------
    # Paired bootstrap
    # ------------------------------------------------------------------
    def _bootstrap_ci(self, baseline: np.ndarray, proposed: np.ndarray) -> Tuple[
        float, float, float, Tuple[float, float, float]]:
        rng = np.random.default_rng(self.random_seed)
        n = baseline.size
        indices = np.arange(n)
        boot_means = np.empty(self.bootstrap_samples, dtype=float)
        boot_improvements = np.full(self.bootstrap_samples, np.nan, dtype=float)

        for i in range(self.bootstrap_samples):
            sample_idx = rng.integers(0, n, size=n)
            b = baseline[sample_idx]
            p = proposed[sample_idx]
            diff = b - p
            boot_means[i] = float(np.mean(diff))
            mean_base = float(np.mean(b))
            if abs(mean_base) > 1e-12:
                boot_improvements[i] = (mean_base - float(np.mean(p))) / abs(mean_base) * 100.0

        finite_improvements = boot_improvements[np.isfinite(boot_improvements)]
        diff_low, diff_high = np.percentile(boot_means,
            [(1.0 - self.confidence_level) * 50.0, 100.0 - (1.0 - self.confidence_level) * 50.0])

        if finite_improvements.size:
            imp_mean = float(np.mean(finite_improvements))
            imp_low, imp_high = np.percentile(finite_improvements,
                [(1.0 - self.confidence_level) * 50.0, 100.0 - (1.0 - self.confidence_level) * 50.0])
        else:
            imp_mean = float("nan")
            imp_low = float("nan")
            imp_high = float("nan")

        return (
            float(np.mean(boot_means)),
            float(diff_low), float(diff_high),
            (float(imp_mean), float(imp_low), float(imp_high)))

    # ------------------------------------------------------------------
    # Multiplicity correction
    # ------------------------------------------------------------------
    @staticmethod
    def adjust_pvalues(p_values: Sequence[Any], method: str = "holm") -> np.ndarray:
        p = np.asarray(p_values, dtype=float).reshape(-1)
        out = np.full(p.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(p)
        if not np.any(finite_mask): return out
        valid = np.clip(p[finite_mask], 0.0, 1.0)
        method = str(method).strip().lower()
        if method == "bonferroni":
            out[finite_mask] = np.clip(valid * valid.size, 0.0, 1.0)
            return out

        if method != "holm": raise ValueError("method must be 'holm' or 'bonferroni'")
        m = valid.size
        order = np.argsort(valid)
        sorted_p = valid[order]
        adjusted_sorted = np.empty(m, dtype=float)
        running_max = 0.0

        for rank in range(m):
            multiplier = m - rank
            candidate = min(1.0, multiplier * sorted_p[rank])
            running_max = max(running_max, candidate)
            adjusted_sorted[rank] = running_max

        adjusted_valid = np.empty(m, dtype=float)
        adjusted_valid[order] = adjusted_sorted
        out[finite_mask] = adjusted_valid
        return out

    # ------------------------------------------------------------------
    # Supplementary post-hoc power diagnostic
    # ------------------------------------------------------------------
    @staticmethod
    def _power_estimate_from_effect(effect_size: float, n: int, alpha: float) -> float:
        if n < 2 or not np.isfinite(effect_size): return float("nan")
        if not 0.0 < alpha < 1.0: return float("nan")
        df = n - 1
        ncp = abs(float(effect_size)) * np.sqrt(n)
        try:
            critical = stats.t.ppf(1.0 - alpha / 2.0, df)
            distribution = stats.nct(df, ncp)
            power = distribution.sf(critical) + distribution.cdf(-critical)
            return float(np.clip(power, 0.0, 1.0))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------
    def compute_significance(
        self, error_baseline: Iterable[Any], error_proposed: Iterable[Any],) -> Dict[str, Any]:
        baseline, proposed, pair_mask = self._paired_finite_arrays(error_baseline, error_proposed)
        diff = baseline - proposed
        try:
            t_stat, p_value = stats.ttest_rel(baseline, proposed, nan_policy="omit")
            t_stat = float(t_stat)
            p_value = float(p_value)
        except Exception: t_stat, p_value = float("nan"), float("nan")
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                diff, zero_method="wilcox", alternative="two-sided", method="auto")
            wilcoxon_stat = float(wilcoxon_stat)
            wilcoxon_p = float(wilcoxon_p)
        except Exception: wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
        normality_name, normality_stat, normality_p = self._normality_test(diff)
        mean_baseline = float(np.mean(baseline))
        mean_proposed = float(np.mean(proposed))
        mean_diff = float(np.mean(diff))
        median_diff = float(np.median(diff))

        improvement_pct = (
            mean_diff / abs(mean_baseline) * 100.0 if abs(mean_baseline) > 1e-12 else float("nan"))

        hedges_g, cohens_d = self._hedges_g(diff)
        effect_category = self._effect_size_category(hedges_g)
        ci_low, ci_high = self._t_ci_mean_difference(diff)

        (bootstrap_mean_diff, boot_ci_low, boot_ci_high,
        boot_imp_pack) = self._bootstrap_ci(baseline, proposed)
        boot_imp_mean, boot_imp_low, boot_imp_high = boot_imp_pack

        post_hoc_power = self._power_estimate_from_effect(
            effect_size=hedges_g, n=diff.size, alpha=self.alpha)
        evidence_strength = self._evidence_strength(
            p_value=p_value, effect_size=hedges_g)

        result = StatisticalResult(
            n_samples=int(diff.size),
            t_statistic=t_stat,
            p_value=p_value,
            wilcoxon_statistic=wilcoxon_stat,
            wilcoxon_p_value=wilcoxon_p,
            normality_test=normality_name,
            normality_statistic=normality_stat,
            normality_p_value=normality_p,
            mean_difference=mean_diff,
            median_difference=median_diff,
            mean_baseline=mean_baseline,
            mean_proposed=mean_proposed,
            improvement_percent=float(improvement_pct),
            effect_size=float(hedges_g),
            effect_size_uncorrected=float(cohens_d),
            effect_category=effect_category,
            confidence_interval_95=(float(ci_low), float(ci_high)),
            bootstrap_mean_difference=float(bootstrap_mean_diff),
            bootstrap_ci_95=(float(boot_ci_low), float(boot_ci_high)),
            bootstrap_improvement_percent=float(boot_imp_mean),
            bootstrap_improvement_ci_95=(float(boot_imp_low), float(boot_imp_high)),
            post_hoc_power=float(post_hoc_power),
            evidence_strength=evidence_strength)

        result_dict: Dict[str, Any] = {
            "n_samples": result.n_samples,
            "n_input_pairs": int(pair_mask.size),
            "n_discarded_pairs": int(pair_mask.size - np.sum(pair_mask)),
            "pair_retention_rate": float(np.mean(pair_mask)),

            "mean_baseline_error": result.mean_baseline,
            "mean_proposed_error": result.mean_proposed,
            "mean_error_difference": result.mean_difference,
            "median_error_difference": result.median_difference,
            "improvement_percent": result.improvement_percent,

            "t_statistic": result.t_statistic,
            "p_value": result.p_value,
            "ci95_mean_difference": result.confidence_interval_95,

            "wilcoxon_statistic": result.wilcoxon_statistic,
            "wilcoxon_p_value": result.wilcoxon_p_value,

            "normality_test": result.normality_test,
            "normality_statistic": result.normality_statistic,
            "normality_p_value": result.normality_p_value,
            "normality_rejects_at_alpha": bool(np.isfinite(
                result.normality_p_value) and result.normality_p_value < self.normality_alpha),

            "hedges_g": result.effect_size,
            "cohens_d_paired": result.effect_size_uncorrected,
            "effect_category": result.effect_category,

            "bootstrap_mean_difference": result.bootstrap_mean_difference,
            "bootstrap_ci95_mean_difference": result.bootstrap_ci_95,
            "bootstrap_improvement_percent": result.bootstrap_improvement_percent,
            "bootstrap_ci95_improvement_percent": (result.bootstrap_improvement_ci_95),

            "post_hoc_power": result.post_hoc_power,
            "evidence_strength": result.evidence_strength,

            "is_significant": bool(np.isfinite(result.p_value) and result.p_value < self.alpha),
            "wilcoxon_is_significant": bool(np.isfinite(
                result.wilcoxon_p_value) and result.wilcoxon_p_value < self.alpha),
            "has_nontrivial_effect": bool(np.isfinite(result.effect_size) and abs(
                result.effect_size) >= 0.2 and result.improvement_percent > 0.0),
            "ci_supports_improvement": bool(np.isfinite(ci_low) and ci_low > 0.0),
            "bootstrap_ci_supports_improvement": bool(np.isfinite(boot_ci_low) and boot_ci_low > 0.0),
        }
        return result_dict

    # ------------------------------------------------------------------
    # Summary formatting
    # ------------------------------------------------------------------
    def summary_text(self, error_baseline: Iterable[Any], error_proposed: Iterable[Any]) -> str:
        out = self.compute_significance(error_baseline, error_proposed)
        p_value = out["p_value"]
        p_text = f"{p_value:.3e}" if np.isfinite(p_value) else "nan"
        improvement = out["improvement_percent"]
        improvement_text = (f"{improvement:.2f}%" if np.isfinite(improvement) else "nan")
        power = out["post_hoc_power"]
        power_text = f"{power:.3f}" if np.isfinite(power) else "nan"

        return (
            f"n={out['n_samples']}, "
            f"p={p_text}, "
            f"Wilcoxon p={out['wilcoxon_p_value']:.3e}, "
            f"g={out['hedges_g']:.3f}, "
            f"improvement={improvement_text}, "
            f"post_hoc_power={power_text}, "
            f"evidence={out['evidence_strength']}")

def compute_significance(
    error_baseline: Iterable[Any], error_proposed: Iterable[Any]) -> Dict[str, Any]:
    return StatisticalEvaluator().compute_significance(error_baseline, error_proposed)
__all__ = ["StatisticalResult", "StatisticalEvaluator", "compute_significance"]