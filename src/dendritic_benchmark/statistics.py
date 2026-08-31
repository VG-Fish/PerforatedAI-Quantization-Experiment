"""Seed-paired statistics for dendrite-effect claims.

One seed cannot separate a dendritic arm from its dense control: the August 30
effect audit found per-run validation noise of the same magnitude as several
reported "wins".  Every claim therefore has to be built from paired seeds, and
the reporting layer has to be able to say *insufficient evidence* instead of
silently publishing a one-seed difference.

This module owns that policy.  It takes per-seed metric observations, pairs
them by seed, and returns an estimate that always carries the seed count, the
control's own spread (the noise floor), and an explicit verdict.  Nothing here
touches the filesystem or the plotting layer; ``results`` supplies the records.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .specs import MetricDirection, condition_by_key, model_by_key

#: A dendrite effect is never claimable from fewer paired seeds than this.
MINIMUM_PAIRED_SEEDS = 3

#: Two-sided significance level used for the reportable/within-noise verdict.
SIGNIFICANCE_LEVEL = 0.05

#: Difference below which two float metrics count as the same measurement.
METRIC_TOLERANCE = 1e-12

VERDICT_INSUFFICIENT_SEEDS = "insufficient_seeds"
VERDICT_WITHIN_NOISE = "within_noise"
VERDICT_SUPPORTED = "supported"
VERDICT_CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class SeedObservation:
    """One completed arm of a paired comparison."""

    seed: int
    metric_value: float


@dataclass(frozen=True)
class EffectEstimate:
    """A seed-paired dendrite-effect estimate with its own evidence gate."""

    model_key: str
    condition_key: str
    baseline_condition_key: str
    metric_name: str
    metric_direction: MetricDirection
    paired_seeds: tuple[int, ...]
    baseline_mean: float
    treatment_mean: float
    mean_improvement: float
    improvement_std: float
    noise_floor: float
    t_statistic: float | None
    p_value: float | None
    verdict: str
    reason: str

    @property
    def seed_count(self) -> int:
        return len(self.paired_seeds)

    @property
    def exceeds_noise_floor(self) -> bool:
        """Whether the effect is larger than the control's own seed spread."""
        return abs(self.mean_improvement) > self.noise_floor + METRIC_TOLERANCE

    @property
    def claimable(self) -> bool:
        return self.verdict == VERDICT_SUPPORTED

    def to_row(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "condition_key": self.condition_key,
            "baseline_condition_key": self.baseline_condition_key,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "paired_seeds": ",".join(str(seed) for seed in self.paired_seeds),
            "seed_count": self.seed_count,
            "minimum_paired_seeds": MINIMUM_PAIRED_SEEDS,
            "baseline_mean": self.baseline_mean,
            "treatment_mean": self.treatment_mean,
            "mean_improvement": self.mean_improvement,
            "improvement_std": self.improvement_std,
            "noise_floor": self.noise_floor,
            "exceeds_noise_floor": self.exceeds_noise_floor,
            "t_statistic": "" if self.t_statistic is None else self.t_statistic,
            "p_value": "" if self.p_value is None else self.p_value,
            "significance_level": SIGNIFICANCE_LEVEL,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_continued_fraction(
    a: float, b: float, x: float, *, iterations: int = 300, epsilon: float = 1e-14
) -> float:
    """Lentz's continued fraction for the incomplete beta function."""
    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` — enough of the beta family for a Student-t tail."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("beta parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def two_sided_t_p_value(t_statistic: float, degrees_of_freedom: int) -> float:
    """Two-sided p-value for a Student-t statistic."""
    if degrees_of_freedom < 1:
        raise ValueError("a t-test needs at least one degree of freedom")
    if math.isinf(t_statistic):
        return 0.0
    if math.isnan(t_statistic):
        return 1.0
    df = float(degrees_of_freedom)
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t_statistic * t_statistic))


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def paired_seed_values(
    baseline: Iterable[SeedObservation], treatment: Iterable[SeedObservation]
) -> tuple[tuple[int, ...], list[float], list[float]]:
    """Pair two arms on their seeds, dropping seeds only one arm completed."""
    baseline_by_seed = {observation.seed: observation for observation in baseline}
    treatment_by_seed = {observation.seed: observation for observation in treatment}
    seeds = tuple(sorted(set(baseline_by_seed) & set(treatment_by_seed)))
    baseline_values = [baseline_by_seed[seed].metric_value for seed in seeds]
    treatment_values = [treatment_by_seed[seed].metric_value for seed in seeds]
    return seeds, baseline_values, treatment_values


def _signed_improvements(
    baseline_values: Sequence[float],
    treatment_values: Sequence[float],
    direction: MetricDirection,
) -> list[float]:
    sign = 1.0 if direction == "maximize" else -1.0
    return [
        sign * (treatment - control)
        for control, treatment in zip(baseline_values, treatment_values, strict=True)
    ]


def estimate_effect(
    *,
    model_key: str,
    condition_key: str,
    baseline_condition_key: str,
    metric_name: str,
    metric_direction: MetricDirection,
    baseline: Iterable[SeedObservation],
    treatment: Iterable[SeedObservation],
) -> EffectEstimate:
    """Estimate one dendritic arm's effect against its dense control.

    ``mean_improvement`` is always signed so that a positive value means the
    dendritic arm did better, whichever way the metric points.
    """
    seeds, baseline_values, treatment_values = paired_seed_values(baseline, treatment)
    baseline_mean = sum(baseline_values) / len(baseline_values) if baseline_values else math.nan
    treatment_mean = (
        sum(treatment_values) / len(treatment_values) if treatment_values else math.nan
    )
    noise_floor = _sample_std(baseline_values)
    improvements = _signed_improvements(baseline_values, treatment_values, metric_direction)
    mean_improvement = sum(improvements) / len(improvements) if improvements else math.nan
    improvement_std = _sample_std(improvements)

    if len(seeds) < MINIMUM_PAIRED_SEEDS:
        return EffectEstimate(
            model_key=model_key,
            condition_key=condition_key,
            baseline_condition_key=baseline_condition_key,
            metric_name=metric_name,
            metric_direction=metric_direction,
            paired_seeds=seeds,
            baseline_mean=baseline_mean,
            treatment_mean=treatment_mean,
            mean_improvement=mean_improvement,
            improvement_std=improvement_std,
            noise_floor=noise_floor,
            t_statistic=None,
            p_value=None,
            verdict=VERDICT_INSUFFICIENT_SEEDS,
            reason=(
                f"{len(seeds)} paired seed(s); {MINIMUM_PAIRED_SEEDS} are required "
                "before a dendrite effect may be claimed"
            ),
        )

    degrees_of_freedom = len(seeds) - 1
    if math.isclose(improvement_std, 0.0, abs_tol=METRIC_TOLERANCE):
        # Identical differences on every seed: either no effect at all, or a
        # perfectly repeated one. Neither case may divide by a zero spread.
        if math.isclose(mean_improvement, 0.0, abs_tol=METRIC_TOLERANCE):
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = math.inf if mean_improvement > 0 else -math.inf
            p_value = 0.0
    else:
        t_statistic = mean_improvement / (improvement_std / math.sqrt(len(seeds)))
        p_value = two_sided_t_p_value(t_statistic, degrees_of_freedom)

    if p_value >= SIGNIFICANCE_LEVEL:
        verdict = VERDICT_WITHIN_NOISE
        reason = (
            f"paired difference is not separable from seed noise "
            f"(p={p_value:.4f} >= {SIGNIFICANCE_LEVEL})"
        )
    elif mean_improvement > 0:
        verdict = VERDICT_SUPPORTED
        reason = f"dendritic arm is better on {len(seeds)} paired seeds (p={p_value:.4f})"
    else:
        verdict = VERDICT_CONTRADICTED
        reason = f"dendritic arm is worse on {len(seeds)} paired seeds (p={p_value:.4f})"

    return EffectEstimate(
        model_key=model_key,
        condition_key=condition_key,
        baseline_condition_key=baseline_condition_key,
        metric_name=metric_name,
        metric_direction=metric_direction,
        paired_seeds=seeds,
        baseline_mean=baseline_mean,
        treatment_mean=treatment_mean,
        mean_improvement=mean_improvement,
        improvement_std=improvement_std,
        noise_floor=noise_floor,
        t_statistic=t_statistic,
        p_value=p_value,
        verdict=verdict,
        reason=reason,
    )


def dense_control_key(condition_key: str) -> str | None:
    """Return the dense condition a dendritic condition must be paired with."""
    if not condition_key.startswith("dendrites_"):
        return None
    return "base_" + condition_key[len("dendrites_") :]


def group_seed_observations(
    observations: Iterable[tuple[str, str, SeedObservation]],
) -> dict[tuple[str, str], list[SeedObservation]]:
    """Collect ``(model_key, condition_key, observation)`` triples by arm."""
    grouped: dict[tuple[str, str], list[SeedObservation]] = {}
    for model_key, condition_key, observation in observations:
        grouped.setdefault((model_key, condition_key), []).append(observation)
    for values in grouped.values():
        values.sort(key=lambda observation: observation.seed)
    return grouped


def estimate_all_effects(
    grouped: dict[tuple[str, str], list[SeedObservation]],
) -> list[EffectEstimate]:
    """Estimate every dendritic arm that has a dense control in ``grouped``."""
    estimates: list[EffectEstimate] = []
    for (model_key, condition_key), treatment in sorted(grouped.items()):
        control_key = dense_control_key(condition_key)
        if control_key is None:
            continue
        baseline = grouped.get((model_key, control_key))
        if baseline is None:
            continue
        try:
            model_spec = model_by_key(model_key)
            condition_by_key(condition_key)
        except KeyError:
            continue
        estimates.append(
            estimate_effect(
                model_key=model_key,
                condition_key=condition_key,
                baseline_condition_key=control_key,
                metric_name=model_spec.metric_name,
                metric_direction=model_spec.metric_direction,
                baseline=baseline,
                treatment=treatment,
            )
        )
    return estimates
