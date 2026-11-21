"""Performance statistics for reset-stack practice matches."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean, stdev
from typing import Iterable


@dataclass(frozen=True)
class EvaluationReport:
    hands: int
    independent_samples: int
    total_profit_bb: float
    mean_profit_bb: float
    bb_per_100: float
    standard_error_bb: float | None
    ci95_mean_low_bb: float | None
    ci95_mean_high_bb: float | None
    ci95_bb_per_100_low: float | None
    ci95_bb_per_100_high: float | None
    confidence_method: str

    def as_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def _validated_payoffs(payoffs_bb: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in payoffs_bb)
    if not values:
        raise ValueError("at least one payoff is required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("payoffs must be finite")
    return values


def _build_report(
    hand_values: tuple[float, ...],
    independent_values: tuple[float, ...],
    *,
    method: str,
) -> EvaluationReport:
    count = len(hand_values)
    total = math.fsum(hand_values)
    mean = fmean(hand_values)
    if len(independent_values) >= 2:
        standard_error = stdev(independent_values) / math.sqrt(len(independent_values))
        margin = 1.959963984540054 * standard_error
        mean_low: float | None = mean - margin
        mean_high: float | None = mean + margin
        bb100_low: float | None = 100.0 * mean_low
        bb100_high: float | None = 100.0 * mean_high
    else:
        standard_error = None
        mean_low = mean_high = bb100_low = bb100_high = None
    return EvaluationReport(
        hands=count,
        independent_samples=len(independent_values),
        total_profit_bb=total,
        mean_profit_bb=mean,
        bb_per_100=100.0 * mean,
        standard_error_bb=standard_error,
        ci95_mean_low_bb=mean_low,
        ci95_mean_high_bb=mean_high,
        ci95_bb_per_100_low=bb100_low,
        ci95_bb_per_100_high=bb100_high,
        confidence_method=method,
    )


def summarize_payoffs(payoffs_bb: Iterable[float]) -> EvaluationReport:
    """Summarize independent per-hand payoffs with a normal 95% CI."""

    values = _validated_payoffs(payoffs_bb)
    return _build_report(values, values, method="normal_independent_hands")


def summarize_paired_payoffs(payoffs_bb: Iterable[float]) -> EvaluationReport:
    """Summarize seat-swapped pairs, treating each deck seed as one sample."""

    values = _validated_payoffs(payoffs_bb)
    if len(values) % 2:
        raise ValueError("paired evaluation requires an even number of payoffs")
    pair_means = tuple(
        (values[index] + values[index + 1]) / 2.0
        for index in range(0, len(values), 2)
    )
    return _build_report(values, pair_means, method="normal_seed_pair_clusters")


def paired_seat_schedule(hands: int, base_seed: int = 0) -> tuple[tuple[int, int], ...]:
    """Return ``(seed, hero_seat)`` pairs, alternating seats per shuffled deck."""

    if isinstance(hands, bool) or not isinstance(hands, int) or hands <= 0:
        raise ValueError("hands must be a positive integer")
    if hands % 2:
        raise ValueError("paired seat evaluation requires an even number of hands")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    return tuple((base_seed + index // 2, index % 2) for index in range(hands))


__all__ = [
    "EvaluationReport",
    "paired_seat_schedule",
    "summarize_paired_payoffs",
    "summarize_payoffs",
]
