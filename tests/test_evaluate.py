import pytest

from poker_ai.evaluate import (
    paired_seat_schedule,
    summarize_paired_payoffs,
    summarize_payoffs,
)


def test_bb_per_100_and_ci() -> None:
    report = summarize_payoffs((1.0, -0.5, 0.5, 1.0))
    assert report.hands == 4
    assert report.total_profit_bb == 2.0
    assert report.bb_per_100 == 50.0
    assert report.ci95_bb_per_100_low < 50.0 < report.ci95_bb_per_100_high


def test_paired_schedule_reuses_seed_after_seat_swap() -> None:
    assert paired_seat_schedule(6, 10) == (
        (10, 0),
        (10, 1),
        (11, 0),
        (11, 1),
        (12, 0),
        (12, 1),
    )


def test_paired_ci_uses_seed_pairs_as_independent_samples() -> None:
    report = summarize_paired_payoffs((10.0, -6.0, -2.0, 4.0))
    assert report.hands == 4
    assert report.independent_samples == 2
    assert report.mean_profit_bb == 1.5
    assert report.confidence_method == "normal_seed_pair_clusters"


def test_single_sample_ci_is_unavailable() -> None:
    report = summarize_payoffs((1.0,))
    assert report.standard_error_bb is None
    assert report.ci95_bb_per_100_low is None


def test_paired_schedule_rejects_unpaired_hand_count() -> None:
    with pytest.raises(ValueError, match="even"):
        paired_seat_schedule(5, 10)
