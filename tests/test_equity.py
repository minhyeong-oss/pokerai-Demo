from __future__ import annotations

import pytest

from poker_ai.equity import monte_carlo_equity
from poker_ai.ranges import WeightedRange


def test_fixed_river_win_uses_pokerkit_evaluation() -> None:
    result = monte_carlo_equity(
        "As Ad",
        "2c 3d 4h 5s 9c",
        simulations=25,
        seed=7,
        opponent_range=WeightedRange({("Ks", "Kd"): 1}),
    )

    assert (result.wins, result.ties, result.losses) == (25, 0, 0)
    assert result.equity == 1.0
    assert result.confidence_interval_95 == (1.0, 1.0)
    assert result.seed == 7


def test_board_playing_tie_splits_equity_among_ten_players() -> None:
    result = monte_carlo_equity(
        "2c 3d",
        "As Ks Qs Js Ts",
        opponents=9,
        simulations=7,
        seed=99,
    )

    assert (result.wins, result.ties, result.losses) == (0, 7, 0)
    assert result.tie_probability == 1.0
    assert result.equity == pytest.approx(0.1)


def test_same_seed_reproduces_complete_result() -> None:
    kwargs = dict(opponents=3, simulations=80, seed=20261014)

    first = monte_carlo_equity("As Qs", (), **kwargs)
    second = monte_carlo_equity("As Qs", (), **kwargs)

    assert first == second
    assert first.wins + first.ties + first.losses == first.simulations
    assert first.win_probability + first.tie_probability + first.loss_probability == pytest.approx(1)
    assert first.ci95_low <= first.equity <= first.ci95_high


def test_omitted_seed_is_returned_for_replay() -> None:
    first = monte_carlo_equity("As Qs", (), simulations=30)
    replay = monte_carlo_equity("As Qs", (), simulations=30, seed=first.seed)

    assert isinstance(first.seed, int)
    assert first == replay


def test_weighted_heads_up_range_affects_equity_and_filters_blockers() -> None:
    # On this river hero beats KK but loses to 76.  The specified prior makes
    # hero's theoretical equity 25%.
    weighted = WeightedRange({("Ks", "Kd"): 1, ("6c", "7c"): 3})
    result = monte_carlo_equity(
        "As Ad",
        "2c 3d 4h 5s 9c",
        simulations=2_000,
        seed=44,
        opponent_range=weighted,
    )
    assert result.equity == pytest.approx(0.25, abs=0.04)

    # The overwhelmingly weighted combo is illegal because As is visible, so
    # conditioning must leave only KK instead of leaking/sampling the blocker.
    blocked = WeightedRange({("As", "Kh"): 1_000, ("Ks", "Kd"): 1})
    filtered = monte_carlo_equity(
        "As Ad",
        "2c 3d 4h 5s 9c",
        simulations=20,
        seed=44,
        opponent_range=blocked,
    )
    assert filtered.equity == 1.0


@pytest.mark.parametrize("count", [0, 10, -1])
def test_rejects_invalid_opponent_count(count: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 9"):
        monte_carlo_equity("As Qs", opponents=count, simulations=1)


@pytest.mark.parametrize("board", ["2c", "2c 3d"])
def test_rejects_impossible_public_board_lengths(board: str) -> None:
    with pytest.raises(ValueError, match="0, 3, 4, or 5"):
        monte_carlo_equity("As Qs", board, simulations=1)


def test_rejects_card_collision_and_invalid_simulation_count() -> None:
    with pytest.raises(ValueError, match="duplicate card As"):
        monte_carlo_equity("As Qs", "As 2d 3h", simulations=1)
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_equity("As Qs", simulations=0)


def test_weighted_range_is_explicitly_heads_up_only() -> None:
    with pytest.raises(ValueError, match="only heads-up"):
        monte_carlo_equity(
            "As Qs",
            opponents=2,
            simulations=1,
            opponent_range=WeightedRange({("Kd", "Kh"): 1}),
        )


def test_result_dictionary_is_json_friendly() -> None:
    result = monte_carlo_equity(
        "As Ad",
        "2c 3d 4h 5s 9c",
        simulations=1,
        seed=5,
        opponent_range={("Ks", "Kd"): 1},
    )

    payload = result.as_dict()
    assert payload["hero_cards"] == ["As", "Ad"]
    assert payload["board_cards"] == ["2c", "3d", "4h", "5s", "9c"]
    assert payload["confidence_interval_95"] == [1.0, 1.0]
