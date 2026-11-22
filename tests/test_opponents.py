import pytest
from pokerkit import Card

from poker_ai.actions import CandidateAction
from poker_ai.opponents import (
    OpponentActionModel,
    STYLES,
    make_range_profile,
    visible_hand_strength,
)
from poker_ai.ranges import canonical_combo
from poker_ai.seeding import action_candidate_id


FACING = (
    CandidateAction("fold"),
    CandidateAction("call"),
    CandidateAction("raise", 500),
    CandidateAction("raise", 1000),
    CandidateAction("raise", 2000),
)


def test_style_parameter_table_matches_prd() -> None:
    assert (STYLES["balanced"].f0, STYLES["balanced"].c0, STYLES["balanced"].r0, STYLES["balanced"].b) == (1, 0, -1, 0)
    assert (STYLES["tight"].f0, STYLES["tight"].c0, STYLES["tight"].r0, STYLES["tight"].b) == (2, -0.5, -2, -0.5)
    assert (STYLES["calling"].f0, STYLES["calling"].c0, STYLES["calling"].r0, STYLES["calling"].b) == (-1, 2, -2, -0.5)
    assert (STYLES["aggressive"].f0, STYLES["aggressive"].c0, STYLES["aggressive"].r0, STYLES["aggressive"].b) == (0, 0, 1, 0.5)
    assert all(style.epsilon == 0.01 for style in STYLES.values())


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        (("As", "Ah"), 1.0),
        (("As", "Ks"), 0.8061728395061728),
        (("2s", "2h"), 0.25925925925925924),
        (("Js", "Td"), 0.5617283950617283),
        (("7s", "2d"), 0.15432098765432098),
    ],
)
def test_preflop_strength_matches_prd_values(cards, expected) -> None:
    assert visible_hand_strength(canonical_combo(cards)) == pytest.approx(expected)


def test_postflop_strength_uses_category_and_tiebreak_vector() -> None:
    board = tuple(Card.parse("QsJsTs2d3c"))
    straight_flush = visible_hand_strength(canonical_combo(("As", "Ks")), board)
    one_pair = visible_hand_strength(canonical_combo(("Qh", "7h")), board)
    assert 0 <= one_pair < straight_flush < 1


def test_wheel_straight_tiebreak_uses_five_as_high_card() -> None:
    board = tuple(Card.parse("2c3d4h9sKd"))
    wheel = visible_hand_strength(canonical_combo(("As", "5s")), board)
    six_high = visible_hand_strength(canonical_combo(("5h", "6s")), board)
    assert six_high > wheel


@pytest.mark.parametrize("style", ("balanced", "tight", "calling", "aggressive"))
def test_candidate_probabilities_are_positive_normalized_and_legal(style) -> None:
    model = OpponentActionModel(style)
    probabilities = model.probabilities(
        canonical_combo(("As", "Ah")),
        board_cards=(),
        pot=1000,
        call_cost=200,
        own_street_contribution=100,
        candidate_actions=FACING,
    )
    assert tuple(probabilities) == FACING
    assert all(probability > 0 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_raise_size_softmax_changes_direction_with_hand_strength() -> None:
    model = OpponentActionModel("balanced")
    strong = model.probabilities(
        canonical_combo(("As", "Ah")),
        board_cards=(), pot=1000, call_cost=200,
        own_street_contribution=100, candidate_actions=FACING,
    )
    weak = model.probabilities(
        canonical_combo(("7s", "2d")),
        board_cards=(), pot=1000, call_cost=200,
        own_street_contribution=100, candidate_actions=FACING,
    )
    assert strong[CandidateAction("raise", 2000)] > strong[CandidateAction("raise", 500)]
    assert weak[CandidateAction("raise", 2000)] < weak[CandidateAction("raise", 500)]


def test_random_model_is_uniform_over_exact_candidates() -> None:
    probabilities = OpponentActionModel("random").probabilities(
        canonical_combo(("As", "Ah")),
        board_cards=(), pot=1000, call_cost=200,
        own_street_contribution=100, candidate_actions=FACING,
    )
    assert all(
        probability == pytest.approx(1 / len(FACING))
        for probability in probabilities.values()
    )


def test_range_profile_uses_exact_raise_size_likelihood() -> None:
    profile = make_range_profile(
        "aggressive",
        board_cards=(),
        pot=1000,
        call_cost=200,
        own_street_contribution=100,
        candidate_actions=FACING,
    )
    combo = canonical_combo(("As", "Ah"))
    total = sum(profile.likelihood(action_candidate_id(action), combo) for action in FACING)
    assert total == pytest.approx(1.0)
    assert profile.likelihood("raise:2000", combo) != profile.likelihood("raise:500", combo)
