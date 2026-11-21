from __future__ import annotations

import random

import pytest

from poker_ai.cards import card_code
from poker_ai.ranges import (
    OpponentProfile,
    UniformRange,
    WeightedRange,
    bayes_update,
    canonical_combo,
)


def test_uniform_range_has_all_combinations_and_conditions_on_blockers() -> None:
    hand_range = UniformRange()

    assert len(hand_range) == 1326
    assert sum(hand_range.weights.values()) == pytest.approx(1.0)

    conditioned = hand_range.probabilities("As Kd")
    assert len(conditioned) == 1225  # C(50, 2)
    assert sum(conditioned.values()) == pytest.approx(1.0)
    assert all(card_code(card) not in {"As", "Kd"} for combo in conditioned for card in combo)


def test_weighted_range_normalizes_and_combines_reversed_combo() -> None:
    hand_range = WeightedRange(
        [
            (("As", "Kd"), 1.0),
            (("Kd", "As"), 2.0),
            (("Qc", "Qd"), 1.0),
            (("2c", "3c"), 0.0),
        ],
    )

    assert len(hand_range) == 2
    assert hand_range.weights[canonical_combo(("As", "Kd"))] == pytest.approx(0.75)
    assert hand_range.weights[canonical_combo(("Qc", "Qd"))] == pytest.approx(0.25)


def test_weighted_sampling_respects_dead_cards_and_seed() -> None:
    hand_range = WeightedRange({("As", "Kd"): 99, ("Qc", "Qd"): 1})
    first_rng = random.Random(17)
    second_rng = random.Random(17)

    first = [hand_range.sample(first_rng, "As") for _ in range(10)]
    second = [hand_range.sample(second_rng, "As") for _ in range(10)]

    assert first == second
    assert set(first) == {canonical_combo(("Qc", "Qd"))}


def test_bayes_update_uses_action_likelihood_and_normalizes() -> None:
    aces = canonical_combo(("As", "Ah"))
    kings = canonical_combo(("Ks", "Kh"))
    prior = WeightedRange({aces: 1, kings: 1})
    profile = OpponentProfile(
        "tight",
        {"raise": {aces: 0.8, kings: 0.2}, "fold": lambda combo: 0.1},
    )

    posterior = bayes_update(prior, "RAISE", profile)

    assert posterior.weights[aces] == pytest.approx(0.8)
    assert posterior.weights[kings] == pytest.approx(0.2)


def test_bayes_update_removes_public_blocked_combos() -> None:
    blocked = canonical_combo(("As", "Ah"))
    possible = canonical_combo(("Ks", "Kh"))
    prior = WeightedRange({blocked: 1, possible: 1})
    profile = OpponentProfile("constant", {"call": 0.5})

    posterior = bayes_update(prior, "call", profile, dead_cards="As 2d 3h")

    assert posterior.weights == {possible: 1.0}


def test_profile_rejects_invalid_or_impossible_likelihoods() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        OpponentProfile("bad", {"raise": 1.1})

    prior = WeightedRange({("As", "Ah"): 1, ("Ks", "Kh"): 1})
    impossible = OpponentProfile("never", {"raise": 0.0})
    with pytest.raises(ValueError, match="zero total likelihood"):
        bayes_update(prior, "raise", impossible)
    with pytest.raises(KeyError, match="no likelihood"):
        bayes_update(prior, "call", OpponentProfile("known", {"raise": 0.5}))


def test_range_rejects_invalid_combos_and_weights() -> None:
    with pytest.raises(ValueError):
        WeightedRange({("As", "As"): 1})
    with pytest.raises(ValueError):
        WeightedRange({("As", "Kd"): -1})
    with pytest.raises(ValueError):
        WeightedRange({("As", "Kd"): 0})
