"""Explicit, inspectable baseline opponent models.

These profiles are deliberately simple.  They provide a reproducible baseline
for testing range updates and action EVs; they are not presented as solved or
casino-grade poker strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from pokerkit import Card

from .cards import CardsLike, parse_cards
from .ranges import HandRange, HoleCards, OpponentProfile


_RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


@dataclass(frozen=True)
class StyleParameters:
    name: str
    fold_bias: float
    call_bias: float
    aggression: float
    bluff_frequency: float


STYLES = {
    "balanced": StyleParameters("balanced", 0.30, 0.48, 0.42, 0.08),
    "tight": StyleParameters("tight", 0.52, 0.30, 0.34, 0.03),
    "calling": StyleParameters("calling", 0.13, 0.72, 0.20, 0.03),
    "aggressive": StyleParameters("aggressive", 0.24, 0.32, 0.68, 0.18),
}


def get_style(name: str) -> StyleParameters:
    try:
        return STYLES[name.strip().lower()]
    except (AttributeError, KeyError) as exc:
        choices = ", ".join(sorted(STYLES))
        raise ValueError(f"unknown opponent style {name!r}; choose one of: {choices}") from exc


def _rank(card: Card) -> int:
    return _RANK_VALUE[repr(card)[0]]


def _suit(card: Card) -> str:
    return repr(card)[1]


def visible_hand_strength(combo: HoleCards, board_cards: Iterable[Card] = ()) -> float:
    """Return a cheap public-information strength feature in ``[0, 1]``.

    This is a feature for the baseline likelihood model, not an equity
    calculator.  It considers hole ranks, pairing, board interaction, flush
    density, and rough connectivity without looking at any hidden card.
    """

    first, second = combo
    ranks = sorted((_rank(first), _rank(second)), reverse=True)
    suited = _suit(first) == _suit(second)
    gap = abs(ranks[0] - ranks[1])

    if ranks[0] == ranks[1]:
        strength = 0.48 + 0.035 * ranks[0]
    else:
        strength = 0.04 + 0.026 * ranks[0] + 0.012 * ranks[1]
        if suited:
            strength += 0.06
        if gap <= 1:
            strength += 0.06
        elif gap == 2:
            strength += 0.03

    board = tuple(board_cards)
    if board:
        board_ranks = [_rank(card) for card in board]
        board_suits = [_suit(card) for card in board]
        for hole_rank in ranks:
            matches = board_ranks.count(hole_rank)
            strength += 0.18 * matches
        if ranks[0] == ranks[1]:
            strength += 0.08

        suit_counts = {
            suit: board_suits.count(suit) + sum(_suit(card) == suit for card in combo)
            for suit in "cdhs"
        }
        maximum_suit_count = max(suit_counts.values())
        if maximum_suit_count >= 5:
            strength += 0.24
        elif maximum_suit_count == 4:
            strength += 0.09

        unique_ranks = set(board_ranks + ranks)
        if 14 in unique_ranks:
            unique_ranks.add(1)  # Ace may be low in A-2-3-4-5.
        longest = 1
        run = 1
        ordered = sorted(unique_ranks)
        for previous, current in zip(ordered, ordered[1:]):
            if current == previous + 1:
                run += 1
                longest = max(longest, run)
            elif current != previous:
                run = 1
        if longest >= 5:
            strength += 0.18
        elif longest == 4:
            strength += 0.07

    return max(0.01, min(0.99, strength))


def make_range_profile(
    style_name: str,
    *,
    board_cards: CardsLike | None = (),
    facing_bet: bool,
) -> OpponentProfile:
    """Build observable-action likelihoods for a named baseline style."""

    style = get_style(style_name)
    board = parse_cards(board_cards, field_name="board_cards")

    def strength(combo: HoleCards) -> float:
        return visible_hand_strength(combo, board)

    if facing_bet:
        def probabilities(combo: HoleCards) -> tuple[float, float, float]:
            value = strength(combo)
            raw_fold = 0.02 + style.fold_bias * (1.15 - value)
            raw_call = 0.02 + style.call_bias * (0.30 + 0.70 * value)
            raw_raise = 0.01 + style.bluff_frequency + style.aggression * value * value
            total = raw_fold + raw_call + raw_raise
            return raw_fold / total, raw_call / total, raw_raise / total

        fold_likelihood = lambda combo: probabilities(combo)[0]
        call_likelihood = lambda combo: probabilities(combo)[1]
        raise_likelihood = lambda combo: probabilities(combo)[2]
        check_likelihood: Callable[[HoleCards], float] = lambda combo: 0.0
    else:
        def unopened_probabilities(combo: HoleCards) -> tuple[float, float]:
            value = strength(combo)
            raw_check = 0.05 + (1.05 - style.aggression) * (1.05 - value)
            raw_raise = 0.01 + style.bluff_frequency + style.aggression * (value ** 1.7)
            total = raw_check + raw_raise
            return raw_check / total, raw_raise / total

        check_likelihood = lambda combo: unopened_probabilities(combo)[0]
        raise_likelihood = lambda combo: unopened_probabilities(combo)[1]
        call_likelihood = lambda combo: 0.0
        fold_likelihood = lambda combo: 0.0

    return OpponentProfile(
        f"{style.name}:{'facing_bet' if facing_bet else 'unopened'}",
        {
            "fold": fold_likelihood,
            "check": check_likelihood,
            "call": call_likelihood,
            "raise": raise_likelihood,
            "all_in": raise_likelihood,
        },
    )


def estimate_fold_probability(
    style_name: str,
    *,
    raise_cost: float,
    pot: float,
) -> float:
    """Return a bounded aggregate fold estimate for a proposed raise."""

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
        for value in (raise_cost, pot)
    ):
        raise ValueError("raise_cost and pot must be finite, non-negative numbers")
    style = get_style(style_name)
    pressure = raise_cost / max(pot + raise_cost, 1e-9)
    estimate = style.fold_bias + (0.55 - style.call_bias * 0.35) * pressure
    return max(0.02, min(0.90, estimate))


def range_fold_probability(
    opponent_range: HandRange,
    style_name: str,
    *,
    hero_cards: CardsLike,
    board_cards: CardsLike | None,
    raise_cost: float,
    pot: float,
) -> float:
    """Aggregate ``P(fold|hand, public state)`` over the current range."""

    if not isinstance(opponent_range, HandRange):
        raise TypeError("opponent_range must be a HandRange")
    profile = make_range_profile(
        style_name,
        board_cards=board_cards,
        facing_bet=True,
    )
    dead_cards = (*parse_cards(hero_cards), *parse_cards(board_cards))
    probabilities = opponent_range.probabilities(dead_cards)
    base = math.fsum(
        probability * profile.likelihood("fold", combo)
        for combo, probability in probabilities.items()
    )
    # The explicit profile represents a medium wager.  Move its odds smoothly
    # toward more folds as the amount the opponent must add grows.
    pressure = float(raise_cost) / max(float(pot) + float(raise_cost), 1e-9)
    adjusted = base + (1.0 - base) * 0.55 * pressure
    return max(0.01, min(0.95, adjusted))


__all__ = [
    "STYLES",
    "StyleParameters",
    "estimate_fold_probability",
    "get_style",
    "make_range_profile",
    "range_fold_probability",
    "visible_hand_strength",
]
