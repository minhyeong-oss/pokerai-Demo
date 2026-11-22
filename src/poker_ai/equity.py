"""Monte Carlo showdown equity using PokerKit's standard hand evaluator."""

from __future__ import annotations

import math
import random
import secrets
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from pokerkit import Card, StandardHighHand

from .cards import CardsLike, STANDARD_DECK, card_code, parse_cards, validate_unique_cards
from .ranges import HandRange, HoleCards, WeightedRange

_LEGAL_BOARD_COUNTS = frozenset((0, 3, 4, 5))
_Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Reproducible aggregate result from one Monte Carlo calculation."""

    hero_cards: tuple[str, str]
    board_cards: tuple[str, ...]
    opponents: int
    simulations: int
    wins: int
    ties: int
    losses: int
    win_probability: float
    tie_probability: float
    loss_probability: float
    equity: float
    ci95_low: float
    ci95_high: float
    seed: int

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """Return the 95% normal confidence interval for split-pot equity."""

        return self.ci95_low, self.ci95_high

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        data = asdict(self)
        data["hero_cards"] = list(self.hero_cards)
        data["board_cards"] = list(self.board_cards)
        data["confidence_interval_95"] = [self.ci95_low, self.ci95_high]
        return data


def monte_carlo_equity(
    hero_cards: CardsLike,
    board_cards: CardsLike | None = (),
    *,
    opponents: int = 1,
    simulations: int = 10_000,
    seed: int | None = None,
    opponent_range: HandRange | Mapping[object, float] | None = None,
) -> EquityResult:
    """Estimate hero showdown equity against one to nine opponents.

    Only hero cards, public board cards, opponent count, and an optional range
    distribution enter this function.  No actual hidden opponent holding is an
    accepted input.  With no range supplied, every legal joint deal is sampled
    uniformly without replacement.  A weighted range is supported for the
    heads-up case and is conditioned on hero/public blockers.

    Args:
        hero_cards: Exactly two private cards visible to the hero.
        board_cards: Zero (preflop), three, four, or five public cards.
        opponents: Number of active opponents, from one through nine.
        simulations: Positive number of independent deals to sample.
        seed: Optional integer RNG seed.  A secure random seed is generated and
            returned when omitted, making every result reproducible afterward.
        opponent_range: Optional heads-up distribution over opponent holdings.

    Returns:
        Counts, probabilities, split-pot equity, its approximate 95% confidence
        interval, and the exact RNG seed used.
    """

    hero = parse_cards(hero_cards, expected_count=2, field_name="hero_cards")
    board = parse_cards(board_cards, field_name="board_cards")
    if len(board) not in _LEGAL_BOARD_COUNTS:
        raise ValueError(
            "board_cards must contain 0, 3, 4, or 5 cards for a Hold'em decision point",
        )
    validate_unique_cards(hero, board, field_names=("hero_cards", "board_cards"))

    if isinstance(opponents, bool) or not isinstance(opponents, int):
        raise TypeError("opponents must be an integer")
    if not 1 <= opponents <= 9:
        raise ValueError("opponents must be between 1 and 9")
    if isinstance(simulations, bool) or not isinstance(simulations, int):
        raise TypeError("simulations must be an integer")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")

    actual_seed = secrets.randbits(64) if seed is None else seed
    rng = random.Random(actual_seed)

    if isinstance(opponent_range, Mapping):
        opponent_range = WeightedRange(opponent_range)
    if opponent_range is not None:
        if not isinstance(opponent_range, HandRange):
            raise TypeError("opponent_range must be a HandRange or combo-to-weight mapping")
        if opponents != 1:
            raise ValueError("a weighted opponent_range is currently supported only heads-up")

    known = frozenset((*hero, *board))
    remaining = tuple(card for card in STANDARD_DECK if card not in known)
    missing_board_count = 5 - len(board)
    required_unknowns = opponents * 2 + missing_board_count
    if required_unknowns > len(remaining):
        raise ValueError("not enough unknown cards remain for the requested deal")

    range_sampler = _prepare_range_sampler(opponent_range, known) if opponent_range else None

    wins = 0
    ties = 0
    losses = 0
    equity_sum = 0.0
    equity_square_sum = 0.0

    for _ in range(simulations):
        if range_sampler is None:
            dealt = rng.sample(remaining, required_unknowns)
            opponent_hands = tuple(
                (dealt[index], dealt[index + 1])
                for index in range(0, opponents * 2, 2)
            )
            runout = tuple(dealt[opponents * 2 :])
        else:
            opponent_hand = range_sampler(rng)
            opponent_hands = (opponent_hand,)
            available_runout = tuple(card for card in remaining if card not in opponent_hand)
            runout = tuple(rng.sample(available_runout, missing_board_count))

        complete_board = (*board, *runout)
        hero_hand = StandardHighHand.from_game(hero, complete_board)
        villain_hands = tuple(
            StandardHighHand.from_game(hole_cards, complete_board)
            for hole_cards in opponent_hands
        )
        best_villain = max(villain_hands)

        if hero_hand > best_villain:
            wins += 1
            share = 1.0
        elif hero_hand < best_villain:
            losses += 1
            share = 0.0
        else:
            tied_villains = sum(hand == hero_hand for hand in villain_hands)
            ties += 1
            share = 1.0 / (tied_villains + 1)

        equity_sum += share
        equity_square_sum += share * share

    equity = equity_sum / simulations
    ci95_low, ci95_high = _equity_confidence_interval(
        equity,
        equity_square_sum,
        simulations,
    )

    return EquityResult(
        hero_cards=(card_code(hero[0]), card_code(hero[1])),
        board_cards=tuple(card_code(card) for card in board),
        opponents=opponents,
        simulations=simulations,
        wins=wins,
        ties=ties,
        losses=losses,
        win_probability=wins / simulations,
        tie_probability=ties / simulations,
        loss_probability=losses / simulations,
        equity=equity,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        seed=actual_seed,
    )


def _prepare_range_sampler(
    opponent_range: HandRange,
    known_cards: frozenset[Card],
):
    def sample(rng: random.Random) -> HoleCards:
        return opponent_range.sample(rng, tuple(known_cards))

    return sample


def _equity_confidence_interval(
    mean: float,
    square_sum: float,
    sample_count: int,
) -> tuple[float, float]:
    if sample_count < 2:
        return mean, mean
    centered_sum = max(0.0, square_sum - sample_count * mean * mean)
    sample_variance = centered_sum / (sample_count - 1)
    standard_error = math.sqrt(sample_variance / sample_count)
    margin = _Z_95 * standard_error
    return max(0.0, mean - margin), min(1.0, mean + margin)


# A descriptive alias for callers that prefer an estimator-oriented name.
estimate_equity = monte_carlo_equity


__all__ = ["EquityResult", "estimate_equity", "monte_carlo_equity"]
