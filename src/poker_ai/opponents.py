"""Shared, explicit opponent action model for likelihoods and baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from pokerkit import Card

from .actions import CandidateAction
from .cards import CardsLike, parse_cards, validate_unique_cards
from .ranges import HandRange, HoleCards, OpponentProfile, canonical_combo
from .seeding import action_candidate_id


_RANK_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


@dataclass(frozen=True)
class StyleParameters:
    name: str
    f0: float
    c0: float
    r0: float
    b: float
    epsilon: float = 0.01

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


STYLES = {
    "balanced": StyleParameters("balanced", 1.0, 0.0, -1.0, 0.0),
    "tight": StyleParameters("tight", 2.0, -0.5, -2.0, -0.5),
    "calling": StyleParameters("calling", -1.0, 2.0, -2.0, -0.5),
    "aggressive": StyleParameters("aggressive", 0.0, 0.0, 1.0, 0.5),
    "random": StyleParameters("random", 0.0, 0.0, 0.0, 0.0),
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


def _best_five_value(cards: Sequence[Card]) -> tuple[int, tuple[int, ...]]:
    from itertools import combinations

    best: tuple[int, tuple[int, ...]] | None = None
    for five in combinations(cards, 5):
        ranks = sorted((_rank(card) for card in five), reverse=True)
        counts = {rank: ranks.count(rank) for rank in set(ranks)}
        groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
        flush = len({_suit(card) for card in five}) == 1
        unique = sorted(set(ranks), reverse=True)
        if unique == [14, 5, 4, 3, 2]:
            straight_high = 5
        elif len(unique) == 5 and unique[0] - unique[-1] == 4:
            straight_high = unique[0]
        else:
            straight_high = 0

        if flush and straight_high:
            value = (8, (straight_high,))
        elif groups[0][0] == 4:
            value = (7, (groups[0][1], groups[1][1]))
        elif groups[0][0] == 3 and groups[1][0] == 2:
            value = (6, (groups[0][1], groups[1][1]))
        elif flush:
            value = (5, tuple(ranks))
        elif straight_high:
            value = (4, (straight_high,))
        elif groups[0][0] == 3:
            kickers = sorted((rank for rank in ranks if rank != groups[0][1]), reverse=True)
            value = (3, (groups[0][1], *kickers))
        elif groups[0][0] == 2 and groups[1][0] == 2:
            high_pair = max(groups[0][1], groups[1][1])
            low_pair = min(groups[0][1], groups[1][1])
            kicker = next(rank for rank in ranks if rank not in (high_pair, low_pair))
            value = (2, (high_pair, low_pair, kicker))
        elif groups[0][0] == 2:
            pair = groups[0][1]
            kickers = sorted((rank for rank in ranks if rank != pair), reverse=True)
            value = (1, (pair, *kickers))
        else:
            value = (0, tuple(ranks))
        if best is None or value > best:
            best = value
    if best is None:
        raise ValueError("postflop strength requires at least five total cards")
    return best


def visible_hand_strength(combo: HoleCards, board_cards: Iterable[Card] = ()) -> float:
    """Return the exact PRD v1.3 heuristic feature ``h`` in ``[0, 1]``."""

    combo = canonical_combo(combo)
    board = tuple(board_cards)
    validate_unique_cards(combo, board, field_names=("combo", "board_cards"))
    high, low = sorted((_rank(combo[0]), _rank(combo[1])), reverse=True)
    if not board:
        pair = float(high == low)
        suited = float(_suit(combo[0]) == _suit(combo[1]))
        adjacent = float(high - low == 1)
        value = (
            (high + low - 4) / 24
            + 0.35 * pair
            + 0.08 * suited
            + 0.05 * adjacent
        ) / 1.35
        return max(0.0, min(1.0, value))

    category, vector = _best_five_value((*combo, *board))
    padded = (*vector[:5], *(0 for _ in range(5 - len(vector))))
    kicker_fraction = math.fsum(rank / (15**index) for index, rank in enumerate(padded, 1))
    return (category + kicker_fraction) / 9


def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    exponentials = {key: math.exp(value - maximum) for key, value in scores.items()}
    total = math.fsum(exponentials.values())
    return {key: value / total for key, value in exponentials.items()}


class OpponentActionModel:
    """One shared candidate distribution used by inference and actual play."""

    def __init__(self, style: str | StyleParameters = "balanced") -> None:
        self.parameters = get_style(style) if isinstance(style, str) else style
        if not isinstance(self.parameters, StyleParameters):
            raise TypeError("style must be a style name or StyleParameters")
        if not 0.0 <= self.parameters.epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")

    @property
    def name(self) -> str:
        return self.parameters.name

    def probabilities(
        self,
        combo: HoleCards,
        *,
        board_cards: CardsLike | None,
        pot: int,
        call_cost: int,
        own_street_contribution: int,
        candidate_actions: Sequence[CandidateAction],
    ) -> Mapping[CandidateAction, float]:
        candidates = tuple(candidate_actions)
        if not candidates:
            raise ValueError("candidate_actions cannot be empty")
        if not all(isinstance(action, CandidateAction) for action in candidates):
            raise TypeError("candidate_actions must contain CandidateAction values")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate_actions cannot contain duplicates")
        for name, amount in (
            ("pot", pot),
            ("call_cost", call_cost),
            ("own_street_contribution", own_street_contribution),
        ):
            if isinstance(amount, bool) or not isinstance(amount, int):
                raise TypeError(f"{name} must be an integer chip amount")
            if amount < 0:
                raise ValueError(f"{name} cannot be negative")

        if self.name == "random":
            probability = 1.0 / len(candidates)
            return MappingProxyType({action: probability for action in candidates})

        board = parse_cards(board_cards, field_name="board_cards")
        h = visible_hand_strength(canonical_combo(combo), board)
        q = call_cost / max(1, pot + call_cost)
        p = self.parameters
        family_scores_all = {
            "fold": p.f0 - 4 * h + 2 * q,
            "check": 0.0,
            "call": p.c0 + 2 * h - 2 * q,
            "raise": p.r0 + 4 * h - q,
        }
        families = tuple(dict.fromkeys(action.kind for action in candidates))
        family_probabilities = _softmax({kind: family_scores_all[kind] for kind in families})
        raises = tuple(action for action in candidates if action.kind == "raise")
        if raises:
            size_scores = {}
            for action in raises:
                assert action.raise_to is not None
                z = (
                    action.raise_to - own_street_contribution - call_cost
                ) / max(1, pot + call_cost)
                size_scores[action_candidate_id(action)] = (2 * h - 1 + p.b) * z
            size_probabilities = _softmax(size_scores)
        else:
            size_probabilities = {}

        base: dict[CandidateAction, float] = {}
        for action in candidates:
            family_probability = family_probabilities[action.kind]
            if action.kind == "raise":
                base[action] = family_probability * size_probabilities[action_candidate_id(action)]
            else:
                base[action] = family_probability
        epsilon_share = p.epsilon / len(candidates)
        mixed = {
            action: (1.0 - p.epsilon) * probability + epsilon_share
            for action, probability in base.items()
        }
        normalizer = math.fsum(mixed.values())
        return MappingProxyType({action: probability / normalizer for action, probability in mixed.items()})

    def probability(self, action: CandidateAction, combo: HoleCards, **context: object) -> float:
        probabilities = self.probabilities(combo, **context)  # type: ignore[arg-type]
        try:
            return probabilities[action]
        except KeyError as exc:
            raise ValueError("action is not in the supplied candidate set") from exc


def make_range_profile(
    style_name: str,
    *,
    board_cards: CardsLike | None,
    pot: int,
    call_cost: int,
    own_street_contribution: int,
    candidate_actions: Sequence[CandidateAction],
) -> OpponentProfile:
    """Adapt the shared candidate model to the Bayesian range API."""

    model = OpponentActionModel(style_name)
    candidates = tuple(candidate_actions)
    likelihoods = {}
    for candidate in candidates:
        key = action_candidate_id(candidate)
        likelihoods[key] = lambda combo, candidate=candidate: model.probability(
            candidate,
            combo,
            board_cards=board_cards,
            pot=pot,
            call_cost=call_cost,
            own_street_contribution=own_street_contribution,
            candidate_actions=candidates,
        )
    return OpponentProfile(f"{model.name}:candidate-model-v1", likelihoods)


def estimate_fold_probability(style_name: str, *, raise_cost: float, pot: float) -> float:
    """Compatibility aggregate used by the current rollout until stage 7."""

    if raise_cost < 0 or pot < 0:
        raise ValueError("raise_cost and pot must be non-negative")
    if style_name == "random":
        return 0.5
    p = get_style(style_name)
    q = float(raise_cost) / max(1.0, float(pot) + float(raise_cost))
    probabilities = _softmax({"fold": p.f0 - 2.0 + 2 * q, "call": p.c0 + 1.0 - 2 * q})
    return (1.0 - p.epsilon) * probabilities["fold"] + p.epsilon / 2


def range_fold_probability(
    opponent_range: HandRange,
    style_name: str,
    *,
    hero_cards: CardsLike,
    board_cards: CardsLike | None,
    raise_cost: float,
    pot: float,
) -> float:
    """Aggregate fold probability with the shared model over a range."""

    if not isinstance(opponent_range, HandRange):
        raise TypeError("opponent_range must be a HandRange")
    call_cost = int(raise_cost)
    pot_chips = int(pot)
    candidates = (CandidateAction("fold"), CandidateAction("call"))
    model = OpponentActionModel(style_name)
    dead_cards = (*parse_cards(hero_cards), *parse_cards(board_cards))
    probabilities = opponent_range.probabilities(dead_cards)
    return math.fsum(
        range_probability
        * model.probability(
            CandidateAction("fold"),
            combo,
            board_cards=board_cards,
            pot=pot_chips,
            call_cost=call_cost,
            own_street_contribution=0,
            candidate_actions=candidates,
        )
        for combo, range_probability in probabilities.items()
    )


__all__ = [
    "OpponentActionModel",
    "STYLES",
    "StyleParameters",
    "estimate_fold_probability",
    "get_style",
    "make_range_profile",
    "range_fold_probability",
    "visible_hand_strength",
]
