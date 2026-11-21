"""Opponent hole-card ranges and observable-action Bayesian updates.

Ranges contain probabilities over *possible* two-card holdings.  They never
contain a distinguished "actual opponent hand"; callers condition them only
on cards that are public to the hero and on observed actions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping
from itertools import combinations
from types import MappingProxyType
from typing import Any, TypeAlias

from pokerkit import Card

from .cards import CardLike, CardsLike, STANDARD_DECK, parse_cards

HoleCards: TypeAlias = tuple[Card, Card]
ComboLike: TypeAlias = str | Iterable[CardLike]
LikelihoodFunction: TypeAlias = Callable[[HoleCards], float]
LikelihoodSpec: TypeAlias = float | LikelihoodFunction | Mapping[Any, float]

_CARD_POSITION = {card: index for index, card in enumerate(STANDARD_DECK)}
_ALL_COMBOS: tuple[HoleCards, ...] = tuple(combinations(STANDARD_DECK, 2))


def canonical_combo(cards: ComboLike) -> HoleCards:
    """Parse and return a deterministic two-card combination."""

    parsed = parse_cards(cards, expected_count=2, field_name="hole-card combination")
    first, second = sorted(parsed, key=_CARD_POSITION.__getitem__)
    return first, second


def _validate_weight(value: object, *, label: str, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    weight = float(value)
    lower_bound_ok = weight >= 0 if allow_zero else weight > 0
    if not math.isfinite(weight) or not lower_bound_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return weight


def _parse_dead_cards(dead_cards: CardsLike | None) -> frozenset[Card]:
    return frozenset(parse_cards(dead_cards, field_name="dead_cards"))


class HandRange:
    """Read-only probability distribution over two-card holdings."""

    @property
    def weights(self) -> Mapping[HoleCards, float]:
        """Normalized, unconditioned probabilities for every stored combo."""

        raise NotImplementedError

    def probabilities(self, dead_cards: CardsLike | None = ()) -> dict[HoleCards, float]:
        """Return probabilities conditioned on the supplied known cards."""

        dead = _parse_dead_cards(dead_cards)
        eligible = {
            combo: weight
            for combo, weight in self.weights.items()
            if combo[0] not in dead and combo[1] not in dead and weight > 0
        }
        total = math.fsum(eligible.values())
        if total <= 0:
            raise ValueError("the range has no positive-weight combo after removing dead cards")
        return {combo: weight / total for combo, weight in eligible.items()}

    def sample(
        self,
        rng: random.Random,
        dead_cards: CardsLike | None = (),
    ) -> HoleCards:
        """Sample a combo after removing holdings blocked by known cards."""

        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random")

        probabilities = self.probabilities(dead_cards)
        threshold = rng.random()
        cumulative = 0.0
        last_combo: HoleCards | None = None
        for combo, probability in probabilities.items():
            last_combo = combo
            cumulative += probability
            if threshold < cumulative:
                return combo

        # Floating-point summation can leave cumulative infinitesimally below
        # one.  The final eligible combo owns that remainder.
        assert last_combo is not None
        return last_combo

    def __len__(self) -> int:
        return len(self.weights)


class WeightedRange(HandRange):
    """A normalized range constructed from non-negative combo weights.

    Reversed duplicates such as ``("As", "Kd")`` and ``("Kd", "As")`` are
    combined before normalization.  Zero-weight entries are omitted.
    """

    def __init__(
        self,
        weights: Mapping[object, float] | Iterable[tuple[ComboLike, float]],
    ) -> None:
        items = weights.items() if isinstance(weights, Mapping) else weights
        combined: dict[HoleCards, float] = {}
        try:
            for raw_combo, raw_weight in items:
                combo = canonical_combo(raw_combo)  # type: ignore[arg-type]
                weight = _validate_weight(raw_weight, label=f"weight for {combo}")
                combined[combo] = combined.get(combo, 0.0) + weight
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise TypeError("weights must contain (two-card combo, numeric weight) pairs") from exc

        positive = {combo: weight for combo, weight in combined.items() if weight > 0}
        total = math.fsum(positive.values())
        if total <= 0:
            raise ValueError("a range must contain at least one positive-weight combo")

        ordered = sorted(
            positive.items(),
            key=lambda item: (_CARD_POSITION[item[0][0]], _CARD_POSITION[item[0][1]]),
        )
        normalized = {combo: weight / total for combo, weight in ordered}
        self._weights: Mapping[HoleCards, float] = MappingProxyType(normalized)

    @property
    def weights(self) -> Mapping[HoleCards, float]:
        return self._weights


class UniformRange(WeightedRange):
    """The uniform prior over all 1,326 standard two-card combinations."""

    def __init__(self) -> None:
        super().__init__((combo, 1.0) for combo in _ALL_COMBOS)


class OpponentProfile:
    """Observable-action likelihood model used for Bayesian range updates.

    Each action maps to one of:

    * a constant likelihood,
    * a callable ``combo -> likelihood``, or
    * a mapping of combos to likelihoods (unlisted combos receive zero).

    Likelihoods represent ``P(action | combo, public information)`` and must be
    between zero and one.  A callable may close over public context such as
    position or bet size; it must not inspect the simulator's hidden state.
    """

    def __init__(self, name: str, action_likelihoods: Mapping[str, LikelihoodSpec]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("profile name must be a non-empty string")
        if not isinstance(action_likelihoods, Mapping) or not action_likelihoods:
            raise ValueError("action_likelihoods must be a non-empty mapping")

        parsed: dict[str, LikelihoodSpec] = {}
        for raw_action, raw_spec in action_likelihoods.items():
            if not isinstance(raw_action, str) or not raw_action.strip():
                raise ValueError("action names must be non-empty strings")
            action = raw_action.strip().lower()
            if action in parsed:
                raise ValueError(f"duplicate action after normalization: {action!r}")

            if callable(raw_spec):
                spec: LikelihoodSpec = raw_spec
            elif isinstance(raw_spec, Mapping):
                combo_values: dict[HoleCards, float] = {}
                for raw_combo, raw_value in raw_spec.items():
                    combo = canonical_combo(raw_combo)  # type: ignore[arg-type]
                    combo_values[combo] = _validate_likelihood(
                        raw_value,
                        label=f"{action} likelihood for {combo}",
                    )
                spec = MappingProxyType(combo_values)
            else:
                spec = _validate_likelihood(raw_spec, label=f"{action} likelihood")
            parsed[action] = spec

        self.name = name.strip()
        self._action_likelihoods: Mapping[str, LikelihoodSpec] = MappingProxyType(parsed)

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(self._action_likelihoods)

    def likelihood(self, action: str, combo: ComboLike) -> float:
        """Return ``P(action | combo)`` for one possible holding."""

        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        normalized_action = action.strip().lower()
        try:
            spec = self._action_likelihoods[normalized_action]
        except KeyError as exc:
            raise KeyError(
                f"profile {self.name!r} has no likelihood for action {normalized_action!r}",
            ) from exc

        parsed_combo = canonical_combo(combo)
        if callable(spec):
            raw_value = spec(parsed_combo)
        elif isinstance(spec, Mapping):
            raw_value = spec.get(parsed_combo, 0.0)
        else:
            raw_value = spec
        return _validate_likelihood(
            raw_value,
            label=f"{normalized_action} likelihood for {parsed_combo}",
        )


def _validate_likelihood(value: object, *, label: str) -> float:
    likelihood = _validate_weight(value, label=label)
    if likelihood > 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return likelihood


def bayes_update(
    prior: HandRange | Mapping[object, float],
    action: str,
    profile: OpponentProfile,
    *,
    dead_cards: CardsLike | None = (),
) -> WeightedRange:
    """Condition a hand range on one observed opponent action.

    The returned weights are proportional to
    ``P(action | combo) * P(combo)`` after removing public dead cards.
    """

    if isinstance(prior, Mapping):
        prior = WeightedRange(prior)
    if not isinstance(prior, HandRange):
        raise TypeError("prior must be a HandRange or combo-to-weight mapping")
    if not isinstance(profile, OpponentProfile):
        raise TypeError("profile must be an OpponentProfile")

    prior_probabilities = prior.probabilities(dead_cards)
    posterior = {
        combo: probability * profile.likelihood(action, combo)
        for combo, probability in prior_probabilities.items()
    }
    if math.fsum(posterior.values()) <= 0:
        raise ValueError(
            f"action {action!r} has zero total likelihood under profile {profile.name!r}",
        )
    return WeightedRange(posterior)


__all__ = [
    "ComboLike",
    "HandRange",
    "HoleCards",
    "LikelihoodFunction",
    "OpponentProfile",
    "UniformRange",
    "WeightedRange",
    "bayes_update",
    "canonical_combo",
]
