"""Quantized opponent ranges and observable-action Bayesian updates."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal, ROUND_HALF_EVEN
from itertools import combinations
from types import MappingProxyType
from typing import Any, TypeAlias

from pokerkit import Card

from .cards import CardLike, CardsLike, STANDARD_DECK, parse_cards


QUANTIZATION_SCALE = 10**15

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


def combo_sort_key(combo: HoleCards) -> tuple[int, int]:
    return _CARD_POSITION[combo[0]], _CARD_POSITION[combo[1]]


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


def _quantized_masses(weights: Mapping[HoleCards, float]) -> dict[HoleCards, int]:
    total = math.fsum(weights.values())
    if total <= 0:
        raise ValueError("a range must contain at least one positive-weight combo")
    scale = Decimal(QUANTIZATION_SCALE)
    masses: dict[HoleCards, int] = {}
    for combo in sorted(weights, key=combo_sort_key):
        probability = weights[combo] / total
        mass = int(
            (Decimal(str(probability)) * scale).quantize(
                Decimal("1"), rounding=ROUND_HALF_EVEN
            )
        )
        if mass > 0:
            masses[combo] = mass
    if not masses:
        raise ValueError("range quantization removed every combo")
    return masses


class HandRange:
    """Read-only integer-mass distribution over two-card holdings."""

    @property
    def masses(self) -> Mapping[HoleCards, int]:
        raise NotImplementedError

    @property
    def total_mass(self) -> int:
        raise NotImplementedError

    @property
    def weights(self) -> Mapping[HoleCards, float]:
        """Logical probabilities derived from, but not stored over, masses."""

        total = self.total_mass
        return MappingProxyType({combo: mass / total for combo, mass in self.masses.items()})

    def probabilities(self, dead_cards: CardsLike | None = ()) -> dict[HoleCards, float]:
        """Return logical probabilities after removing supplied blockers."""

        dead = _parse_dead_cards(dead_cards)
        eligible = {
            combo: mass
            for combo, mass in self.masses.items()
            if combo[0] not in dead and combo[1] not in dead and mass > 0
        }
        total = sum(eligible.values())
        if total <= 0:
            raise ValueError("the range has no positive-mass combo after removing dead cards")
        return {combo: mass / total for combo, mass in eligible.items()}

    def conditioned(self, dead_cards: CardsLike | None = ()) -> "WeightedRange":
        """Remove blockers, normalize, and quantize the resulting distribution."""

        return WeightedRange(self.probabilities(dead_cards))

    def sample(
        self,
        rng: random.Random,
        dead_cards: CardsLike | None = (),
    ) -> HoleCards:
        """Sample using an integer draw in ``[0, eligible_total_mass)``."""

        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random")
        dead = _parse_dead_cards(dead_cards)
        eligible = tuple(
            (combo, mass)
            for combo, mass in self.masses.items()
            if combo[0] not in dead and combo[1] not in dead and mass > 0
        )
        total = sum(mass for _, mass in eligible)
        if total <= 0:
            raise ValueError("the range has no positive-mass combo after removing dead cards")
        threshold = rng.randrange(total)
        cumulative = 0
        for combo, mass in eligible:
            cumulative += mass
            if threshold < cumulative:
                return combo
        raise RuntimeError("integer range sampling failed")  # pragma: no cover

    def __len__(self) -> int:
        return len(self.masses)


class WeightedRange(HandRange):
    """A canonical range quantized to fixed integer probability masses."""

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
        self._set_masses(_quantized_masses(positive))

    @classmethod
    def from_masses(
        cls,
        masses: Mapping[object, int] | Iterable[tuple[ComboLike, int]],
    ) -> "WeightedRange":
        """Restore already-quantized masses without changing their values."""

        items = masses.items() if isinstance(masses, Mapping) else masses
        parsed: dict[HoleCards, int] = {}
        for raw_combo, raw_mass in items:
            combo = canonical_combo(raw_combo)  # type: ignore[arg-type]
            if isinstance(raw_mass, bool) or not isinstance(raw_mass, int):
                raise TypeError(f"mass for {combo} must be an integer")
            if raw_mass < 0:
                raise ValueError(f"mass for {combo} cannot be negative")
            if raw_mass > 0:
                parsed[combo] = parsed.get(combo, 0) + raw_mass
        if not parsed:
            raise ValueError("a range must contain at least one positive mass")
        instance = cls.__new__(cls)
        instance._set_masses(parsed)
        return instance

    def _set_masses(self, masses: Mapping[HoleCards, int]) -> None:
        ordered = {combo: masses[combo] for combo in sorted(masses, key=combo_sort_key)}
        total = sum(ordered.values())
        if total <= 0:
            raise ValueError("total_mass must be positive")
        self._masses: Mapping[HoleCards, int] = MappingProxyType(ordered)
        self._total_mass = total

    @property
    def masses(self) -> Mapping[HoleCards, int]:
        return self._masses

    @property
    def total_mass(self) -> int:
        return self._total_mass


class UniformRange(WeightedRange):
    """The uniform prior over all 1,326 standard two-card combinations."""

    def __init__(self) -> None:
        super().__init__((combo, 1.0) for combo in _ALL_COMBOS)


class OpponentProfile:
    """Observable-action likelihoods used for Bayesian range updates."""

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
                        raw_value, label=f"{action} likelihood for {combo}"
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
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        normalized_action = action.strip().lower()
        try:
            spec = self._action_likelihoods[normalized_action]
        except KeyError as exc:
            raise KeyError(
                f"profile {self.name!r} has no likelihood for action {normalized_action!r}"
            ) from exc
        parsed_combo = canonical_combo(combo)
        if callable(spec):
            raw_value = spec(parsed_combo)
        elif isinstance(spec, Mapping):
            raw_value = spec.get(parsed_combo, 0.0)
        else:
            raw_value = spec
        return _validate_likelihood(
            raw_value, label=f"{normalized_action} likelihood for {parsed_combo}"
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
    """Condition a range on one observed public action and quantize it."""

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
            f"action {action!r} has zero total likelihood under profile {profile.name!r}"
        )
    return WeightedRange(posterior)


__all__ = [
    "ComboLike",
    "HandRange",
    "HoleCards",
    "LikelihoodFunction",
    "OpponentProfile",
    "QUANTIZATION_SCALE",
    "UniformRange",
    "WeightedRange",
    "bayes_update",
    "canonical_combo",
    "combo_sort_key",
]
