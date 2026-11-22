"""Legal-action descriptions and deterministic action abstraction.

The policy works with a deliberately small no-limit action set.  Raise sizes are
expressed as ``raise_to`` (the player's total contribution on the current
street), never as the additional number of chips to put in.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

if TYPE_CHECKING:  # Avoid a runtime cycle: observation imports LegalActions.
    from .observation import Observation


_ACTION_KINDS = frozenset(("fold", "check", "call", "raise"))


def _chip_amount(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer chip amount".format(name))
    return value


def _finite_fraction(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError("{} must be a number, not bool".format(name))
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(fraction):
        raise ValueError("{} must be finite".format(name))
    return fraction


@dataclass(frozen=True)
class LegalActions:
    """Actions and raise bounds supplied by the game engine.

    ``min_raise_to`` and ``max_raise_to`` are both ``None`` when raising is not
    legal.  Normally they are inclusive bounds.  For an engine-approved short
    all-in, ``max_raise_to`` may be below the minimum full-raise amount and is
    then the only aggressive target.
    """

    check: bool
    call: bool
    fold: bool
    min_raise_to: Optional[int] = None
    max_raise_to: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("check", "call", "fold"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("{} must be bool".format(name))

        minimum = self.min_raise_to
        maximum = self.max_raise_to
        if (minimum is None) != (maximum is None):
            raise ValueError(
                "min_raise_to and max_raise_to must either both be set or both be None"
            )
        if minimum is None:
            return

        minimum_value = _chip_amount(minimum, "min_raise_to")
        maximum_value = _chip_amount(maximum, "max_raise_to")
        if minimum_value < 0 or maximum_value < 0:
            raise ValueError("raise bounds cannot be negative")
        # PokerKit may expose a short all-in whose maximum is below the
        # minimum amount that would constitute a full raise.  In that case the
        # maximum itself is the only aggressive target.
        object.__setattr__(self, "min_raise_to", minimum_value)
        object.__setattr__(self, "max_raise_to", maximum_value)

    @property
    def can_raise(self) -> bool:
        return self.min_raise_to is not None


@dataclass(frozen=True)
class CandidateAction:
    """A canonical action considered by the policy."""

    kind: str
    raise_to: Optional[int] = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in _ACTION_KINDS:
            raise ValueError("unsupported action kind: {!r}".format(self.kind))
        object.__setattr__(self, "kind", kind)

        if kind == "raise":
            if self.raise_to is None:
                raise ValueError("raise actions require raise_to")
            amount = _chip_amount(self.raise_to, "raise_to")
            if amount < 0:
                raise ValueError("raise_to cannot be negative")
            object.__setattr__(self, "raise_to", amount)
        elif self.raise_to is not None:
            raise ValueError("only raise actions may specify raise_to")

    @property
    def is_aggressive(self) -> bool:
        return self.kind == "raise"


def _deduplicate_amounts(amounts: Sequence[int]) -> Tuple[int, ...]:
    unique: list[int] = []
    for amount in amounts:
        if amount not in unique:
            unique.append(amount)
    return tuple(unique)


def generate_candidate_actions(
    observation: "Observation",
    pot_fractions: Sequence[float] = (0.5, 1.0),
) -> Tuple[CandidateAction, ...]:
    """Return legal passive actions plus a compact set of raise sizes.

    Fractional sizes use the pot *after calling*::

        raise_to = hero_street_contribution + call_cost
                   + floor(fraction * (pot + call_cost))

    When facing a bet, the minimum full raise is also considered.  With no call
    cost it is omitted.  Out-of-range fraction sizes are removed rather than
    clamped.  The all-in target is ``B + S`` and duplicates are removed.
    """

    legal = observation.legal_actions
    call_cost = observation.call_cost
    actions = []
    if call_cost == 0:
        if legal.check:
            actions.append(CandidateAction("check"))
    else:
        if legal.fold:
            actions.append(CandidateAction("fold"))
        if legal.call:
            actions.append(CandidateAction("call"))

    if not legal.can_raise:
        if not actions:
            raise ValueError("observation has no legal candidate action")
        return tuple(actions)

    minimum = legal.min_raise_to
    maximum = legal.max_raise_to
    assert minimum is not None and maximum is not None
    hero_contribution = observation.street_contributions[observation.hero_index]
    pot = observation.pot
    base_after_call = hero_contribution + call_cost
    pot_after_call = pot + call_cost

    hero_stack = observation.stacks[observation.hero_index]
    if call_cost >= hero_stack:
        if not actions:
            raise ValueError("observation has no legal candidate action")
        return tuple(actions)

    raw_sizes = [minimum] if call_cost > 0 else []
    for fraction in pot_fractions:
        fraction_value = _finite_fraction(fraction, "pot fraction")
        if fraction_value < 0.0:
            raise ValueError("pot fractions cannot be negative")
        raw_sizes.append(base_after_call + math.floor(fraction_value * pot_after_call))
    general_sizes = [size for size in raw_sizes if minimum <= size <= maximum]

    all_in_to = hero_contribution + hero_stack
    full_raise_all_in = minimum <= all_in_to <= maximum
    short_all_in = maximum < minimum and all_in_to == maximum
    if full_raise_all_in or short_all_in:
        general_sizes.append(all_in_to)

    for amount in sorted(_deduplicate_amounts(general_sizes)):
        actions.append(CandidateAction("raise", amount))

    if not actions:
        raise ValueError("observation has no legal candidate action")
    return tuple(actions)


# A short alias is convenient at integration boundaries while keeping the
# descriptive public name above as the canonical API.
candidate_actions = generate_candidate_actions


__all__ = [
    "CandidateAction",
    "LegalActions",
    "candidate_actions",
    "generate_candidate_actions",
]
