"""PokerKit adapter for one reproducible heads-up no-limit Hold'em hand.

PokerKit owns the rules, betting state, and settlement.  This module keeps
that omniscient engine state behind an actor-specific :class:`Observation` so
policies never receive the opponent's cards or the remaining deck.

Public monetary values are big blinds.  PokerKit receives integer chip units
at a resolution of one hundred units per big blind; this avoids floating-point
chips while retaining centi-big-blind raise sizes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP
import math
import random
from typing import TypeAlias

from pokerkit import (
    Automation,
    Card,
    HandHistory,
    Mode,
    NoLimitTexasHoldem,
    State,
)

from .actions import CandidateAction, LegalActions
from .cards import CardLike, STANDARD_DECK, card_code, parse_cards
from .observation import Observation


UNITS_PER_BB = 100
SMALL_BLIND_UNITS = 50
BIG_BLIND_UNITS = 100
STARTING_STACK_UNITS = 100 * UNITS_PER_BB
PLAYER_COUNT = 2

_STREET_NAMES = ("preflop", "flop", "turn", "river")
_NON_CARD_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)

Policy: TypeAlias = Callable[[Observation], CandidateAction]
PolicyCollection: TypeAlias = Policy | Sequence[Policy] | Mapping[int, Policy]


def _validate_player_index(player_index: int, *, name: str = "player_index") -> int:
    if isinstance(player_index, bool) or not isinstance(player_index, int):
        raise TypeError(f"{name} must be an integer")
    if player_index not in range(PLAYER_COUNT):
        raise ValueError(f"{name} must be 0 or 1")
    return player_index


def _to_bb(amount: int) -> float:
    return amount / UNITS_PER_BB


def _to_units(amount: float, *, name: str) -> int:
    """Convert a finite BB amount to the nearest centi-BB chip unit."""

    if isinstance(amount, bool):
        raise TypeError(f"{name} must be a number, not bool")
    try:
        numeric = float(amount)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0:
        raise ValueError(f"{name} cannot be negative")

    # Decimal(str(...)) makes the user-facing decimal amount authoritative and
    # avoids surprises such as int(2.3 * 100) becoming 229.
    scaled = Decimal(str(numeric)) * UNITS_PER_BB
    return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


class PokerKitHeadsUpHand:
    """Play one seeded 100 BB heads-up no-limit Hold'em hand.

    Seat ``0`` is PokerKit's heads-up big blind and seat ``1`` is the
    small-blind/button.  A fresh instance always starts both seats at 100 BB.

    Args:
        seed: Seed for the local RNG used to build the explicit deck order.
        hero_index: Seat whose payoff is normally being evaluated.
        deck_order: Optional complete 52-card order, with the first item dealt
            first.  It is mainly useful for exact scenario tests and overrides
            the seeded shuffle.
    """

    units_per_bb = UNITS_PER_BB
    small_blind = SMALL_BLIND_UNITS / UNITS_PER_BB
    big_blind = BIG_BLIND_UNITS / UNITS_PER_BB
    starting_stack = STARTING_STACK_UNITS / UNITS_PER_BB
    button_index = 1

    def __init__(
        self,
        seed: int = 0,
        *,
        hero_index: int = 0,
        deck_order: Iterable[CardLike] | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        self.seed = seed
        self.hero_index = _validate_player_index(hero_index, name="hero_index")
        self.game = NoLimitTexasHoldem(
            _NON_CARD_AUTOMATIONS,
            True,  # Uniform-ante trimming; there is no ante in the MVP.
            0,
            (SMALL_BLIND_UNITS, BIG_BLIND_UNITS),
            BIG_BLIND_UNITS,
            mode=Mode.TOURNAMENT,
        )
        self._state = self.game(
            (STARTING_STACK_UNITS, STARTING_STACK_UNITS),
            PLAYER_COUNT,
        )
        self._history: list[dict[str, object]] = []

        if deck_order is None:
            ordered_deck = list(STANDARD_DECK)
            random.Random(seed).shuffle(ordered_deck)
        else:
            ordered_deck = list(
                parse_cards(
                    deck_order,
                    expected_count=len(STANDARD_DECK),
                    field_name="deck_order",
                ),
            )

        # State construction shuffles its own deck.  Card automation is
        # intentionally disabled, so replacing it here occurs before any card
        # is dealt and makes the entire hand a pure function of this order and
        # the chosen actions.
        self._initial_deck_order = tuple(ordered_deck)
        self._state.deck_cards.clear()
        self._state.deck_cards.extend(ordered_deck)
        self._advance_cards_until_decision_or_completion()
        self._assert_chip_conservation()

    @property
    def state(self) -> State:
        """Return the omniscient engine state for orchestration and tests.

        Policy callbacks are deliberately never passed this object; they only
        receive :meth:`observation_for` results.
        """

        return self._state

    @property
    def history(self) -> tuple[dict[str, object], ...]:
        """Return defensive copies of the public player-action history."""

        return tuple(dict(event) for event in self._history)

    @property
    def is_complete(self) -> bool:
        return not self._state.status

    @property
    def stacks(self) -> tuple[float, float]:
        return tuple(_to_bb(amount) for amount in self._state.stacks)  # type: ignore[return-value]

    @property
    def payoffs(self) -> tuple[float, float]:
        """Net chip results relative to the two 100 BB starting stacks."""

        return tuple(_to_bb(amount) for amount in self._state.payoffs)  # type: ignore[return-value]

    def _street_name(self) -> str:
        street_index = self._state.street_index
        if street_index is None or street_index not in range(len(_STREET_NAMES)):
            raise RuntimeError("the hand is not at a policy decision street")
        return _STREET_NAMES[street_index]

    def _advance_cards_until_decision_or_completion(self) -> None:
        """Deal from the explicit deck until a player must act or hand ends."""

        transition_count = 0
        while self._state.status and self._state.actor_index is None:
            transition_count += 1
            if transition_count > 64:  # Defensive guard against API drift.
                raise RuntimeError("PokerKit did not reach a decision or settlement")

            if self._state.can_burn_card():
                self._state.burn_card()
            elif self._state.can_deal_hole():
                self._state.deal_hole()
            elif self._state.can_deal_board():
                self._state.deal_board()
            else:
                raise RuntimeError(
                    "PokerKit stopped in an unexpected non-player phase",
                )

    def _assert_chip_conservation(self) -> None:
        chips_in_play = sum(self._state.stacks) + self._state.total_pot_amount
        expected = PLAYER_COUNT * STARTING_STACK_UNITS
        if chips_in_play != expected:
            raise RuntimeError(
                f"chip conservation failed: expected {expected}, got {chips_in_play}",
            )

    def _legal_actions_for(self, player_index: int) -> LegalActions:
        if player_index != self._state.actor_index:
            return LegalActions(check=False, call=False, fold=False)

        can_check_or_call = self._state.can_check_or_call()
        call_units = self._state.checking_or_calling_amount
        check = can_check_or_call and call_units == 0
        call = can_check_or_call and call_units is not None and call_units > 0

        if self._state.can_complete_bet_or_raise_to():
            minimum = self._state.min_completion_betting_or_raising_to_amount
            maximum = self._state.max_completion_betting_or_raising_to_amount
            if minimum is None or maximum is None:  # pragma: no cover - API invariant
                raise RuntimeError("PokerKit reported a raise without raise bounds")
            min_raise_to = _to_bb(minimum)
            max_raise_to = _to_bb(maximum)
        else:
            min_raise_to = None
            max_raise_to = None

        return LegalActions(
            check=check,
            call=call,
            fold=self._state.can_fold(),
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
        )

    def observation_for(self, player_index: int) -> Observation:
        """Return a copied, hidden-information-safe view for one seat.

        Legal actions are populated only when *player_index* is the current
        actor.  A non-acting seat may still obtain a public snapshot, but it
        cannot accidentally feed that snapshot to the action generator as if
        it were its turn.
        """

        player_index = _validate_player_index(player_index)
        if self.is_complete:
            raise RuntimeError("cannot observe a completed hand")
        actor_index = self._state.actor_index
        if actor_index is None:  # Card draining should make this unreachable.
            raise RuntimeError("there is currently no player action pending")

        board = tuple(card_code(card) for card in self._state.get_board_cards(0))
        hero_cards = tuple(card_code(card) for card in self._state.hole_cards[player_index])
        if len(hero_cards) != 2:  # pragma: no cover - construction invariant
            raise RuntimeError("the player's two hole cards have not been dealt")

        if self._state.statuses[player_index] and self._state.stacks[player_index]:
            to_call_units = min(
                self._state.stacks[player_index],
                max(self._state.bets) - self._state.bets[player_index],
            )
        else:
            to_call_units = 0

        player_statuses = []
        for index in range(PLAYER_COUNT):
            if not self._state.statuses[index]:
                player_statuses.append("folded")
            elif not self._state.stacks[index]:
                player_statuses.append("all_in")
            else:
                player_statuses.append("active")

        return Observation(
            hero_cards=hero_cards,  # type: ignore[arg-type]
            board=board,
            street=self._street_name(),
            hero_index=player_index,
            button_index=self.button_index,
            actor_index=actor_index,
            pot=_to_bb(self._state.total_pot_amount),
            to_call=_to_bb(to_call_units),
            stacks=tuple(_to_bb(amount) for amount in self._state.stacks),
            street_contributions=tuple(_to_bb(amount) for amount in self._state.bets),
            hand_contributions=tuple(
                _to_bb(max(0, -payoff)) for payoff in self._state.payoffs
            ),
            player_statuses=tuple(player_statuses),
            action_history=self.history,
            legal_actions=self._legal_actions_for(player_index),
        )

    def apply_action(self, action: CandidateAction) -> Observation | None:
        """Validate and apply one current-actor action.

        Raise amounts arrive in BB ``raise_to`` units and are converted to the
        nearest centi-BB engine unit.  Values outside PokerKit's inclusive
        range are rejected before any state mutation.  The next actor's
        observation is returned, or ``None`` after exact engine settlement.
        """

        if not isinstance(action, CandidateAction):
            raise TypeError("action must be a CandidateAction")
        if self.is_complete:
            raise RuntimeError("cannot act in a completed hand")

        actor_index = self._state.actor_index
        if actor_index is None:
            raise RuntimeError("there is currently no player action pending")
        street = self._street_name()

        if action.kind == "fold":
            if not self._state.can_fold():
                raise ValueError("fold is not legal in the current state")
            operation = self._state.fold()
            record: dict[str, object] = {
                "sequence": len(self._history),
                "street": street,
                "player_index": actor_index,
                "kind": "fold",
            }
        elif action.kind in ("check", "call"):
            if not self._state.can_check_or_call():
                raise ValueError(f"{action.kind} is not legal in the current state")
            call_units = self._state.checking_or_calling_amount
            if call_units is None:  # pragma: no cover - API invariant
                raise RuntimeError("PokerKit omitted the check/call amount")
            if action.kind == "check" and call_units != 0:
                raise ValueError("check is not legal while facing a bet")
            if action.kind == "call" and call_units == 0:
                raise ValueError("call is not legal when checking is available")
            operation = self._state.check_or_call()
            record = {
                "sequence": len(self._history),
                "street": street,
                "player_index": actor_index,
                "kind": action.kind,
                "amount": _to_bb(operation.amount),
            }
        elif action.kind == "raise":
            if action.raise_to is None:  # CandidateAction enforces this too.
                raise ValueError("raise actions require raise_to")
            minimum = self._state.min_completion_betting_or_raising_to_amount
            maximum = self._state.max_completion_betting_or_raising_to_amount
            if (
                not self._state.can_complete_bet_or_raise_to()
                or minimum is None
                or maximum is None
            ):
                raise ValueError("raise is not legal in the current state")

            requested = float(action.raise_to)
            minimum_bb = _to_bb(minimum)
            maximum_bb = _to_bb(maximum)
            if requested < minimum_bb or requested > maximum_bb:
                raise ValueError(
                    "raise_to must be within the inclusive legal range "
                    f"[{minimum_bb}, {maximum_bb}] BB",
                )

            raise_to_units = _to_units(requested, name="raise_to")
            # Rounding an in-range decimal at the edge must stay on the engine
            # range.  This is not used to rescue an out-of-range request.
            raise_to_units = max(minimum, min(maximum, raise_to_units))
            if not self._state.can_complete_bet_or_raise_to(raise_to_units):
                raise ValueError("raise_to is not legal in the current state")
            operation = self._state.complete_bet_or_raise_to(raise_to_units)
            record = {
                "sequence": len(self._history),
                "street": street,
                "player_index": actor_index,
                "kind": "raise",
                "raise_to": _to_bb(operation.amount),
            }
        else:  # pragma: no cover - CandidateAction rejects other kinds.
            raise ValueError(f"unsupported action kind: {action.kind!r}")

        self._history.append(record)
        self._advance_cards_until_decision_or_completion()
        self._assert_chip_conservation()

        if self.is_complete:
            if sum(self._state.stacks) != PLAYER_COUNT * STARTING_STACK_UNITS:
                raise RuntimeError("settlement did not return every chip to a stack")
            if sum(self._state.payoffs) != 0:
                raise RuntimeError("zero-rake heads-up payoffs are not zero-sum")
            return None

        next_actor = self._state.actor_index
        if next_actor is None:  # pragma: no cover - drain invariant
            raise RuntimeError("PokerKit did not expose the next actor")
        return self.observation_for(next_actor)

    def _policy_for(self, policies: PolicyCollection, player_index: int) -> Policy:
        if isinstance(policies, Mapping):
            try:
                policy = policies[player_index]
            except KeyError as exc:
                raise ValueError(f"no policy supplied for player {player_index}") from exc
        elif callable(policies):
            policy = policies
        else:
            if len(policies) != PLAYER_COUNT:
                raise ValueError("policies must contain exactly two callbacks")
            policy = policies[player_index]
        if not callable(policy):
            raise TypeError(f"policy for player {player_index} is not callable")
        return policy

    def run(
        self,
        policies: PolicyCollection,
        *,
        max_actions: int = 256,
    ) -> tuple[float, float]:
        """Run policy callbacks to settlement and return net payoffs in BB.

        A single callable may control both seats; alternatively pass a two-item
        sequence or a mapping keyed by seat index.  Every callback receives
        only its actor-specific :class:`Observation`.
        """

        if isinstance(max_actions, bool) or not isinstance(max_actions, int):
            raise TypeError("max_actions must be an integer")
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")

        action_count = 0
        while not self.is_complete:
            if action_count >= max_actions:
                raise RuntimeError("policy run exceeded max_actions")
            actor_index = self._state.actor_index
            if actor_index is None:  # pragma: no cover - drain invariant
                raise RuntimeError("PokerKit has no actor before settlement")
            observation = self.observation_for(actor_index)
            policy = self._policy_for(policies, actor_index)
            action = policy(observation)
            self.apply_action(action)
            action_count += 1
        return self.payoffs

    def to_hand_history(self) -> HandHistory:
        """Create PokerKit's PHH representation for diagnostics/replay."""

        kwargs: dict[str, object] = {
            "players": ["player_0", "player_1"],
        }
        if self.is_complete:
            kwargs["finishing_stacks"] = list(self._state.stacks)
        return HandHistory.from_game_state(self.game, self._state, **kwargs)


# A concise alternative name for callers that treat this as their only game
# environment rather than specifically as a PokerKit adapter.
HeadsUpEnvironment = PokerKitHeadsUpHand


__all__ = [
    "BIG_BLIND_UNITS",
    "HeadsUpEnvironment",
    "PLAYER_COUNT",
    "PokerKitHeadsUpHand",
    "SMALL_BLIND_UNITS",
    "STARTING_STACK_UNITS",
    "UNITS_PER_BB",
]
