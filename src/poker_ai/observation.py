"""Public-information-only input passed to a poker policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any, Iterator, Tuple

from .actions import LegalActions


_CARD_RE = re.compile(r"^([2-9TJQKA])([cdhs])$")
_BOARD_SIZE_BY_STREET = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}


class _FrozenMapping(Mapping):
    """Tiny immutable mapping used to freeze structured action-history data."""

    def __init__(self, items: Tuple[Tuple[str, Any], ...]) -> None:
        self._items = items

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        body = ", ".join("{!r}: {!r}".format(key, value) for key, value in self._items)
        return "FrozenMapping({{{}}})".format(body)

    def __hash__(self) -> int:
        return hash(self._items)


def _freeze(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("action-history numbers must be finite")
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item_value in value.items():
            if type(key) is not str:
                raise TypeError("action-history mapping keys must be strings")
            items.append((key, _freeze(item_value)))
        return _FrozenMapping(tuple(items))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    raise TypeError(
        "action_history may contain only strings, numbers, booleans, null, "
        "mappings, and sequences"
    )


def _normalise_card(card: object) -> str:
    # PokerKit Card uses its repr as the canonical two-character code.  Public
    # strings stay intentionally strict so malformed/OCR input is not repaired
    # silently at the policy boundary.
    text = card if isinstance(card, str) else repr(card)
    match = _CARD_RE.fullmatch(text)
    if match is None:
        raise ValueError("invalid card {!r}; use rank+suit notation such as 'As'".format(card))
    rank, suit = match.groups()
    return rank + suit


def _normalise_label(value: object, name: str) -> str:
    if hasattr(value, "value"):
        value = getattr(value, "value")
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        raise ValueError("{} cannot be empty".format(name))
    return text


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError("{} must be a number, not bool".format(name))
    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(amount):
        raise ValueError("{} must be finite".format(name))
    if amount < 0.0:
        raise ValueError("{} cannot be negative".format(name))
    return amount


def _player_index(value: object, name: str, player_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer".format(name))
    if not 0 <= value < player_count:
        raise ValueError("{} is outside the player array".format(name))
    return value


@dataclass(frozen=True)
class Observation:
    """Everything a policy may observe at one decision point.

    Deliberately absent are opponents' hole cards, deck order, and any other
    simulator-private state.  Sequence inputs are copied into immutable tuples
    so later engine mutation cannot change a recorded decision.
    """

    hero_cards: Tuple[str, str]
    board: Tuple[str, ...]
    street: str
    hero_index: int
    button_index: int
    actor_index: int
    pot: float
    to_call: float
    stacks: Tuple[float, ...]
    street_contributions: Tuple[float, ...]
    hand_contributions: Tuple[float, ...]
    player_statuses: Tuple[str, ...]
    action_history: Tuple[Any, ...]
    legal_actions: LegalActions

    def __post_init__(self) -> None:
        hero_cards = tuple(_normalise_card(card) for card in self.hero_cards)
        board = tuple(_normalise_card(card) for card in self.board)
        if len(hero_cards) != 2:
            raise ValueError("hero_cards must contain exactly two cards")

        street = _normalise_label(self.street, "street")
        if street not in _BOARD_SIZE_BY_STREET:
            raise ValueError("unsupported street: {!r}".format(self.street))
        required_board_size = _BOARD_SIZE_BY_STREET[street]
        if len(board) != required_board_size:
            raise ValueError(
                "{} observations require {} board cards".format(
                    street, required_board_size
                )
            )

        visible_cards = hero_cards + board
        if len(set(visible_cards)) != len(visible_cards):
            raise ValueError("hero_cards and board contain a duplicate card")

        stacks = tuple(
            _finite_nonnegative(value, "stacks[{}]".format(index))
            for index, value in enumerate(self.stacks)
        )
        player_count = len(stacks)
        if not 2 <= player_count <= 10:
            raise ValueError("observations require between 2 and 10 players")

        street_contributions = tuple(
            _finite_nonnegative(value, "street_contributions[{}]".format(index))
            for index, value in enumerate(self.street_contributions)
        )
        hand_contributions = tuple(
            _finite_nonnegative(value, "hand_contributions[{}]".format(index))
            for index, value in enumerate(self.hand_contributions)
        )
        player_statuses = tuple(
            _normalise_label(value, "player_statuses[{}]".format(index))
            for index, value in enumerate(self.player_statuses)
        )
        for name, values in (
            ("street_contributions", street_contributions),
            ("hand_contributions", hand_contributions),
            ("player_statuses", player_statuses),
        ):
            if len(values) != player_count:
                raise ValueError("{} must have one item per player".format(name))

        hero_index = _player_index(self.hero_index, "hero_index", player_count)
        button_index = _player_index(self.button_index, "button_index", player_count)
        actor_index = _player_index(self.actor_index, "actor_index", player_count)
        pot = _finite_nonnegative(self.pot, "pot")
        to_call = _finite_nonnegative(self.to_call, "to_call")
        if not isinstance(self.legal_actions, LegalActions):
            raise TypeError("legal_actions must be a LegalActions instance")
        if self.legal_actions.check and not math.isclose(to_call, 0.0, abs_tol=1e-9):
            raise ValueError("check cannot be legal when to_call is positive")

        action_history = tuple(_freeze(item) for item in self.action_history)

        object.__setattr__(self, "hero_cards", hero_cards)
        object.__setattr__(self, "board", board)
        object.__setattr__(self, "street", street)
        object.__setattr__(self, "hero_index", hero_index)
        object.__setattr__(self, "button_index", button_index)
        object.__setattr__(self, "actor_index", actor_index)
        object.__setattr__(self, "pot", pot)
        object.__setattr__(self, "to_call", to_call)
        object.__setattr__(self, "stacks", stacks)
        object.__setattr__(self, "street_contributions", street_contributions)
        object.__setattr__(self, "hand_contributions", hand_contributions)
        object.__setattr__(self, "player_statuses", player_statuses)
        object.__setattr__(self, "action_history", action_history)


__all__ = ["Observation"]
