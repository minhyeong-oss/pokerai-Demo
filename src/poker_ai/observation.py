"""Validated public-information contracts passed to poker policies."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Tuple, TypeAlias

from .actions import CandidateAction, LegalActions


_CARD_RE = re.compile(r"^([2-9TJQKA])([cdhs])$")
_BOARD_SIZE_BY_STREET = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
_STATUSES = frozenset(("active", "folded", "all-in"))
_ACTIONS = frozenset(("fold", "check", "call", "raise"))


def _normalise_card(card: object) -> str:
    text = card if isinstance(card, str) else repr(card)
    match = _CARD_RE.fullmatch(text)
    if match is None:
        raise ValueError("invalid card {!r}; use rank+suit notation such as 'As'".format(card))
    rank, suit = match.groups()
    return rank + suit


def _normalise_label(value: object, name: str) -> str:
    if hasattr(value, "value"):
        value = getattr(value, "value")
    text = str(value).strip().lower()
    if not text:
        raise ValueError("{} cannot be empty".format(name))
    return text


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError("{} must be a string".format(name))
    text = value.strip()
    if not text:
        raise ValueError("{} cannot be empty".format(name))
    return text


def _chip_amount(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer chip amount".format(name))
    if value < 0:
        raise ValueError("{} cannot be negative".format(name))
    return value


def _player_index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer".format(name))
    if value not in (0, 1):
        raise ValueError("{} must be 0 or 1".format(name))
    return value


def _cards(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("{} must be a card sequence".format(name))
    cards = tuple(_normalise_card(card) for card in value)
    if len(set(cards)) != len(cards):
        raise ValueError("{} contains a duplicate card".format(name))
    return cards


def _amount_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("{} must be a two-item sequence".format(name))
    amounts = tuple(
        _chip_amount(amount, "{}[{}]".format(name, index))
        for index, amount in enumerate(value)
    )
    if len(amounts) != 2:
        raise ValueError("{} must contain exactly two items".format(name))
    return amounts  # type: ignore[return-value]


@dataclass(frozen=True)
class PreState:
    """Public state captured immediately before one action event."""

    board: Tuple[str, ...]
    pot: int
    stacks: Tuple[int, int]
    street_contributions: Tuple[int, int]
    hand_contributions: Tuple[int, int]
    call_cost: int
    legal_actions: LegalActions
    candidate_actions: Tuple[CandidateAction, ...]

    def __post_init__(self) -> None:
        board = _cards(self.board, "pre_state.board")
        if len(board) not in (0, 3, 4, 5):
            raise ValueError("pre_state.board must contain 0, 3, 4, or 5 cards")
        if not isinstance(self.legal_actions, LegalActions):
            raise TypeError("pre_state.legal_actions must be LegalActions")
        candidates = tuple(self.candidate_actions)
        if not candidates:
            raise ValueError("pre_state.candidate_actions cannot be empty")
        if not all(isinstance(action, CandidateAction) for action in candidates):
            raise TypeError("pre_state.candidate_actions must contain CandidateAction values")
        if len(set(candidates)) != len(candidates):
            raise ValueError("pre_state.candidate_actions cannot contain duplicates")

        object.__setattr__(self, "board", board)
        object.__setattr__(self, "pot", _chip_amount(self.pot, "pre_state.pot"))
        object.__setattr__(self, "stacks", _amount_pair(self.stacks, "pre_state.stacks"))
        object.__setattr__(self, "street_contributions", _amount_pair(
            self.street_contributions, "pre_state.street_contributions"
        ))
        object.__setattr__(self, "hand_contributions", _amount_pair(
            self.hand_contributions, "pre_state.hand_contributions"
        ))
        object.__setattr__(self, "call_cost", _chip_amount(
            self.call_cost, "pre_state.call_cost"
        ))
        object.__setattr__(self, "candidate_actions", candidates)


@dataclass(frozen=True)
class ActionEvent:
    """One public action and the public state that preceded it."""

    event_id: str
    actor_index: int
    street: str
    action: str
    paid_amount: int
    raise_to: int | None
    pre_state: PreState
    type: str = field(default="action", init=False)

    def __post_init__(self) -> None:
        event_id = _nonempty_text(self.event_id, "event_id")
        street = _normalise_label(self.street, "street")
        if street not in _BOARD_SIZE_BY_STREET:
            raise ValueError("unsupported street: {!r}".format(self.street))
        action = _normalise_label(self.action, "action")
        if action not in _ACTIONS:
            raise ValueError("unsupported action: {!r}".format(self.action))
        paid_amount = _chip_amount(self.paid_amount, "paid_amount")
        if action in ("fold", "check") and paid_amount != 0:
            raise ValueError("fold and check events must have paid_amount=0")
        if action == "raise":
            if self.raise_to is None:
                raise ValueError("raise events require raise_to")
            raise_to = _chip_amount(self.raise_to, "raise_to")
        else:
            if self.raise_to is not None:
                raise ValueError("only raise events may specify raise_to")
            raise_to = None
        if not isinstance(self.pre_state, PreState):
            raise TypeError("pre_state must be a PreState")
        if len(self.pre_state.board) != _BOARD_SIZE_BY_STREET[street]:
            raise ValueError("pre_state board does not match the event street")

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "actor_index", _player_index(self.actor_index, "actor_index"))
        object.__setattr__(self, "street", street)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "paid_amount", paid_amount)
        object.__setattr__(self, "raise_to", raise_to)


@dataclass(frozen=True)
class BoardEvent:
    """New public board cards and the resulting complete board."""

    event_id: str
    street: str
    cards: Tuple[str, ...]
    board: Tuple[str, ...]
    type: str = field(default="board", init=False)

    def __post_init__(self) -> None:
        event_id = _nonempty_text(self.event_id, "event_id")
        street = _normalise_label(self.street, "street")
        if street not in ("flop", "turn", "river"):
            raise ValueError("board events require flop, turn, or river street")
        cards = _cards(self.cards, "cards")
        board = _cards(self.board, "board")
        required_cards = 3 if street == "flop" else 1
        if len(cards) != required_cards:
            raise ValueError("{} board events require {} new cards".format(street, required_cards))
        if len(board) != _BOARD_SIZE_BY_STREET[street]:
            raise ValueError("{} board events require {} total cards".format(
                street, _BOARD_SIZE_BY_STREET[street]
            ))
        if board[-len(cards):] != cards:
            raise ValueError("board event cards must be the newly appended board cards")

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "street", street)
        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "board", board)


PublicEvent: TypeAlias = ActionEvent | BoardEvent


@dataclass(frozen=True)
class Observation:
    """Everything a heads-up policy may observe at one decision point."""

    hand_id: str
    decision_id: str
    hole_cards: Tuple[str, str]
    board: Tuple[str, ...]
    street: str
    hero_index: int
    button_index: int
    actor_index: int
    pot: int
    call_cost: int
    stacks: Tuple[int, int]
    street_contributions: Tuple[int, int]
    hand_contributions: Tuple[int, int]
    statuses: Tuple[str, str]
    public_history: Tuple[PublicEvent, ...]
    legal_actions: LegalActions

    def __post_init__(self) -> None:
        hand_id = _nonempty_text(self.hand_id, "hand_id")
        decision_id = _nonempty_text(self.decision_id, "decision_id")
        if re.fullmatch(re.escape(hand_id) + r":d\d+", decision_id) is None:
            raise ValueError("decision_id must use '<hand_id>:d<index>' format")

        hole_cards = _cards(self.hole_cards, "hole_cards")
        if len(hole_cards) != 2:
            raise ValueError("hole_cards must contain exactly two cards")
        board = _cards(self.board, "board")
        street = _normalise_label(self.street, "street")
        if street not in _BOARD_SIZE_BY_STREET:
            raise ValueError("unsupported street: {!r}".format(self.street))
        if len(board) != _BOARD_SIZE_BY_STREET[street]:
            raise ValueError("{} observations require {} board cards".format(
                street, _BOARD_SIZE_BY_STREET[street]
            ))
        if len(set(hole_cards + board)) != len(hole_cards) + len(board):
            raise ValueError("hole_cards and board contain a duplicate card")

        hero_index = _player_index(self.hero_index, "hero_index")
        button_index = _player_index(self.button_index, "button_index")
        actor_index = _player_index(self.actor_index, "actor_index")
        pot = _chip_amount(self.pot, "pot")
        call_cost = _chip_amount(self.call_cost, "call_cost")
        stacks = _amount_pair(self.stacks, "stacks")
        street_contributions = _amount_pair(self.street_contributions, "street_contributions")
        hand_contributions = _amount_pair(self.hand_contributions, "hand_contributions")
        if call_cost > stacks[hero_index]:
            raise ValueError("call_cost cannot exceed the hero's remaining stack")

        statuses = tuple(_normalise_label(value, "statuses") for value in self.statuses)
        if len(statuses) != 2:
            raise ValueError("statuses must contain exactly two items")
        if any(status not in _STATUSES for status in statuses):
            raise ValueError("statuses may contain only active, folded, or all-in")

        if not isinstance(self.legal_actions, LegalActions):
            raise TypeError("legal_actions must be a LegalActions instance")
        if self.legal_actions.check and call_cost != 0:
            raise ValueError("check cannot be legal when call_cost is positive")

        public_history = tuple(self.public_history)
        if not all(isinstance(event, (ActionEvent, BoardEvent)) for event in public_history):
            raise TypeError("public_history must contain only ActionEvent or BoardEvent values")
        event_ids = tuple(event.event_id for event in public_history)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("public_history event_id values must be unique")
        for index, event_id in enumerate(event_ids):
            if event_id != "{}:e{}".format(hand_id, index):
                raise ValueError("public_history event_id values must be monotonic from zero")
        board_events = tuple(event for event in public_history if isinstance(event, BoardEvent))
        if board_events and board_events[-1].board != board:
            raise ValueError("the latest board event must match the observation board")

        object.__setattr__(self, "hand_id", hand_id)
        object.__setattr__(self, "decision_id", decision_id)
        object.__setattr__(self, "hole_cards", hole_cards)
        object.__setattr__(self, "board", board)
        object.__setattr__(self, "street", street)
        object.__setattr__(self, "hero_index", hero_index)
        object.__setattr__(self, "button_index", button_index)
        object.__setattr__(self, "actor_index", actor_index)
        object.__setattr__(self, "pot", pot)
        object.__setattr__(self, "call_cost", call_cost)
        object.__setattr__(self, "stacks", stacks)
        object.__setattr__(self, "street_contributions", street_contributions)
        object.__setattr__(self, "hand_contributions", hand_contributions)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "public_history", public_history)


__all__ = ["ActionEvent", "BoardEvent", "Observation", "PreState", "PublicEvent"]
