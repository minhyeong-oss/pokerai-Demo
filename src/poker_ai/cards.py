"""Strict card parsing and validation for standard Texas Hold'em cards.

PokerKit deliberately accepts several convenient input formats.  The public
boundary of this project is stricter so malformed observations fail loudly
instead of being silently repaired.  A card code is exactly two characters:
an upper-case rank followed by a lower-case suit (for example, ``"As"``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeAlias

from pokerkit import Card, Deck

CardLike: TypeAlias = Card | str
CardsLike: TypeAlias = CardLike | Iterable[CardLike]

_CARD_PATTERN = re.compile(r"[2-9TJQKA][cdhs]\Z")
STANDARD_DECK: tuple[Card, ...] = tuple(Deck.STANDARD)
_STANDARD_CARDS = frozenset(STANDARD_DECK)


def parse_card(value: CardLike) -> Card:
    """Return one validated standard-deck card.

    String values must use the canonical two-character form.  In particular,
    whitespace, comma separation, ``"10s"``, lower-case ranks, unknown cards,
    and strings containing more than one card are rejected.

    Args:
        value: A canonical card code or a PokerKit :class:`~pokerkit.Card`.

    Raises:
        TypeError: If *value* is neither ``str`` nor ``Card``.
        ValueError: If the value is not one canonical standard card.
    """

    if isinstance(value, Card):
        if value not in _STANDARD_CARDS:
            raise ValueError(f"card is not in the standard 52-card deck: {value!r}")
        return value

    if not isinstance(value, str):
        raise TypeError(f"card must be a string or pokerkit.Card, got {type(value).__name__}")
    if _CARD_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"invalid card code {value!r}; expected exactly [2-9TJQKA][cdhs], e.g. 'As'",
        )

    # The regular expression has already made PokerKit's permissive parser a
    # strict, single-card operation.
    return next(Card.parse(value))


def parse_cards(
    values: CardsLike | None,
    *,
    expected_count: int | None = None,
    unique: bool = True,
    field_name: str = "cards",
) -> tuple[Card, ...]:
    """Parse a collection of cards and optionally enforce its size.

    A string collection is whitespace-delimited (``"As Qh"``).  Concatenated
    PokerKit notation such as ``"AsQh"`` is intentionally rejected.  Passing a
    single :class:`~pokerkit.Card` is equivalent to a one-item collection.

    Args:
        values: Card objects/codes, or a whitespace-delimited string.
        expected_count: Required number of parsed cards, when supplied.
        unique: Whether duplicates within this collection are forbidden.
        field_name: Name used in validation error messages.
    """

    if values is None:
        raw_values: tuple[CardLike, ...] = ()
    elif isinstance(values, Card):
        raw_values = (values,)
    elif isinstance(values, str):
        if values == "":
            raw_values = ()
        else:
            raw_values = tuple(values.split(" "))
            if any(token == "" for token in raw_values):
                raise ValueError(
                    f"{field_name} must use exactly one ASCII space between card codes",
                )
    else:
        try:
            raw_values = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{field_name} must be an iterable of cards") from exc

    cards = tuple(parse_card(value) for value in raw_values)

    if expected_count is not None:
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise TypeError("expected_count must be an integer")
        if expected_count < 0:
            raise ValueError("expected_count cannot be negative")
        if len(cards) != expected_count:
            raise ValueError(
                f"{field_name} must contain exactly {expected_count} cards, got {len(cards)}",
            )

    if unique:
        validate_unique_cards(cards, field_names=(field_name,))

    return cards


def validate_unique_cards(
    *groups: Iterable[Card],
    field_names: Iterable[str] | None = None,
) -> None:
    """Raise if a card occurs more than once across one or more groups.

    This function is used at every public calculation boundary to catch both
    duplicates inside a field and collisions between fields (for example a
    hero card repeated on the board).
    """

    names = tuple(field_names) if field_names is not None else ()
    if names and len(names) != len(groups):
        raise ValueError("field_names must contain one name for each card group")

    first_seen: dict[Card, str] = {}
    for group_index, group in enumerate(groups):
        group_name = names[group_index] if names else f"group {group_index + 1}"
        for raw_card in group:
            card = parse_card(raw_card)
            if card in first_seen:
                raise ValueError(
                    f"duplicate card {card!r}: present in both {first_seen[card]} and {group_name}",
                )
            first_seen[card] = group_name


def card_code(card: CardLike) -> str:
    """Return the canonical two-character code for a validated card."""

    return repr(parse_card(card))


__all__ = [
    "CardLike",
    "CardsLike",
    "STANDARD_DECK",
    "card_code",
    "parse_card",
    "parse_cards",
    "validate_unique_cards",
]
