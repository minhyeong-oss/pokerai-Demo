from __future__ import annotations

import pytest
from pokerkit import Card, Rank, Suit

from poker_ai.cards import card_code, parse_card, parse_cards, validate_unique_cards


def test_parse_card_accepts_only_canonical_standard_card() -> None:
    card = parse_card("As")

    assert isinstance(card, Card)
    assert card_code(card) == "As"


@pytest.mark.parametrize(
    "raw",
    ["as", "AS", "10s", "As ", " As", "AsKd", "A", "Ax", "??", ""],
)
def test_parse_card_rejects_noncanonical_input(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_card(raw)


def test_parse_card_rejects_unknown_pokerkit_card() -> None:
    with pytest.raises(ValueError, match="standard 52-card deck"):
        parse_card(Card(Rank.UNKNOWN, Suit.UNKNOWN))


def test_parse_cards_accepts_space_delimited_or_iterable_input() -> None:
    assert parse_cards("As Qh") == parse_cards(["As", "Qh"])
    assert parse_cards((), expected_count=0) == ()


def test_parse_cards_rejects_ambiguous_format_and_wrong_count() -> None:
    with pytest.raises(ValueError):
        parse_cards("AsQh")
    with pytest.raises(ValueError, match="exactly one ASCII space"):
        parse_cards("As  Qh")
    with pytest.raises(ValueError, match="exactly 2"):
        parse_cards("As", expected_count=2)


def test_duplicate_validation_catches_within_and_across_fields() -> None:
    with pytest.raises(ValueError, match="duplicate card As"):
        parse_cards(["As", "As"])

    hero = parse_cards("As Qh")
    board = parse_cards("2c As 4d")
    with pytest.raises(ValueError, match="hero.*board"):
        validate_unique_cards(hero, board, field_names=("hero", "board"))


def test_parse_card_rejects_non_string_non_card() -> None:
    with pytest.raises(TypeError):
        parse_card(42)  # type: ignore[arg-type]
