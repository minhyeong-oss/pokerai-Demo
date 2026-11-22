from __future__ import annotations

from copy import deepcopy
from itertools import chain

import pytest

from poker_ai.actions import CandidateAction, generate_candidate_actions
from poker_ai.cards import STANDARD_DECK, card_code
from poker_ai.environment import PokerKitHeadsUpHand
from poker_ai.observation import Observation
from poker_ai.observation import ActionEvent, BoardEvent


def _card_partition(hand: PokerKitHeadsUpHand) -> tuple[str, ...]:
    state = hand.state
    cards = chain(
        state.deck_cards,
        chain.from_iterable(state.hole_cards),
        chain.from_iterable(state.board_cards),
        state.burn_cards,
        state.mucked_cards,
        chain.from_iterable(state.discarded_cards),
    )
    return tuple(card_code(card) for card in cards)


def _passive_action(observation: Observation) -> CandidateAction:
    if observation.legal_actions.check:
        return CandidateAction("check")
    if observation.legal_actions.call:
        return CandidateAction("call")
    return CandidateAction("fold")


def test_seeded_and_explicit_decks_are_reproducible_and_unique() -> None:
    first = PokerKitHeadsUpHand(seed=731)
    second = PokerKitHeadsUpHand(seed=731)
    different = PokerKitHeadsUpHand(seed=732)

    first_holes = tuple(tuple(map(card_code, cards)) for cards in first.state.hole_cards)
    second_holes = tuple(tuple(map(card_code, cards)) for cards in second.state.hole_cards)
    different_holes = tuple(
        tuple(map(card_code, cards)) for cards in different.state.hole_cards
    )
    assert first_holes == second_holes
    assert first_holes != different_holes

    explicit = PokerKitHeadsUpHand(deck_order=STANDARD_DECK)
    assert tuple(map(card_code, explicit.state.hole_cards[0])) == ("2c", "2h")
    assert tuple(map(card_code, explicit.state.hole_cards[1])) == ("2d", "2s")

    partition = _card_partition(explicit)
    assert len(partition) == 52
    assert len(set(partition)) == 52


def test_invalid_explicit_deck_is_rejected() -> None:
    duplicate_deck = (*STANDARD_DECK[:-1], STANDARD_DECK[0])
    with pytest.raises(ValueError, match="duplicate card"):
        PokerKitHeadsUpHand(deck_order=duplicate_deck)


def test_observations_are_actor_specific_and_do_not_leak_hidden_cards() -> None:
    hand = PokerKitHeadsUpHand(seed=19, hero_index=0, run_id="env-test")
    assert hand.state.actor_index == 1  # Heads-up button/small blind acts first.

    actor_observation = hand.observation_for(1)
    waiting_observation = hand.observation_for(0)
    opponent_cards = tuple(map(card_code, hand.state.hole_cards[0]))

    assert actor_observation.hero_index == 1
    assert actor_observation.actor_index == 1
    assert actor_observation.button_index == 1
    assert actor_observation.hole_cards == tuple(
        map(card_code, hand.state.hole_cards[1]),
    )
    assert not set(opponent_cards) & set(actor_observation.hole_cards)
    assert not hasattr(actor_observation, "opponent_cards")
    assert not hasattr(actor_observation, "deck")

    # A public snapshot may be requested for the waiting seat, but it receives
    # no legal actions until it actually becomes the actor.
    waiting_legal = waiting_observation.legal_actions
    assert not waiting_legal.check
    assert not waiting_legal.call
    assert not waiting_legal.fold
    assert not waiting_legal.can_raise

    next_observation = hand.apply_action(CandidateAction("raise", 200))
    assert next_observation is not None
    assert next_observation.hero_index == 0
    assert next_observation.actor_index == 0
    assert next_observation.hole_cards == opponent_cards
    event = next_observation.public_history[0]
    assert isinstance(event, ActionEvent)
    assert event.event_id == "env-test:h0:e0"
    assert event.action == "raise"
    assert event.raise_to == 200
    assert event.paid_amount == 150
    assert event.pre_state.pot == 150
    hidden_cards = set(map(card_code, hand.state.hole_cards[1]))
    assert not hidden_cards & set(next_observation.hole_cards + next_observation.board)
    assert all(card not in repr(next_observation.public_history) for card in hidden_cards)


def test_initial_legal_actions_and_raise_to_use_integer_chips() -> None:
    hand = PokerKitHeadsUpHand(seed=2)
    observation = hand.observation_for(1)

    assert observation.street == "preflop"
    assert observation.pot == 150
    assert observation.call_cost == 50
    assert observation.stacks == (9900, 9950)
    assert observation.street_contributions == (100, 50)
    assert observation.hand_contributions == (100, 50)
    assert observation.legal_actions.fold
    assert observation.legal_actions.call
    assert not observation.legal_actions.check
    assert observation.legal_actions.min_raise_to == 200
    assert observation.legal_actions.max_raise_to == 10_000

    next_observation = hand.apply_action(CandidateAction("raise", 201))
    assert next_observation is not None
    assert hand.state.bets == [100, 201]
    assert hand.history[-1].raise_to == 201
    assert next_observation.call_cost == 101
    assert next_observation.legal_actions.min_raise_to == 302


def test_run_hand_decision_and_event_ids_do_not_embed_seed() -> None:
    hand = PokerKitHeadsUpHand(
        seed=987654,
        run_id="public-run",
        hand_sequence=4,
    )
    first = hand.observation_for(1)
    assert first.hand_id == "public-run:h4"
    assert first.decision_id == "public-run:h4:d0"
    assert "987654" not in first.hand_id

    second = hand.apply_action(CandidateAction("call"))
    assert second is not None
    assert second.decision_id == "public-run:h4:d1"
    assert second.public_history[0].event_id == "public-run:h4:e0"


def test_illegal_actions_are_rejected_without_mutating_state() -> None:
    hand = PokerKitHeadsUpHand(seed=4)
    before = (
        tuple(hand.state.stacks),
        tuple(hand.state.bets),
        hand.state.actor_index,
        hand.history,
    )

    with pytest.raises(ValueError, match="check is not legal"):
        hand.apply_action(CandidateAction("check"))
    with pytest.raises(ValueError, match="legal full-raise range"):
        hand.apply_action(CandidateAction("raise", 199))
    with pytest.raises(ValueError, match="legal full-raise range"):
        hand.apply_action(CandidateAction("raise", 10_001))

    after = (
        tuple(hand.state.stacks),
        tuple(hand.state.bets),
        hand.state.actor_index,
        hand.history,
    )
    assert after == before


def test_every_generated_candidate_is_accepted_by_environment() -> None:
    hand = PokerKitHeadsUpHand(seed=27)

    while not hand.is_complete:
        actor_index = hand.state.actor_index
        assert actor_index is not None
        observation = hand.observation_for(actor_index)
        candidates = generate_candidate_actions(observation)
        assert candidates

        # Try every proposed action on an isolated engine copy.  The actual
        # hand then follows the passive line to exercise all four streets.
        for candidate in candidates:
            branch = deepcopy(hand)
            branch.apply_action(candidate)

        hand.apply_action(_passive_action(observation))

    assert len(tuple(event for event in hand.history if isinstance(event, ActionEvent))) == 8
    assert len(tuple(event for event in hand.history if isinstance(event, BoardEvent))) == 3
    assert len(tuple(hand.state.get_board_cards(0))) == 5
    assert sum(hand.stacks) == 20_000
    assert sum(hand.payoffs) == 0


def test_fold_is_settled_exactly_and_chips_are_conserved() -> None:
    hand = PokerKitHeadsUpHand(seed=8)
    result = hand.apply_action(CandidateAction("fold"))

    assert result is None
    assert hand.is_complete
    assert hand.stacks == (10_050, 9_950)
    assert hand.payoffs == (50, -50)
    assert hand.state.total_pot_amount == 0
    assert sum(hand.state.stacks) == 20_000
    assert len(hand.history) == 1
    assert hand.history[0].action == "fold"
    assert hand.history[0].paid_amount == 0


def test_all_in_runs_out_board_and_engine_settles_the_whole_pot() -> None:
    hand = PokerKitHeadsUpHand(seed=11)
    hand.apply_action(CandidateAction("raise", 10_000))
    result = hand.apply_action(CandidateAction("call"))

    assert result is None
    assert hand.is_complete
    assert len(tuple(hand.state.get_board_cards(0))) == 5
    assert hand.state.total_pot_amount == 0
    assert sum(hand.stacks) == 20_000
    assert hand.payoffs == tuple(stack - 10_000 for stack in hand.stacks)
    assert sum(hand.payoffs) == 0
    assert len(_card_partition(hand)) == 52
    assert len(set(_card_partition(hand))) == 52

    operation_names = {type(operation).__name__ for operation in hand.state.operations}
    assert "ChipsPushing" in operation_names
    assert "ChipsPulling" in operation_names


def test_asymmetric_starting_stacks_settle_and_conserve_exact_total() -> None:
    hand = PokerKitHeadsUpHand(
        seed=12,
        starting_stacks=(4_000, 10_000),
        run_id="short-stack",
    )
    assert hand.starting_stack == (4_000, 10_000)
    assert sum(hand.state.stacks) + hand.state.total_pot_amount == 14_000

    hand.apply_action(CandidateAction("raise", 10_000))
    hand.apply_action(CandidateAction("call"))

    assert hand.is_complete
    assert sum(hand.stacks) == 14_000
    assert hand.payoffs == tuple(
        stack - starting
        for stack, starting in zip(hand.stacks, (4_000, 10_000))
    )


def test_short_all_in_candidate_matches_pokerkit_legality() -> None:
    short_raise = PokerKitHeadsUpHand(
        seed=13,
        starting_stacks=(10_000, 150),
        run_id="short-raise",
    )
    observation = short_raise.observation_for(1)
    candidates = generate_candidate_actions(observation)
    assert CandidateAction("raise", 150) in candidates
    short_raise.apply_action(CandidateAction("raise", 150))

    call_only = PokerKitHeadsUpHand(
        seed=14,
        starting_stacks=(10_000, 100),
        run_id="call-only",
    )
    observation = call_only.observation_for(1)
    assert observation.call_cost == observation.stacks[1] == 50
    assert all(
        candidate.kind != "raise"
        for candidate in generate_candidate_actions(observation)
    )


def test_run_passes_only_observations_and_phh_replays_final_stacks() -> None:
    hand = PokerKitHeadsUpHand(seed=101, hero_index=1)
    seen: list[Observation] = []

    def policy(observation: Observation) -> CandidateAction:
        assert isinstance(observation, Observation)
        assert not hasattr(observation, "state")
        seen.append(observation)
        return _passive_action(observation)

    payoffs = hand.run({0: policy, 1: policy})

    assert hand.is_complete
    assert payoffs == hand.payoffs
    assert len(seen) == 8
    assert all(observation.hero_index == observation.actor_index for observation in seen)

    history = hand.to_hand_history()
    replayed_states = list(history)
    assert replayed_states[-1].stacks == hand.state.stacks
