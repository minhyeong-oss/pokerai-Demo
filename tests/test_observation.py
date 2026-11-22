from dataclasses import FrozenInstanceError
import unittest

from poker_ai.actions import CandidateAction, LegalActions
from poker_ai.observation import ActionEvent, BoardEvent, Observation, PreState


def make_pre_state(**overrides):
    values = {
        "board": (),
        "pot": 300,
        "stacks": (9900, 9800),
        "street_contributions": (100, 200),
        "hand_contributions": (100, 200),
        "call_cost": 100,
        "legal_actions": LegalActions(False, True, True, 400, 10_000),
        "candidate_actions": (
            CandidateAction("fold"),
            CandidateAction("call"),
            CandidateAction("raise", 400),
        ),
    }
    values.update(overrides)
    return PreState(**values)


def make_observation(**overrides):
    values = {
        "hand_id": "test-run:h0",
        "decision_id": "test-run:h0:d0",
        "hole_cards": ("As", "Qh"),
        "board": (),
        "street": "preflop",
        "hero_index": 0,
        "button_index": 0,
        "actor_index": 0,
        "pot": 300,
        "call_cost": 100,
        "stacks": (9900, 9800),
        "street_contributions": (100, 200),
        "hand_contributions": (100, 200),
        "statuses": ("active", "active"),
        "public_history": (),
        "legal_actions": LegalActions(False, True, True, 400, 10_000),
    }
    values.update(overrides)
    return Observation(**values)


class ObservationTests(unittest.TestCase):
    def test_normalizes_sequences_and_is_frozen(self):
        event = ActionEvent(
            event_id="test-run:h0:e0",
            actor_index=1,
            street="preflop",
            action="raise",
            paid_amount=150,
            raise_to=200,
            pre_state=make_pre_state(call_cost=50),
        )
        observation = make_observation(
            decision_id="test-run:h0:d1",
            hole_cards=["As", "Qh"],
            stacks=[9900, 9800],
            public_history=[event],
        )

        self.assertEqual(observation.hole_cards, ("As", "Qh"))
        self.assertEqual(observation.stacks, (9900, 9800))
        self.assertEqual(observation.public_history[0].action, "raise")
        with self.assertRaises(FrozenInstanceError):
            observation.pot = 400

    def test_rejects_private_or_malformed_card_shapes(self):
        with self.assertRaises(ValueError):
            make_observation(hole_cards=("As", "as"))
        with self.assertRaises(ValueError):
            make_observation(hole_cards=("As", "10h"))
        with self.assertRaises(ValueError):
            make_observation(hole_cards=("As", "As"))

    def test_enforces_street_board_count_and_visible_uniqueness(self):
        flop = make_observation(
            board=("2c", "7d", "Jh"),
            street="FLOP",
        )
        self.assertEqual(flop.street, "flop")
        with self.assertRaises(ValueError):
            make_observation(board=("2c",), street="flop")
        with self.assertRaises(ValueError):
            make_observation(board=("2c", "7d", "As"), street="flop")

    def test_requires_exactly_two_players_valid_indexes_and_statuses(self):
        with self.assertRaises(ValueError):
            make_observation(street_contributions=(100,))
        with self.assertRaises(ValueError):
            make_observation(actor_index=2)
        with self.assertRaises(ValueError):
            make_observation(stacks=(9900, 9800, 9700))
        with self.assertRaises(ValueError):
            make_observation(statuses=("active", "all_in"))
        self.assertEqual(make_observation(statuses=("active", "all-in")).statuses[1], "all-in")

    def test_requires_integer_chip_amounts(self):
        with self.assertRaises(TypeError):
            make_observation(pot=300.0)
        with self.assertRaises(TypeError):
            LegalActions(False, True, True, 400.0, 10_000)
        with self.assertRaises(TypeError):
            CandidateAction("raise", 400.0)

    def test_rejects_check_while_facing_a_bet(self):
        with self.assertRaises(ValueError):
            make_observation(legal_actions=LegalActions(True, False, False))

    def test_public_history_requires_typed_monotonic_events(self):
        with self.assertRaises(TypeError):
            make_observation(public_history=({"type": "action"},))
        event = ActionEvent(
            event_id="test-run:h0:e1",
            actor_index=1,
            street="preflop",
            action="check",
            paid_amount=0,
            raise_to=None,
            pre_state=make_pre_state(call_cost=0),
        )
        with self.assertRaises(ValueError, msg="event sequence must start at zero"):
            make_observation(public_history=(event,))

    def test_board_event_validates_new_and_total_board(self):
        event = BoardEvent(
            event_id="test-run:h0:e0",
            street="flop",
            cards=("2c", "7d", "Jh"),
            board=("2c", "7d", "Jh"),
        )
        observation = make_observation(
            decision_id="test-run:h0:d1",
            street="flop",
            board=event.board,
            public_history=(event,),
        )
        self.assertEqual(observation.public_history[0].type, "board")

    def test_identifier_format_is_tied_to_hand(self):
        with self.assertRaises(ValueError):
            make_observation(decision_id="another:h0:d0")


class LegalActionsTests(unittest.TestCase):
    def test_raise_bounds_are_a_pair_and_allow_short_all_in(self):
        with self.assertRaises(ValueError):
            LegalActions(True, False, False, min_raise_to=200)
        legal = LegalActions(False, True, True, min_raise_to=500, max_raise_to=400)
        self.assertTrue(legal.can_raise)
        self.assertEqual(legal.max_raise_to, 400)


if __name__ == "__main__":
    unittest.main()
