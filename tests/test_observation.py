from dataclasses import FrozenInstanceError
import unittest

from poker_ai.actions import LegalActions
from poker_ai.observation import Observation


def make_observation(**overrides):
    values = {
        "hero_cards": ("As", "Qh"),
        "board": (),
        "street": "preflop",
        "hero_index": 0,
        "button_index": 0,
        "actor_index": 0,
        "pot": 3.0,
        "to_call": 1.0,
        "stacks": (99.0, 98.0),
        "street_contributions": (1.0, 2.0),
        "hand_contributions": (1.0, 2.0),
        "player_statuses": ("active", "active"),
        "action_history": ({"actor_index": 1, "kind": "raise", "raise_to": 2.0},),
        "legal_actions": LegalActions(
            check=False,
            call=True,
            fold=True,
            min_raise_to=4.0,
            max_raise_to=100.0,
        ),
    }
    values.update(overrides)
    return Observation(**values)


class ObservationTests(unittest.TestCase):
    def test_normalizes_sequences_and_is_frozen(self):
        history = [{"actor_index": 1, "tags": ["open"]}]
        observation = make_observation(
            hero_cards=["As", "Qh"],
            stacks=[99, 98],
            action_history=history,
        )

        history[0]["tags"].append("mutated")
        self.assertEqual(observation.hero_cards, ("As", "Qh"))
        self.assertEqual(observation.stacks, (99.0, 98.0))
        self.assertEqual(observation.action_history[0]["tags"], ("open",))
        with self.assertRaises(FrozenInstanceError):
            observation.pot = 4.0

    def test_rejects_private_or_malformed_card_shapes(self):
        with self.assertRaises(ValueError):
            make_observation(hero_cards=("As", "as"))
        with self.assertRaises(ValueError):
            make_observation(hero_cards=("As", "10h"))
        with self.assertRaises(ValueError):
            make_observation(hero_cards=("As", "As"))

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

    def test_requires_aligned_player_arrays_and_valid_indexes(self):
        with self.assertRaises(ValueError):
            make_observation(street_contributions=(1.0,))
        with self.assertRaises(ValueError):
            make_observation(actor_index=2)
        with self.assertRaises(ValueError):
            make_observation(stacks=(99.0,))

    def test_rejects_check_while_facing_a_bet(self):
        with self.assertRaises(ValueError):
            make_observation(
                legal_actions=LegalActions(check=True, call=False, fold=False)
            )

    def test_action_history_rejects_arbitrary_objects(self):
        with self.assertRaises(TypeError):
            make_observation(action_history=(object(),))
        with self.assertRaises(ValueError):
            make_observation(action_history=({"amount": float("nan")},))


class LegalActionsTests(unittest.TestCase):
    def test_raise_bounds_are_a_pair(self):
        with self.assertRaises(ValueError):
            LegalActions(True, False, False, min_raise_to=2.0)
        with self.assertRaises(ValueError):
            LegalActions(False, True, True, min_raise_to=5.0, max_raise_to=4.0)


if __name__ == "__main__":
    unittest.main()
