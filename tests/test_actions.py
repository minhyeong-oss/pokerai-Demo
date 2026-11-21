from dataclasses import FrozenInstanceError
import unittest

from poker_ai.actions import (
    CandidateAction,
    LegalActions,
    generate_candidate_actions,
)
from poker_ai.observation import Observation


def make_observation(legal_actions, *, pot=10.0, to_call=2.0):
    return Observation(
        hero_cards=("As", "Qh"),
        board=(),
        street="preflop",
        hero_index=0,
        button_index=0,
        actor_index=0,
        pot=pot,
        to_call=to_call,
        stacks=(99.0, 97.0),
        street_contributions=(1.0, 3.0),
        hand_contributions=(1.0, 3.0),
        player_statuses=("active", "active"),
        action_history=(),
        legal_actions=legal_actions,
    )


class CandidateActionTests(unittest.TestCase):
    def test_candidate_action_is_validated_and_frozen(self):
        action = CandidateAction(" RAISE ", 6)
        self.assertEqual(action, CandidateAction("raise", 6.0))
        with self.assertRaises(FrozenInstanceError):
            action.raise_to = 8.0
        with self.assertRaises(ValueError):
            CandidateAction("call", 2.0)
        with self.assertRaises(ValueError):
            CandidateAction("raise")

    def test_generates_half_pot_pot_min_and_max_raise_to(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=5.0, max_raise_to=20.0)
        )
        actions = generate_candidate_actions(observation)

        self.assertEqual(actions[:2], (CandidateAction("fold"), CandidateAction("call")))
        self.assertEqual(
            tuple(action.raise_to for action in actions if action.kind == "raise"),
            (5.0, 9.0, 15.0, 20.0),
        )

    def test_clamps_then_deduplicates_raise_sizes(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=10.0, max_raise_to=12.0)
        )
        actions = generate_candidate_actions(observation)

        self.assertEqual(
            tuple(action.raise_to for action in actions if action.kind == "raise"),
            (10.0, 12.0),
        )

    def test_returns_only_enabled_passive_actions_without_raise_bounds(self):
        observation = make_observation(
            LegalActions(False, True, True),
        )
        self.assertEqual(
            generate_candidate_actions(observation),
            (CandidateAction("fold"), CandidateAction("call")),
        )

    def test_rejects_negative_pot_fraction(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=5.0, max_raise_to=20.0)
        )
        with self.assertRaises(ValueError):
            generate_candidate_actions(observation, pot_fractions=(-0.5,))


if __name__ == "__main__":
    unittest.main()
