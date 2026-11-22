from dataclasses import FrozenInstanceError
import unittest

from poker_ai.actions import (
    CandidateAction,
    LegalActions,
    generate_candidate_actions,
)
from poker_ai.observation import Observation


def make_observation(legal_actions, *, pot=1000, call_cost=200):
    return Observation(
        hand_id="test:h0",
        decision_id="test:h0:d0",
        hole_cards=("As", "Qh"),
        board=(),
        street="preflop",
        hero_index=0,
        button_index=0,
        actor_index=0,
        pot=pot,
        call_cost=call_cost,
        stacks=(9900, 9700),
        street_contributions=(100, 300),
        hand_contributions=(100, 300),
        statuses=("active", "active"),
        public_history=(),
        legal_actions=legal_actions,
    )


class CandidateActionTests(unittest.TestCase):
    def test_candidate_action_is_validated_and_frozen(self):
        action = CandidateAction(" RAISE ", 6)
        self.assertEqual(action, CandidateAction("raise", 6))
        with self.assertRaises(FrozenInstanceError):
            action.raise_to = 8
        with self.assertRaises(ValueError):
            CandidateAction("call", 2)
        with self.assertRaises(ValueError):
            CandidateAction("raise")

    def test_generates_half_pot_pot_min_and_max_raise_to(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=500, max_raise_to=2000)
        )
        actions = generate_candidate_actions(observation)

        self.assertEqual(actions[:2], (CandidateAction("fold"), CandidateAction("call")))
        self.assertEqual(
            tuple(action.raise_to for action in actions if action.kind == "raise"),
            (500, 900, 1500, 2000),
        )

    def test_clamps_then_deduplicates_raise_sizes(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=1000, max_raise_to=1200)
        )
        actions = generate_candidate_actions(observation)

        self.assertEqual(
            tuple(action.raise_to for action in actions if action.kind == "raise"),
            (1000, 1200),
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
            LegalActions(False, True, True, min_raise_to=500, max_raise_to=2000)
        )
        with self.assertRaises(ValueError):
            generate_candidate_actions(observation, pot_fractions=(-0.5,))


if __name__ == "__main__":
    unittest.main()
