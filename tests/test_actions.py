from dataclasses import FrozenInstanceError
import unittest

from poker_ai.actions import CandidateAction, LegalActions, generate_candidate_actions
from poker_ai.observation import Observation


def make_observation(
    legal_actions,
    *,
    pot=1000,
    call_cost=200,
    stacks=(1900, 9700),
    contributions=(100, 300),
):
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
        stacks=stacks,
        street_contributions=contributions,
        hand_contributions=contributions,
        statuses=("active", "active"),
        public_history=(),
        legal_actions=legal_actions,
    )


class CandidateActionTests(unittest.TestCase):
    def test_candidate_action_is_validated_and_frozen(self):
        action = CandidateAction(" RAISE ", 600)
        self.assertEqual(action, CandidateAction("raise", 600))
        with self.assertRaises(FrozenInstanceError):
            action.raise_to = 800
        with self.assertRaises(ValueError):
            CandidateAction("call", 200)
        with self.assertRaises(ValueError):
            CandidateAction("raise")

    def test_facing_bet_includes_min_fraction_sizes_and_all_in(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=500, max_raise_to=2000)
        )
        actions = generate_candidate_actions(observation)

        self.assertEqual(actions[:2], (CandidateAction("fold"), CandidateAction("call")))
        self.assertEqual(
            tuple(action.raise_to for action in actions if action.kind == "raise"),
            (500, 900, 1500, 2000),
        )

    def test_no_call_cost_omits_min_raise_and_removes_out_of_range_sizes(self):
        observation = make_observation(
            LegalActions(True, False, True, min_raise_to=800, max_raise_to=2000),
            call_cost=0,
            stacks=(2000, 9700),
            contributions=(0, 0),
        )
        actions = generate_candidate_actions(observation)

        # Half-pot 500 is below the engine minimum and must be removed, not
        # clamped to 800.  The minimum itself is not a no-call candidate.
        self.assertEqual(
            actions,
            (
                CandidateAction("check"),
                CandidateAction("raise", 1000),
                CandidateAction("raise", 2000),
            ),
        )

    def test_fractional_size_uses_floor_before_range_check(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=400, max_raise_to=2000),
            pot=1001,
        )
        raises = tuple(
            action.raise_to
            for action in generate_candidate_actions(observation, pot_fractions=(0.5,))
            if action.kind == "raise"
        )
        # B + C + floor(0.5 * (P + C)) = 100 + 200 + floor(600.5)
        self.assertEqual(raises, (400, 900, 2000))

    def test_all_in_duplicate_is_emitted_once(self):
        observation = make_observation(
            LegalActions(True, False, False, min_raise_to=200, max_raise_to=1000),
            pot=1000,
            call_cost=0,
            stacks=(1000, 9700),
            contributions=(0, 0),
        )
        raises = tuple(
            action.raise_to
            for action in generate_candidate_actions(observation)
            if action.kind == "raise"
        )
        self.assertEqual(raises, (500, 1000))

    def test_stack_not_exceeding_call_cost_merges_all_in_into_call(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=500, max_raise_to=500),
            call_cost=200,
            stacks=(200, 9700),
        )
        self.assertEqual(
            generate_candidate_actions(observation),
            (CandidateAction("fold"), CandidateAction("call")),
        )

    def test_engine_approved_short_all_in_below_minimum_is_kept(self):
        observation = make_observation(
            LegalActions(False, True, True, min_raise_to=500, max_raise_to=400),
            call_cost=100,
            stacks=(300, 9700),
            contributions=(100, 200),
        )
        self.assertEqual(
            generate_candidate_actions(observation),
            (
                CandidateAction("fold"),
                CandidateAction("call"),
                CandidateAction("raise", 400),
            ),
        )

    def test_returns_only_enabled_passive_actions_without_raise_bounds(self):
        observation = make_observation(LegalActions(False, True, True))
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
