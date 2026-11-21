from dataclasses import dataclass
import unittest

from poker_ai.actions import CandidateAction, LegalActions
from poker_ai.observation import Observation
from poker_ai.policy import RolloutEstimate, SimpleResponseModel, choose_action


@dataclass(frozen=True)
class EquityStub:
    equity: float
    ci95_low: float
    ci95_high: float
    simulations: int = 10_000


def make_observation(
    legal_actions,
    *,
    pot=30.0,
    to_call=10.0,
    stacks=(90.0, 90.0),
    contributions=(10.0, 10.0),
):
    return Observation(
        hero_cards=("As", "Qh"),
        board=(),
        street="preflop",
        hero_index=0,
        button_index=0,
        actor_index=0,
        pot=pot,
        to_call=to_call,
        stacks=stacks,
        street_contributions=contributions,
        hand_contributions=contributions,
        player_statuses=("active", "active"),
        action_history=(),
        legal_actions=legal_actions,
    )


class PolicyTests(unittest.TestCase):
    def test_call_formula_and_confidence_interval(self):
        observation = make_observation(LegalActions(False, True, True))
        decision = choose_action(observation, EquityStub(0.35, 0.30, 0.40))

        self.assertEqual(decision.action, CandidateAction("call"))
        call = next(item for item in decision.evaluations if item.action.kind == "call")
        self.assertAlmostEqual(call.expected_value, 4.0)
        self.assertAlmostEqual(call.ci_low, 2.0)
        self.assertAlmostEqual(call.ci_high, 6.0)

    def test_fold_wins_when_call_ev_is_negative(self):
        observation = make_observation(LegalActions(False, True, True))
        decision = choose_action(observation, EquityStub(0.10, 0.08, 0.12))
        self.assertEqual(decision.action, CandidateAction("fold"))

    def test_rollout_estimate_overrides_showdown_proxy(self):
        observation = make_observation(
            LegalActions(True, False, False, min_raise_to=5.0, max_raise_to=10.0),
            pot=10.0,
            to_call=0.0,
            stacks=(100.0, 100.0),
            contributions=(0.0, 0.0),
        )
        raise_ten = CandidateAction("raise", 10.0)
        decision = choose_action(
            observation,
            EquityStub(0.20, 0.18, 0.22),
            rollout_estimates={
                raise_ten: RolloutEstimate(8.0, standard_error=1.0, samples=100)
            },
        )

        self.assertEqual(decision.action, raise_ten)
        self.assertEqual(decision.chosen_evaluation.source, "rollout")
        self.assertLess(decision.chosen_evaluation.ci_low, 8.0)
        self.assertGreater(decision.chosen_evaluation.ci_high, 8.0)

    def test_simple_response_model_accounts_for_fold_equity(self):
        observation = make_observation(
            LegalActions(True, False, False, min_raise_to=5.0, max_raise_to=5.0),
            pot=10.0,
            to_call=0.0,
            stacks=(100.0, 100.0),
            contributions=(0.0, 0.0),
        )
        decision = choose_action(
            observation,
            EquityStub(0.40, 0.38, 0.42),
            response_model=SimpleResponseModel(fold_probability=0.5),
        )

        self.assertEqual(decision.action, CandidateAction("raise", 5.0))
        self.assertAlmostEqual(decision.chosen_evaluation.expected_value, 6.5)

    def test_short_call_returns_unmatched_raise_chips(self):
        observation = make_observation(
            LegalActions(False, False, False, min_raise_to=20.0, max_raise_to=100.0),
            pot=30.0,
            to_call=0.0,
            stacks=(100.0, 20.0),
            contributions=(0.0, 10.0),
        )
        raise_all_in = CandidateAction("raise", 100.0)
        decision = choose_action(
            observation,
            EquityStub(0.25, 0.25, 0.25),
            response_model=SimpleResponseModel(fold_probability=0.0),
            candidates=(raise_all_in,),
        )
        # Villain can add only 20 BB, so the effective matched hero cost is
        # 30 BB and the called pot is 80 BB: 0.25 * 80 - 30 = -10.
        self.assertAlmostEqual(decision.chosen_evaluation.expected_value, -10.0)

    def test_risk_aversion_can_choose_higher_lower_bound(self):
        observation = make_observation(LegalActions(False, True, True))
        call = CandidateAction("call")
        decision = choose_action(
            observation,
            EquityStub(0.35, 0.30, 0.40),
            rollout_estimates={
                CandidateAction("fold"): 0.0,
                call: RolloutEstimate(4.0, ci_low=-3.0, ci_high=11.0),
            },
            risk_aversion=1.0,
        )
        self.assertEqual(decision.action, CandidateAction("fold"))


if __name__ == "__main__":
    unittest.main()
