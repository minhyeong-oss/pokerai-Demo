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
    pot=3000,
    call_cost=1000,
    stacks=(9000, 9000),
    contributions=(1000, 1000),
):
    return Observation(
        hand_id="policy:h0",
        decision_id="policy:h0:d0",
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


class PolicyTests(unittest.TestCase):
    def test_call_formula_and_confidence_interval(self):
        observation = make_observation(LegalActions(False, True, True))
        decision = choose_action(observation, EquityStub(0.35, 0.30, 0.40))

        self.assertEqual(decision.action, CandidateAction("call"))
        call = next(item for item in decision.evaluations if item.action.kind == "call")
        self.assertAlmostEqual(call.expected_value, 400.0)
        self.assertAlmostEqual(call.ci_low, 200.0)
        self.assertAlmostEqual(call.ci_high, 600.0)

    def test_fold_wins_when_call_ev_is_negative(self):
        observation = make_observation(LegalActions(False, True, True))
        decision = choose_action(observation, EquityStub(0.10, 0.08, 0.12))
        self.assertEqual(decision.action, CandidateAction("fold"))

    def test_rollout_estimate_overrides_showdown_proxy(self):
        observation = make_observation(
            LegalActions(True, False, False, min_raise_to=500, max_raise_to=1000),
            pot=1000,
            call_cost=0,
            stacks=(10_000, 10_000),
            contributions=(0, 0),
        )
        raise_ten = CandidateAction("raise", 1000)
        decision = choose_action(
            observation,
            EquityStub(0.20, 0.18, 0.22),
            rollout_estimates={
                raise_ten: RolloutEstimate(800.0, standard_error=100.0, samples=100)
            },
        )

        self.assertEqual(decision.action, raise_ten)
        self.assertEqual(decision.chosen_evaluation.source, "rollout")
        self.assertLess(decision.chosen_evaluation.ci_low, 800.0)
        self.assertGreater(decision.chosen_evaluation.ci_high, 800.0)

    def test_simple_response_model_accounts_for_fold_equity(self):
        observation = make_observation(
            LegalActions(True, False, False, min_raise_to=500, max_raise_to=500),
            pot=1000,
            call_cost=0,
            stacks=(10_000, 10_000),
            contributions=(0, 0),
        )
        decision = choose_action(
            observation,
            EquityStub(0.40, 0.38, 0.42),
            response_model=SimpleResponseModel(fold_probability=0.5),
        )

        self.assertEqual(decision.action, CandidateAction("raise", 500))
        self.assertAlmostEqual(decision.chosen_evaluation.expected_value, 650.0)

    def test_short_call_returns_unmatched_raise_chips(self):
        observation = make_observation(
            LegalActions(False, False, False, min_raise_to=2000, max_raise_to=10_000),
            pot=3000,
            call_cost=0,
            stacks=(10_000, 2000),
            contributions=(0, 1000),
        )
        raise_all_in = CandidateAction("raise", 10_000)
        decision = choose_action(
            observation,
            EquityStub(0.25, 0.25, 0.25),
            response_model=SimpleResponseModel(fold_probability=0.0),
            candidates=(raise_all_in,),
        )
        # Villain can add only 20 BB, so the effective matched hero cost is
        # 30 BB and the called pot is 80 BB: 0.25 * 80 - 30 = -10.
        self.assertAlmostEqual(decision.chosen_evaluation.expected_value, -1000.0)

    def test_risk_aversion_can_choose_higher_lower_bound(self):
        observation = make_observation(LegalActions(False, True, True))
        call = CandidateAction("call")
        decision = choose_action(
            observation,
            EquityStub(0.35, 0.30, 0.40),
            rollout_estimates={
                CandidateAction("fold"): 0.0,
                call: RolloutEstimate(400.0, ci_low=-300.0, ci_high=1100.0),
            },
            risk_aversion=1.0,
        )
        self.assertEqual(decision.action, CandidateAction("fold"))


if __name__ == "__main__":
    unittest.main()
