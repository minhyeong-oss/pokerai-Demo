from poker_ai.actions import LegalActions, generate_candidate_actions
from poker_ai.observation import Observation
from poker_ai.rollout import BaselineRolloutEstimator


class EquityStub:
    equity = 0.35
    ci95_low = 0.30
    ci95_high = 0.40
    simulations = 10_000


def _observation() -> Observation:
    return Observation(
        hero_cards=("As", "Qh"),
        board=("2c", "7d", "Jh"),
        street="flop",
        hero_index=0,
        button_index=1,
        actor_index=0,
        pot=10,
        to_call=2,
        stacks=(98, 98),
        street_contributions=(0, 2),
        hand_contributions=(2, 4),
        player_statuses=("active", "active"),
        action_history=(),
        legal_actions=LegalActions(False, True, True, 4, 98),
    )


def test_rollout_is_reproducible_and_finite() -> None:
    observation = _observation()
    action = next(a for a in generate_candidate_actions(observation) if a.kind == "raise")
    estimator = BaselineRolloutEstimator("calling", simulations=200, seed=17)
    first = estimator.estimate(observation, action, 0.45)
    second = estimator.estimate(observation, action, 0.45)
    assert first == second
    assert first.samples == 200
    assert first.standard_error is not None and first.standard_error > 0


def test_fold_rollout_is_zero_from_decision_point() -> None:
    observation = _observation()
    action = next(a for a in generate_candidate_actions(observation) if a.kind == "fold")
    result = BaselineRolloutEstimator(simulations=10).estimate(observation, action, 0.1)
    assert result.expected_value == 0
    assert result.ci_low == result.ci_high == 0


def test_call_propagates_equity_confidence_interval() -> None:
    observation = _observation()
    action = next(a for a in generate_candidate_actions(observation) if a.kind == "call")
    result = BaselineRolloutEstimator(simulations=10).estimate(
        observation, action, EquityStub()
    )
    assert result.expected_value == 0.35 * 12 - 2
    assert result.ci_low == 0.30 * 12 - 2
    assert result.ci_high == 0.40 * 12 - 2
