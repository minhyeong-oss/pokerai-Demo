import random

from poker_ai.actions import LegalActions, generate_candidate_actions
from poker_ai.baselines import HeuristicBaseline
from poker_ai.observation import Observation
from poker_ai.opponents import OpponentActionModel
from poker_ai.seeding import derive_analysis_seed


def _observation() -> Observation:
    return Observation(
        hand_id="baseline:h0",
        decision_id="baseline:h0:d0",
        hole_cards=("As", "Kd"),
        board=(),
        street="preflop",
        hero_index=1,
        button_index=1,
        actor_index=1,
        pot=150,
        call_cost=50,
        stacks=(9900, 9950),
        street_contributions=(100, 50),
        hand_contributions=(100, 50),
        statuses=("active", "active"),
        public_history=(),
        legal_actions=LegalActions(False, True, True, 200, 10_000),
    )


def test_every_style_returns_a_legal_candidate() -> None:
    observation = _observation()
    legal = generate_candidate_actions(observation)
    for style in ("balanced", "tight", "calling", "aggressive", "random"):
        assert HeuristicBaseline(style, policy_hand_seed=4)(observation) in legal


def test_seeded_random_baseline_is_reproducible() -> None:
    observation = _observation()
    first = HeuristicBaseline("random", policy_hand_seed=12)
    second = HeuristicBaseline("random", policy_hand_seed=12)
    assert first(observation) == second(observation)


def test_baseline_uses_the_shared_action_model_and_parameters() -> None:
    baseline = HeuristicBaseline("aggressive", policy_hand_seed=3)
    assert isinstance(baseline.model, OpponentActionModel)
    assert baseline.parameters == OpponentActionModel("aggressive").parameters.as_dict()
    assert "simulations" not in baseline.__dict__

    observation = _observation()
    candidates = generate_candidate_actions(observation)
    probabilities = baseline.model.probabilities(
        observation.hole_cards,
        board_cards=observation.board,
        pot=observation.pot,
        call_cost=observation.call_cost,
        own_street_contribution=observation.street_contributions[observation.hero_index],
        candidate_actions=candidates,
    )
    seed = derive_analysis_seed(
        3,
        purpose="actual_action",
        decision_index=0,
        candidate_id="none",
        sample_index=0,
    )
    threshold = random.Random(seed).random()
    cumulative = 0.0
    expected = candidates[-1]
    for candidate, probability in probabilities.items():
        cumulative += probability
        if threshold < cumulative:
            expected = candidate
            break
    assert baseline(observation) == expected
