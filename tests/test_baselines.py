from poker_ai.actions import LegalActions, generate_candidate_actions
from poker_ai.baselines import HeuristicBaseline
from poker_ai.observation import Observation


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
        assert HeuristicBaseline(style, simulations=20, policy_hand_seed=4)(observation) in legal


def test_seeded_random_baseline_is_reproducible() -> None:
    observation = _observation()
    first = HeuristicBaseline("random", policy_hand_seed=12)
    second = HeuristicBaseline("random", policy_hand_seed=12)
    assert first(observation) == second(observation)
