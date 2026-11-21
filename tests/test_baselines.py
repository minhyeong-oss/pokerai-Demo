from poker_ai.actions import LegalActions, generate_candidate_actions
from poker_ai.baselines import HeuristicBaseline
from poker_ai.observation import Observation


def _observation() -> Observation:
    return Observation(
        hero_cards=("As", "Kd"),
        board=(),
        street="preflop",
        hero_index=1,
        button_index=1,
        actor_index=1,
        pot=1.5,
        to_call=0.5,
        stacks=(99, 99.5),
        street_contributions=(1, 0.5),
        hand_contributions=(1, 0.5),
        player_statuses=("active", "active"),
        action_history=(),
        legal_actions=LegalActions(False, True, True, 2, 100),
    )


def test_every_style_returns_a_legal_candidate() -> None:
    observation = _observation()
    legal = generate_candidate_actions(observation)
    for style in ("balanced", "tight", "calling", "aggressive", "random"):
        assert HeuristicBaseline(style, simulations=20, seed=4)(observation) in legal


def test_seeded_random_baseline_is_reproducible() -> None:
    observation = _observation()
    first = HeuristicBaseline("random", seed=12)
    second = HeuristicBaseline("random", seed=12)
    assert first(observation) == second(observation)
