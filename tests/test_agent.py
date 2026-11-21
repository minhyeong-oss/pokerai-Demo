from poker_ai.actions import CandidateAction, LegalActions
from poker_ai.agent import ResearchPokerAgent
from poker_ai.observation import Observation
import pytest


def _observation() -> Observation:
    return Observation(
        hero_cards=("As", "Qs"),
        board=(),
        street="preflop",
        hero_index=1,
        button_index=1,
        actor_index=1,
        pot=1.5,
        to_call=0.5,
        stacks=(99.0, 99.5),
        street_contributions=(1.0, 0.5),
        hand_contributions=(1.0, 0.5),
        player_statuses=("active", "active"),
        action_history=(),
        legal_actions=LegalActions(False, True, True, 2.0, 100.0),
    )


def test_agent_decision_is_replayable_from_seed() -> None:
    first = ResearchPokerAgent(equity_simulations=100, rollout_simulations=50, seed=7)
    second = ResearchPokerAgent(equity_simulations=100, rollout_simulations=50, seed=7)
    result1 = first.decide(_observation())
    result2 = second.decide(_observation())
    assert result1.action == result2.action
    assert result1.equity == result2.equity
    assert result1.audit_record["metadata"] == result2.audit_record["metadata"]


def test_observed_action_changes_range_revision() -> None:
    agent = ResearchPokerAgent(equity_simulations=10, rollout_simulations=10)
    applied = agent.observe_opponent_action(
        CandidateAction("raise", 3.0),
        hero_cards=("As", "Qs"),
        board=(),
        facing_bet=False,
        event_id="hand-1:action-0",
    )
    assert applied
    version = agent.range_version
    assert ":r1:" in version
    assert len(agent.opponent_range) == 1225
    assert not agent.observe_opponent_action(
        CandidateAction("raise", 3.0),
        hero_cards=("As", "Qs"),
        board=(),
        facing_bet=False,
        event_id="hand-1:action-0",
    )
    assert agent.range_version == version
    with pytest.raises(ValueError, match="reused"):
        agent.observe_opponent_action(
            CandidateAction("check"),
            hero_cards=("As", "Qs"),
            board=(),
            facing_bet=False,
            event_id="hand-1:action-0",
        )


def test_heads_up_agent_rejects_multiway_observation_early() -> None:
    observation = _observation()
    multiway = Observation(
        hero_cards=observation.hero_cards,
        board=observation.board,
        street=observation.street,
        hero_index=0,
        button_index=1,
        actor_index=0,
        pot=3.0,
        to_call=1.0,
        stacks=(99.0, 99.0, 99.0),
        street_contributions=(0.0, 1.0, 1.0),
        hand_contributions=(0.0, 1.0, 1.0),
        player_statuses=("active", "active", "active"),
        action_history=(),
        legal_actions=LegalActions(False, True, True),
    )
    with pytest.raises(ValueError, match="heads-up"):
        ResearchPokerAgent(equity_simulations=10, rollout_simulations=10).decide(multiway)
