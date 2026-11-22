from poker_ai.actions import CandidateAction, LegalActions
from poker_ai.agent import ResearchPokerAgent
from poker_ai.observation import Observation
import pytest


def _observation() -> Observation:
    return Observation(
        hand_id="agent:h0",
        decision_id="agent:h0:d0",
        hole_cards=("As", "Qs"),
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


def test_agent_decision_is_replayable_from_seed() -> None:
    first = ResearchPokerAgent(equity_simulations=100, rollout_simulations=50, policy_hand_seed=7)
    second = ResearchPokerAgent(equity_simulations=100, rollout_simulations=50, policy_hand_seed=7)
    result1 = first.decide(_observation())
    result2 = second.decide(_observation())
    assert result1.action == result2.action
    assert result1.equity == result2.equity
    assert result1.audit_record["metadata"] == result2.audit_record["metadata"]


def test_observed_action_changes_range_revision() -> None:
    agent = ResearchPokerAgent(equity_simulations=10, rollout_simulations=10)
    applied = agent.observe_opponent_action(
        CandidateAction("raise", 300),
        hole_cards=("As", "Qs"),
        board=(),
        facing_bet=False,
        event_id="hand-1:action-0",
    )
    assert applied
    version = agent.range_version
    assert ":r1:" in version
    assert len(agent.opponent_range) == 1225
    assert not agent.observe_opponent_action(
        CandidateAction("raise", 300),
        hole_cards=("As", "Qs"),
        board=(),
        facing_bet=False,
        event_id="hand-1:action-0",
    )
    assert agent.range_version == version
    with pytest.raises(ValueError, match="reused"):
        agent.observe_opponent_action(
            CandidateAction("check"),
            hole_cards=("As", "Qs"),
            board=(),
            facing_bet=False,
            event_id="hand-1:action-0",
        )


def test_observation_contract_rejects_multiway_before_agent() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="exactly two"):
        Observation(
            hand_id=observation.hand_id,
            decision_id=observation.decision_id,
            hole_cards=observation.hole_cards,
            board=observation.board,
            street=observation.street,
            hero_index=0,
            button_index=1,
            actor_index=0,
            pot=300,
            call_cost=100,
            stacks=(9900, 9900, 9900),
            street_contributions=(0, 100, 100),
            hand_contributions=(0, 100, 100),
            statuses=("active", "active", "active"),
            public_history=(),
            legal_actions=LegalActions(False, True, True),
        )
