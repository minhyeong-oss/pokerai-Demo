from poker_ai.actions import CandidateAction, LegalActions
from poker_ai.agent import ResearchPokerAgent
from poker_ai.environment import PokerKitHeadsUpHand
from poker_ai.observation import BoardEvent, Observation
from poker_ai.range_snapshots import RangeSnapshotStore
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
    candidates = (CandidateAction("check"), CandidateAction("raise", 300))
    applied = agent.observe_opponent_action(
        CandidateAction("raise", 300),
        event_id="hand-1:action-0",
        hole_cards=("As", "Qs"),
        board=(),
        pot=150,
        call_cost=0,
        own_street_contribution=0,
        candidate_actions=candidates,
    )
    assert applied
    version = agent.range_version
    assert ":r1:" in version
    assert len(agent.opponent_range) == 1225
    assert not agent.observe_opponent_action(
        CandidateAction("raise", 300),
        event_id="hand-1:action-0",
        hole_cards=("As", "Qs"),
        board=(),
        pot=150,
        call_cost=0,
        own_street_contribution=0,
        candidate_actions=candidates,
    )
    assert agent.range_version == version
    with pytest.raises(ValueError, match="reused"):
        agent.observe_opponent_action(
            CandidateAction("check"),
            event_id="hand-1:action-0",
            hole_cards=("As", "Qs"),
            board=(),
            pot=150,
            call_cost=0,
            own_street_contribution=0,
            candidate_actions=candidates,
        )


def test_agent_automatically_processes_unseen_public_history() -> None:
    hand = PokerKitHeadsUpHand(seed=19, hero_index=0, run_id="agent-events")
    next_observation = hand.apply_action(CandidateAction("raise", 200))
    assert next_observation is not None and next_observation.hero_index == 0
    agent = ResearchPokerAgent(
        equity_simulations=10,
        rollout_simulations=10,
        policy_hand_seed=7,
    )
    agent.decide(next_observation)
    version = agent.range_version
    assert ":r1:" in version  # action update also applies the hero blockers
    assert next_observation.public_history[0].event_id in agent._processed_event_ids
    agent.decide(next_observation)
    assert agent.range_version == version


def test_agent_persists_snapshot_before_referencing_it(tmp_path) -> None:
    audit_path = tmp_path / "decisions.jsonl"
    snapshot_root = tmp_path / "snapshots"
    with ResearchPokerAgent(
        equity_simulations=10,
        rollout_simulations=10,
        policy_hand_seed=8,
        audit_path=audit_path,
        snapshot_root=snapshot_root,
        policy_id="hero-test",
    ) as agent:
        result = agent.decide(_observation())
    content_id = result.audit_record["metadata"]["range_snapshot_id"]
    restored = RangeSnapshotStore(snapshot_root, "hero-test").load(content_id)
    assert restored.masses == agent.opponent_range.masses
    assert content_id in audit_path.read_text(encoding="utf-8")


def test_agent_removes_new_board_blockers_from_public_event() -> None:
    board = ("2c", "7d", "Jh")
    observation = Observation(
        hand_id="board:h0",
        decision_id="board:h0:d1",
        hole_cards=("As", "Qs"),
        board=board,
        street="flop",
        hero_index=0,
        button_index=1,
        actor_index=0,
        pot=200,
        call_cost=0,
        stacks=(9900, 9900),
        street_contributions=(0, 0),
        hand_contributions=(100, 100),
        statuses=("active", "active"),
        public_history=(
            BoardEvent(
                event_id="board:h0:e0",
                street="flop",
                cards=board,
                board=board,
            ),
        ),
        legal_actions=LegalActions(True, False, True, 100, 9900),
    )
    agent = ResearchPokerAgent(
        equity_simulations=10,
        rollout_simulations=10,
        policy_hand_seed=9,
    )
    agent.decide(observation)
    assert len(agent.opponent_range) == 1081  # C(47, 2)


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
