"""End-to-end practice-hand and paired baseline-match orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import ResearchPokerAgent
from .baselines import HeuristicBaseline
from .cards import card_code
from .environment import PokerKitHeadsUpHand, UNITS_PER_BB
from .evaluate import EvaluationReport, summarize_paired_payoffs
from .observation import PublicEvent
from .seeding import derive_seed, validate_master_seed


@dataclass(frozen=True)
class PracticeHandResult:
    deck_seed: int
    run_id: str
    hand_id: str
    hero_index: int
    button_index: int
    opponent_style: str
    hero_cards: tuple[str, str]
    final_board: tuple[str, ...]
    payoffs_bb: tuple[float, float]
    final_stacks_bb: tuple[float, float]
    hero_payoff_bb: float
    action_history: tuple[PublicEvent, ...]
    decision_records: tuple[dict[str, Any], ...]
    pokerkit_hand_history: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchResult:
    opponent_style: str
    environment_master_seed: int
    hero_policy_master_seed: int
    villain_policy_master_seed: int
    report: EvaluationReport
    hands: tuple[PracticeHandResult, ...]

    def as_dict(self, *, include_hands: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "opponent_style": self.opponent_style,
            "environment_master_seed": self.environment_master_seed,
            "hero_policy_master_seed": self.hero_policy_master_seed,
            "villain_policy_master_seed": self.villain_policy_master_seed,
            "report": self.report.as_dict(),
        }
        if include_hands:
            data["hands"] = [hand.as_dict() for hand in self.hands]
        return data


def _bb(amount: int) -> float:
    return amount / UNITS_PER_BB


def play_practice_hand(
    *,
    environment_master_seed: int = 0,
    hero_policy_master_seed: int = 1,
    villain_policy_master_seed: int = 2,
    hand_sequence: int = 0,
    run_id: str | None = None,
    hero_index: int = 1,
    opponent_style: str = "balanced",
    equity_simulations: int = 2_000,
    rollout_simulations: int = 500,
    baseline_simulations: int = 300,
    risk_aversion: float = 0.0,
    audit_path: str | Path | None = None,
    _pair_index: int | None = None,
    _leg_index: int = 0,
) -> PracticeHandResult:
    """Play and settle one hand with independent environment/policy seeds."""

    environment_master_seed = validate_master_seed(
        environment_master_seed, "environment_master_seed"
    )
    hero_policy_master_seed = validate_master_seed(
        hero_policy_master_seed, "hero_policy_master_seed"
    )
    villain_policy_master_seed = validate_master_seed(
        villain_policy_master_seed, "villain_policy_master_seed"
    )
    if isinstance(hand_sequence, bool) or not isinstance(hand_sequence, int):
        raise TypeError("hand_sequence must be an integer")
    if hand_sequence < 0:
        raise ValueError("hand_sequence cannot be negative")

    if _pair_index is None:
        deck_seed = derive_seed(
            environment_master_seed, "deck", "single", hand_sequence
        )
        hero_hand_seed = derive_seed(
            hero_policy_master_seed,
            "single",
            hand_sequence,
            hero_index,
            "hero",
        )
        villain_hand_seed = derive_seed(
            villain_policy_master_seed,
            "single",
            hand_sequence,
            1 - hero_index,
            f"baseline:{opponent_style}",
        )
    else:
        deck_seed = derive_seed(environment_master_seed, "deck", _pair_index)
        hero_hand_seed = derive_seed(
            hero_policy_master_seed,
            _pair_index,
            _leg_index,
            hero_index,
            "hero",
        )
        villain_hand_seed = derive_seed(
            villain_policy_master_seed,
            _pair_index,
            _leg_index,
            1 - hero_index,
            f"baseline:{opponent_style}",
        )

    environment = PokerKitHeadsUpHand(
        seed=deck_seed,
        hero_index=hero_index,
        run_id=run_id,
        hand_sequence=hand_sequence,
    )
    initial_hero_observation = environment.observation_for(hero_index)
    hero_cards = initial_hero_observation.hole_cards
    model_style = "balanced" if opponent_style == "random" else opponent_style
    decisions: list[dict[str, Any]] = []

    hero_agent = ResearchPokerAgent(
        opponent_style=model_style,
        equity_simulations=equity_simulations,
        rollout_simulations=rollout_simulations,
        policy_hand_seed=hero_hand_seed,
        risk_aversion=risk_aversion,
        audit_path=audit_path,
    )
    opponent = HeuristicBaseline(
        opponent_style,
        simulations=baseline_simulations,
        policy_hand_seed=villain_hand_seed,
    )

    def hero_policy(observation):
        result = hero_agent.decide(observation)
        decisions.append(result.audit_record)
        return result.action

    def opponent_policy(observation):
        action = opponent(observation)
        hero_agent.observe_opponent_action(
            action,
            hole_cards=hero_cards,
            board=observation.board,
            facing_bet=observation.call_cost > 0,
            event_id=f"{environment.hand_id}:e{len(environment.history)}",
        )
        return action

    policies = {hero_index: hero_policy, 1 - hero_index: opponent_policy}
    try:
        environment.run(policies)
    finally:
        hero_agent.close()

    final_board = tuple(card_code(card) for card in environment.state.get_board_cards(0))
    payoffs_bb = tuple(_bb(amount) for amount in environment.payoffs)
    stacks_bb = tuple(_bb(amount) for amount in environment.stacks)
    return PracticeHandResult(
        deck_seed=deck_seed,
        run_id=environment.run_id,
        hand_id=environment.hand_id,
        hero_index=hero_index,
        button_index=environment.button_index,
        opponent_style=opponent_style,
        hero_cards=hero_cards,
        final_board=final_board,
        payoffs_bb=payoffs_bb,  # type: ignore[arg-type]
        final_stacks_bb=stacks_bb,  # type: ignore[arg-type]
        hero_payoff_bb=payoffs_bb[hero_index],
        action_history=environment.history,
        decision_records=tuple(decisions),
        pokerkit_hand_history=environment.to_hand_history().dumps(),
    )


def evaluate_practice_match(
    *,
    hands: int = 20,
    environment_master_seed: int = 0,
    hero_policy_master_seed: int = 1,
    villain_policy_master_seed: int = 2,
    run_id: str | None = None,
    opponent_style: str = "balanced",
    equity_simulations: int = 500,
    rollout_simulations: int = 200,
    baseline_simulations: int = 150,
    risk_aversion: float = 0.0,
    audit_path: str | Path | None = None,
) -> MatchResult:
    """Evaluate swapped seats with the same deck in each two-hand pair."""

    if isinstance(hands, bool) or not isinstance(hands, int) or hands <= 0:
        raise ValueError("hands must be a positive integer")
    if hands % 2:
        raise ValueError("paired seat evaluation requires an even number of hands")
    run_id = str(uuid4()) if run_id is None else run_id

    hand_results = []
    for pair_index in range(hands // 2):
        for leg_index, hero_index in enumerate((0, 1)):
            hand_sequence = pair_index * 2 + leg_index
            hand_results.append(
                play_practice_hand(
                    environment_master_seed=environment_master_seed,
                    hero_policy_master_seed=hero_policy_master_seed,
                    villain_policy_master_seed=villain_policy_master_seed,
                    hand_sequence=hand_sequence,
                    run_id=run_id,
                    hero_index=hero_index,
                    opponent_style=opponent_style,
                    equity_simulations=equity_simulations,
                    rollout_simulations=rollout_simulations,
                    baseline_simulations=baseline_simulations,
                    risk_aversion=risk_aversion,
                    audit_path=audit_path,
                    _pair_index=pair_index,
                    _leg_index=leg_index,
                )
            )
    report = summarize_paired_payoffs(hand.hero_payoff_bb for hand in hand_results)
    return MatchResult(
        opponent_style=opponent_style,
        environment_master_seed=environment_master_seed,
        hero_policy_master_seed=hero_policy_master_seed,
        villain_policy_master_seed=villain_policy_master_seed,
        report=report,
        hands=tuple(hand_results),
    )


__all__ = ["MatchResult", "PracticeHandResult", "evaluate_practice_match", "play_practice_hand"]
