"""End-to-end practice-hand and baseline-match orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agent import ResearchPokerAgent
from .baselines import HeuristicBaseline
from .cards import card_code
from .environment import PokerKitHeadsUpHand
from .evaluate import EvaluationReport, paired_seat_schedule, summarize_paired_payoffs


@dataclass(frozen=True)
class PracticeHandResult:
    seed: int
    hero_index: int
    button_index: int
    opponent_style: str
    hero_cards: tuple[str, str]
    final_board: tuple[str, ...]
    payoffs_bb: tuple[float, float]
    final_stacks_bb: tuple[float, float]
    hero_payoff_bb: float
    action_history: tuple[dict[str, object], ...]
    decision_records: tuple[dict[str, Any], ...]
    pokerkit_hand_history: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchResult:
    opponent_style: str
    base_seed: int
    report: EvaluationReport
    hands: tuple[PracticeHandResult, ...]

    def as_dict(self, *, include_hands: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "opponent_style": self.opponent_style,
            "base_seed": self.base_seed,
            "report": self.report.as_dict(),
        }
        if include_hands:
            data["hands"] = [hand.as_dict() for hand in self.hands]
        return data


def play_practice_hand(
    *,
    seed: int = 0,
    hero_index: int = 1,
    opponent_style: str = "balanced",
    equity_simulations: int = 2_000,
    rollout_simulations: int = 500,
    baseline_simulations: int = 300,
    risk_aversion: float = 0.0,
    audit_path: str | Path | None = None,
) -> PracticeHandResult:
    """Play and exactly settle one reset-stack PokerKit practice hand."""

    environment = PokerKitHeadsUpHand(seed=seed, hero_index=hero_index)
    initial_hero_observation = environment.observation_for(hero_index)
    hero_cards = initial_hero_observation.hero_cards
    model_style = "balanced" if opponent_style == "random" else opponent_style
    decisions: list[dict[str, Any]] = []

    hero_agent = ResearchPokerAgent(
        opponent_style=model_style,
        equity_simulations=equity_simulations,
        rollout_simulations=rollout_simulations,
        seed=seed * 10_000 + hero_index * 1_000,
        risk_aversion=risk_aversion,
        audit_path=audit_path,
    )
    opponent = HeuristicBaseline(
        opponent_style,
        simulations=baseline_simulations,
        seed=seed * 10_000 + (1 - hero_index) * 1_000 + 500,
    )

    def hero_policy(observation):
        result = hero_agent.decide(observation)
        decisions.append(result.audit_record)
        return result.action

    def opponent_policy(observation):
        action = opponent(observation)
        hero_agent.observe_opponent_action(
            action,
            hero_cards=hero_cards,
            board=observation.board,
            facing_bet=observation.to_call > 0.0,
            event_id=f"{seed}:{len(environment.history)}",
        )
        return action

    policies = {hero_index: hero_policy, 1 - hero_index: opponent_policy}
    try:
        environment.run(policies)
    finally:
        hero_agent.close()

    final_board = tuple(
        card_code(card) for card in environment.state.get_board_cards(0)
    )
    payoffs = environment.payoffs
    return PracticeHandResult(
        seed=seed,
        hero_index=hero_index,
        button_index=environment.button_index,
        opponent_style=opponent_style,
        hero_cards=hero_cards,
        final_board=final_board,
        payoffs_bb=payoffs,
        final_stacks_bb=environment.stacks,
        hero_payoff_bb=payoffs[hero_index],
        action_history=environment.history,
        decision_records=tuple(decisions),
        pokerkit_hand_history=environment.to_hand_history().dumps(),
    )


def evaluate_practice_match(
    *,
    hands: int = 20,
    base_seed: int = 0,
    opponent_style: str = "balanced",
    equity_simulations: int = 500,
    rollout_simulations: int = 200,
    baseline_simulations: int = 150,
    risk_aversion: float = 0.0,
    audit_path: str | Path | None = None,
) -> MatchResult:
    """Evaluate alternating seats with paired deck seeds and a 95% CI."""

    hand_results = []
    for seed, hero_index in paired_seat_schedule(hands, base_seed):
        hand_results.append(
            play_practice_hand(
                seed=seed,
                hero_index=hero_index,
                opponent_style=opponent_style,
                equity_simulations=equity_simulations,
                rollout_simulations=rollout_simulations,
                baseline_simulations=baseline_simulations,
                risk_aversion=risk_aversion,
                audit_path=audit_path,
            )
        )
    report = summarize_paired_payoffs(hand.hero_payoff_bb for hand in hand_results)
    return MatchResult(
        opponent_style=opponent_style,
        base_seed=base_seed,
        report=report,
        hands=tuple(hand_results),
    )


__all__ = [
    "MatchResult",
    "PracticeHandResult",
    "evaluate_practice_match",
    "play_practice_hand",
]
