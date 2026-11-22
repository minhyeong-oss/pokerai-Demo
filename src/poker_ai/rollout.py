"""Public-information baseline action rollouts.

The equity calculator already samples opponent cards and future boards.  This
module samples the remaining action branch (fold or call) and terminal chip
outcome without ever accepting an actual hidden hand.  It is intentionally a
small baseline that can later be replaced by a full PokerKit tree rollout.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, stdev

from .actions import CandidateAction
from .observation import Observation
from .opponents import estimate_fold_probability, get_style, range_fold_probability
from .policy import RolloutEstimate
from .ranges import HandRange
from .seeding import action_candidate_id, derive_analysis_seed, validate_master_seed


def _equity_view(result: object) -> tuple[float, float, float]:
    value = result if isinstance(result, (int, float)) else getattr(result, "equity")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("equity must be between zero and one")
    low = float(getattr(result, "ci95_low", value))
    high = float(getattr(result, "ci95_high", value))
    if not 0.0 <= low <= value <= high <= 1.0:
        raise ValueError("equity confidence interval must contain equity in [0, 1]")
    return value, low, high


@dataclass(frozen=True)
class BaselineRolloutEstimator:
    """Sample fold/call and showdown branches for heads-up action comparison."""

    opponent_style: str = "balanced"
    simulations: int = 2_000
    policy_hand_seed: int = 0
    decision_index: int = 0
    opponent_range: HandRange | None = None

    def __post_init__(self) -> None:
        if isinstance(self.simulations, bool) or not isinstance(self.simulations, int):
            raise TypeError("simulations must be an integer")
        if self.simulations <= 1:
            raise ValueError("simulations must be greater than one")
        validate_master_seed(self.policy_hand_seed, "policy_hand_seed")
        if isinstance(self.decision_index, bool) or not isinstance(self.decision_index, int):
            raise TypeError("decision_index must be an integer")
        if self.decision_index < 0:
            raise ValueError("decision_index cannot be negative")
        get_style(self.opponent_style)
        if self.opponent_range is not None and not isinstance(self.opponent_range, HandRange):
            raise TypeError("opponent_range must be a HandRange or None")

    def estimate(
        self,
        observation: Observation,
        action: CandidateAction,
        equity_result: object,
    ) -> RolloutEstimate:
        equity, equity_low, equity_high = _equity_view(equity_result)
        hero_index = observation.hero_index
        opponent_indices = [
            index
            for index, status in enumerate(observation.statuses)
            if index != hero_index and status != "folded"
        ]
        if len(opponent_indices) != 1:
            raise ValueError("BaselineRolloutEstimator currently supports heads-up play only")
        opponent_index = opponent_indices[0]
        if action.kind == "fold":
            return RolloutEstimate(0.0, ci_low=0.0, ci_high=0.0)

        pot = observation.pot

        if action.kind == "check":
            return RolloutEstimate(
                expected_value=equity * pot,
                ci_low=equity_low * pot,
                ci_high=equity_high * pot,
                samples=getattr(equity_result, "simulations", None),
            )
        elif action.kind == "call":
            cost = observation.call_cost
            final_pot = pot + cost
            return RolloutEstimate(
                expected_value=equity * final_pot - cost,
                ci_low=equity_low * final_pot - cost,
                ci_high=equity_high * final_pot - cost,
                samples=getattr(equity_result, "simulations", None),
            )
        else:
            assert action.kind == "raise" and action.raise_to is not None
            full_hero_cost = min(
                observation.stacks[hero_index],
                max(0.0, action.raise_to - observation.street_contributions[hero_index]),
            )
            opponent_cost = max(
                0.0,
                action.raise_to - observation.street_contributions[opponent_index],
            )
            opponent_cost = min(opponent_cost, observation.stacks[opponent_index])
            # PokerKit returns any unmatched excess when the opponent can only
            # call short.  Use the effective matched contribution, not the full
            # nominal raise, for showdown loss exposure.
            opponent_final = (
                observation.street_contributions[opponent_index] + opponent_cost
            )
            effective_hero_cost = max(
                0.0,
                min(action.raise_to, opponent_final)
                - observation.street_contributions[hero_index],
            )
            effective_hero_cost = min(effective_hero_cost, full_hero_cost)
            called_pot = pot + effective_hero_cost + opponent_cost
            if self.opponent_range is None:
                fold_probability = estimate_fold_probability(
                    self.opponent_style,
                    raise_cost=opponent_cost,
                    pot=pot,
                )
            else:
                fold_probability = range_fold_probability(
                    self.opponent_range,
                    self.opponent_style,
                    hero_cards=observation.hole_cards,
                    board_cards=observation.board,
                    raise_cost=opponent_cost,
                    pot=pot,
                )
            called_ev = equity * called_pot - effective_hero_cost
            rng = random.Random(
                derive_analysis_seed(
                    self.policy_hand_seed,
                    purpose="rollout_response",
                    decision_index=self.decision_index,
                    candidate_id=action_candidate_id(action),
                    sample_index=0,
                )
            )
            outcomes: list[float] = []
            called_count = 0
            for _ in range(self.simulations):
                if rng.random() < fold_probability:
                    outcomes.append(pot)
                else:
                    outcomes.append(called_ev)
                    called_count += 1

        mean = fmean(outcomes)
        standard_error = stdev(outcomes) / math.sqrt(self.simulations)
        response_margin = 1.959963984540054 * standard_error
        called_fraction = called_count / self.simulations
        equity_low_shift = called_fraction * (equity_low - equity) * called_pot
        equity_high_shift = called_fraction * (equity_high - equity) * called_pot
        return RolloutEstimate(
            expected_value=mean,
            standard_error=standard_error,
            samples=self.simulations,
            ci_low=mean - response_margin + equity_low_shift,
            ci_high=mean + response_margin + equity_high_shift,
        )


__all__ = ["BaselineRolloutEstimator"]
