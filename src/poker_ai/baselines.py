"""Deterministic seeded baseline players for practice evaluation."""

from __future__ import annotations

import random

from .actions import CandidateAction, generate_candidate_actions
from .equity import monte_carlo_equity
from .observation import Observation
from .seeding import derive_analysis_seed, validate_master_seed


class HeuristicBaseline:
    """A small observable-state opponent with named behavioral tendencies."""

    _SUPPORTED = frozenset(("balanced", "tight", "calling", "aggressive", "random"))

    def __init__(
        self,
        style: str = "balanced",
        *,
        simulations: int = 300,
        policy_hand_seed: int = 0,
    ) -> None:
        normalized = style.strip().lower()
        if normalized not in self._SUPPORTED:
            choices = ", ".join(sorted(self._SUPPORTED))
            raise ValueError(f"unknown baseline style {style!r}; choose one of: {choices}")
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        self.style = normalized
        self.simulations = simulations
        self.policy_hand_seed = validate_master_seed(
            policy_hand_seed, "policy_hand_seed"
        )
        self._decision_index = 0

    def __call__(self, observation: Observation) -> CandidateAction:
        candidates = generate_candidate_actions(observation)
        decision_index = self._decision_index
        self._decision_index += 1
        rng = random.Random(
            derive_analysis_seed(
                self.policy_hand_seed,
                purpose="actual_action",
                decision_index=decision_index,
                candidate_id="none",
                sample_index=0,
            )
        )
        if self.style == "random":
            return rng.choice(candidates)

        equity = monte_carlo_equity(
            observation.hole_cards,
            observation.board,
            simulations=self.simulations,
            seed=derive_analysis_seed(
                self.policy_hand_seed,
                purpose="equity",
                decision_index=decision_index,
                candidate_id="none",
                sample_index=0,
            ),
        ).equity

        by_kind = {
            kind: tuple(candidate for candidate in candidates if candidate.kind == kind)
            for kind in ("fold", "check", "call", "raise")
        }
        raises = by_kind["raise"]
        if observation.call_cost > 0:
            pot_odds = observation.call_cost / max(
                observation.pot + observation.call_cost,
                1,
            )
            looseness = {"tight": -0.03, "balanced": 0.03, "calling": 0.14, "aggressive": 0.05}[self.style]
            raise_threshold = {"tight": 0.78, "balanced": 0.68, "calling": 0.83, "aggressive": 0.58}[self.style]
            bluff_frequency = {"tight": 0.01, "balanced": 0.05, "calling": 0.01, "aggressive": 0.16}[self.style]
            if raises and (equity >= raise_threshold or rng.random() < bluff_frequency):
                return self._pick_raise(raises)
            if equity + looseness >= pot_odds and by_kind["call"]:
                return by_kind["call"][0]
            if by_kind["fold"]:
                return by_kind["fold"][0]
        else:
            bet_threshold = {"tight": 0.72, "balanced": 0.60, "calling": 0.78, "aggressive": 0.50}[self.style]
            bluff_frequency = {"tight": 0.01, "balanced": 0.07, "calling": 0.01, "aggressive": 0.20}[self.style]
            if raises and (equity >= bet_threshold or rng.random() < bluff_frequency):
                return self._pick_raise(raises)
            if by_kind["check"]:
                return by_kind["check"][0]

        # Engine abstractions should make this unreachable, but retaining a
        # deterministic legal fallback makes the baseline robust to short all-ins.
        return candidates[0]

    def _pick_raise(self, raises: tuple[CandidateAction, ...]) -> CandidateAction:
        if self.style in ("tight", "calling"):
            return raises[0]
        if self.style == "aggressive":
            return raises[min(len(raises) - 1, max(0, len(raises) // 2))]
        return raises[min(len(raises) - 1, 1)]


__all__ = ["HeuristicBaseline"]
