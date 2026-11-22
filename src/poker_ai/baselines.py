"""Deterministic seeded baseline players for practice evaluation."""

from __future__ import annotations

import math
import random

from .actions import CandidateAction, generate_candidate_actions
from .observation import Observation
from .opponents import OpponentActionModel
from .seeding import derive_analysis_seed, validate_master_seed


class HeuristicBaseline:
    """A small observable-state opponent with named behavioral tendencies."""

    def __init__(
        self,
        style: str = "balanced",
        *,
        policy_hand_seed: int = 0,
    ) -> None:
        self.model = OpponentActionModel(style)
        self.style = self.model.name
        self.policy_hand_seed = validate_master_seed(
            policy_hand_seed, "policy_hand_seed"
        )

    @property
    def parameters(self) -> dict[str, float | str]:
        return self.model.parameters.as_dict()

    def __call__(self, observation: Observation) -> CandidateAction:
        candidates = generate_candidate_actions(observation)
        decision_index = int(observation.decision_id.rsplit(":d", 1)[1])
        rng = random.Random(
            derive_analysis_seed(
                self.policy_hand_seed,
                purpose="actual_action",
                decision_index=decision_index,
                candidate_id="none",
                sample_index=0,
            )
        )
        probabilities = self.model.probabilities(
            observation.hole_cards,
            board_cards=observation.board,
            pot=observation.pot,
            call_cost=observation.call_cost,
            own_street_contribution=(
                observation.street_contributions[observation.hero_index]
            ),
            candidate_actions=candidates,
        )
        threshold = rng.random()
        cumulative = 0.0
        for candidate, probability in probabilities.items():
            cumulative += probability
            if threshold < cumulative:
                return candidate
        if not math.isclose(cumulative, 1.0, abs_tol=1e-12):  # pragma: no cover
            raise RuntimeError("opponent action probabilities do not sum to one")
        return candidates[-1]


__all__ = ["HeuristicBaseline"]
