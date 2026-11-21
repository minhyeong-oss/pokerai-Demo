"""Small-game CFR reference used to validate the learning pipeline.

Full Hold'em MCCFR is intentionally outside the MVP.  This module implements
tabular vanilla CFR for Kuhn poker so regret matching, average-strategy
collection, deterministic seeding, and exploitability-oriented experiments can
be tested before scaling the same ideas to a much larger game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from types import MappingProxyType
from typing import Dict, Mapping, Tuple


_ACTIONS = ("pass", "bet")
_DEALS = tuple(permutations((0, 1, 2), 2))


@dataclass
class _InfoSet:
    regret_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])
    strategy_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def strategy(self, reach: float) -> tuple[float, float]:
        positives = [max(value, 0.0) for value in self.regret_sum]
        normalizer = sum(positives)
        if normalizer > 0.0:
            strategy = (positives[0] / normalizer, positives[1] / normalizer)
        else:
            strategy = (0.5, 0.5)
        self.strategy_sum[0] += reach * strategy[0]
        self.strategy_sum[1] += reach * strategy[1]
        return strategy

    def average(self) -> tuple[float, float]:
        normalizer = sum(self.strategy_sum)
        if normalizer <= 0.0:
            return 0.5, 0.5
        return self.strategy_sum[0] / normalizer, self.strategy_sum[1] / normalizer


@dataclass(frozen=True)
class KuhnTrainingResult:
    iterations: int
    expected_value_player0: float
    strategy: Mapping[str, Mapping[str, float]]


class KuhnCFRTrainer:
    """Deterministic full-tree CFR trainer for three-card Kuhn poker."""

    def __init__(self) -> None:
        self._nodes: Dict[str, _InfoSet] = {}

    @staticmethod
    def _terminal_payoff(cards: Tuple[int, int], history: str) -> float | None:
        player0_wins = cards[0] > cards[1]
        if history == "pp":
            return 1.0 if player0_wins else -1.0
        if history == "bp":
            return 1.0
        if history == "pbp":
            return -1.0
        if history in ("bb", "pbb"):
            return 2.0 if player0_wins else -2.0
        return None

    @staticmethod
    def _actor(history: str) -> int:
        if history in ("", "pb"):
            return 0
        return 1

    @staticmethod
    def _next_history(history: str, action_index: int) -> str:
        return history + ("p" if action_index == 0 else "b")

    def _cfr(
        self,
        cards: Tuple[int, int],
        history: str,
        reach0: float,
        reach1: float,
    ) -> float:
        terminal = self._terminal_payoff(cards, history)
        if terminal is not None:
            return terminal

        actor = self._actor(history)
        key = f"{cards[actor]}|{history}"
        node = self._nodes.setdefault(key, _InfoSet())
        own_reach = reach0 if actor == 0 else reach1
        strategy = node.strategy(own_reach)
        utilities = [0.0, 0.0]

        for action_index, probability in enumerate(strategy):
            next_history = self._next_history(history, action_index)
            if actor == 0:
                utilities[action_index] = self._cfr(
                    cards,
                    next_history,
                    reach0 * probability,
                    reach1,
                )
            else:
                utilities[action_index] = self._cfr(
                    cards,
                    next_history,
                    reach0,
                    reach1 * probability,
                )

        node_utility = sum(
            probability * utility
            for probability, utility in zip(strategy, utilities)
        )
        if actor == 0:
            counterfactual_reach = reach1
            for index, utility in enumerate(utilities):
                node.regret_sum[index] += counterfactual_reach * (utility - node_utility)
        else:
            counterfactual_reach = reach0
            for index, utility in enumerate(utilities):
                # utilities are from player 0's perspective, so player 1's
                # regret has the opposite sign.
                node.regret_sum[index] += counterfactual_reach * (node_utility - utility)
        return node_utility

    def train(self, iterations: int = 10_000) -> KuhnTrainingResult:
        if isinstance(iterations, bool) or not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        # Each call is a complete, self-contained experiment.  This keeps the
        # returned iteration count, EV average, and average strategy aligned.
        self._nodes = {}
        value_sum = 0.0
        for _ in range(iterations):
            iteration_value = 0.0
            for deal in _DEALS:
                iteration_value += self._cfr(deal, "", 1.0, 1.0)
            value_sum += iteration_value / len(_DEALS)

        readable_strategy = MappingProxyType({
            key: MappingProxyType({
                action: probability
                for action, probability in zip(_ACTIONS, node.average())
            })
            for key, node in sorted(self._nodes.items())
        })
        return KuhnTrainingResult(
            iterations=iterations,
            expected_value_player0=value_sum / iterations,
            strategy=readable_strategy,
        )


def train_kuhn_cfr(iterations: int = 10_000) -> KuhnTrainingResult:
    """Train and return a fresh Kuhn-poker average strategy."""

    return KuhnCFRTrainer().train(iterations)


__all__ = ["KuhnCFRTrainer", "KuhnTrainingResult", "train_kuhn_cfr"]
