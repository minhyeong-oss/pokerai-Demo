"""Stateful research agent joining range updates, equity, rollout, and logging."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from .actions import CandidateAction
from .equity import EquityResult, monte_carlo_equity
from .logging import JsonlAuditWriter, make_audit_record
from .observation import Observation
from .opponents import get_style, make_range_profile
from .policy import Decision, choose_action
from .ranges import HandRange, UniformRange, bayes_update
from .rollout import BaselineRolloutEstimator


@dataclass(frozen=True)
class AgentDecision:
    decision: Decision
    equity: EquityResult
    audit_record: dict[str, Any]

    @property
    def action(self) -> CandidateAction:
        return self.decision.action


@dataclass
class ResearchPokerAgent:
    """Baseline heads-up AI that only consumes public ``Observation`` values."""

    opponent_style: str = "balanced"
    equity_simulations: int = 10_000
    rollout_simulations: int = 2_000
    seed: int = 0
    risk_aversion: float = 0.0
    audit_path: str | Path | None = None
    model_version: str = "baseline-ev-v1"
    opponent_range: HandRange = field(init=False, repr=False)
    _decision_index: int = field(init=False, default=0, repr=False)
    _range_revision: int = field(init=False, default=0, repr=False)
    _writer: JsonlAuditWriter | None = field(init=False, default=None, repr=False)
    _processed_event_ids: dict[str, str] = field(
        init=False, default_factory=dict, repr=False
    )
    _range_digest: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.equity_simulations, self.rollout_simulations)
        ):
            raise TypeError("simulation counts must be integers")
        if self.equity_simulations <= 0 or self.rollout_simulations <= 1:
            raise ValueError("simulation counts must be positive (rollouts greater than one)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.risk_aversion, bool)
            or not isinstance(self.risk_aversion, (int, float))
            or not math.isfinite(float(self.risk_aversion))
            or not 0.0 <= float(self.risk_aversion) <= 1.0
        ):
            raise ValueError("risk_aversion must be between zero and one")
        self.opponent_style = get_style(self.opponent_style).name
        self.opponent_range = UniformRange()
        self._range_digest = hashlib.sha256(
            f"range-v1:{self.opponent_style}".encode("utf-8")
        ).hexdigest()[:16]
        if self.audit_path is not None:
            self._writer = JsonlAuditWriter(self.audit_path)

    def reset_hand(self) -> None:
        self.opponent_range = UniformRange()
        self._range_revision = 0
        self._processed_event_ids.clear()
        self._range_digest = hashlib.sha256(
            f"range-v1:{self.opponent_style}".encode("utf-8")
        ).hexdigest()[:16]

    @property
    def range_version(self) -> str:
        return f"{self.opponent_style}:r{self._range_revision}:{self._range_digest}"

    def observe_opponent_action(
        self,
        action: CandidateAction,
        *,
        hero_cards: tuple[str, str],
        board: tuple[str, ...],
        facing_bet: bool,
        event_id: str | int | None = None,
    ) -> bool:
        """Bayesian-update from one public event, idempotently when identified.

        Returns ``True`` when the event was applied and ``False`` when an
        already-processed ``event_id`` was safely ignored.
        """

        normalized_event_id: str | None = None
        event_payload = {
            "event_id": None if event_id is None else str(event_id),
            "kind": action.kind,
            "raise_to": action.raise_to,
            "hero_cards": hero_cards,
            "board": board,
            "facing_bet": facing_bet,
        }
        event_fingerprint = hashlib.sha256(
            json.dumps(event_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if event_id is not None:
            normalized_event_id = str(event_id)
            if not normalized_event_id:
                raise ValueError("event_id cannot be empty")
            previous = self._processed_event_ids.get(normalized_event_id)
            if previous == event_fingerprint:
                return False
            if previous is not None:
                raise ValueError("event_id was reused for different action data")

        action_name = "raise" if action.kind == "raise" else action.kind
        profile = make_range_profile(
            self.opponent_style,
            board_cards=board,
            facing_bet=facing_bet,
        )
        self.opponent_range = bayes_update(
            self.opponent_range,
            action_name,
            profile,
            dead_cards=(*hero_cards, *board),
        )
        self._range_revision += 1
        self._range_digest = hashlib.sha256(
            (
                self._range_digest
                + json.dumps(event_payload, sort_keys=True, separators=(",", ":"))
            ).encode("utf-8")
        ).hexdigest()[:16]
        if normalized_event_id is not None:
            self._processed_event_ids[normalized_event_id] = event_fingerprint
        return True

    def decide(self, observation: Observation) -> AgentDecision:
        if observation.actor_index != observation.hero_index:
            raise ValueError("the agent may decide only when hero_index is the current actor")
        active_opponents = [
            index
            for index, status in enumerate(observation.player_statuses)
            if index != observation.hero_index and status not in ("folded", "out")
        ]
        if len(observation.stacks) != 2 or len(active_opponents) != 1:
            raise ValueError("ResearchPokerAgent currently supports heads-up observations only")

        started = perf_counter()
        decision_seed = self.seed + self._decision_index * 2
        equity = monte_carlo_equity(
            observation.hero_cards,
            observation.board,
            opponents=1,
            simulations=self.equity_simulations,
            seed=decision_seed,
            opponent_range=self.opponent_range,
        )
        rollout = BaselineRolloutEstimator(
            opponent_style=self.opponent_style,
            simulations=self.rollout_simulations,
            seed=decision_seed + 1,
            opponent_range=self.opponent_range,
        )
        decision = choose_action(
            observation,
            equity,
            rollout_estimates=rollout,
            risk_aversion=self.risk_aversion,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        audit_record = make_audit_record(
            observation,
            decision,
            model_version=self.model_version,
            range_version=self.range_version,
            calculation_ms=elapsed_ms,
            metadata={
                "equity": equity.as_dict(),
                "equity_seed": decision_seed,
                "rollout_seed": decision_seed + 1,
                "opponent_style": self.opponent_style,
                "decision_index": self._decision_index,
                "equity_simulations": self.equity_simulations,
                "rollout_simulations": self.rollout_simulations,
                "risk_aversion": self.risk_aversion,
                "range_digest": self._range_digest,
            },
        )
        if self._writer is not None:
            self._writer.write(audit_record)
        self._decision_index += 1
        return AgentDecision(decision=decision, equity=equity, audit_record=audit_record)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "ResearchPokerAgent":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["AgentDecision", "ResearchPokerAgent"]
