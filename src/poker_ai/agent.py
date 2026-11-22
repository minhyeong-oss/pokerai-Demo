"""Stateful research agent joining range updates, equity, rollout, and logging."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from .actions import CandidateAction
from .cards import parse_cards
from .equity import EquityResult, monte_carlo_equity
from .logging import JsonlAuditWriter, make_audit_record
from .observation import ActionEvent, BoardEvent, Observation
from .opponents import OpponentActionModel, make_range_profile
from .policy import Decision, choose_action
from .ranges import HandRange, UniformRange, bayes_update
from .range_snapshots import RangeSnapshotStore, snapshot_content_id
from .rollout import BaselineRolloutEstimator
from .seeding import action_candidate_id, derive_analysis_seed, validate_master_seed
from .versions import POLICY_MODEL_VERSION


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
    policy_hand_seed: int = 0
    risk_aversion: float = 0.0
    audit_path: str | Path | None = None
    snapshot_root: str | Path | None = None
    policy_id: str = "hero"
    model_version: str = POLICY_MODEL_VERSION
    opponent_range: HandRange = field(init=False, repr=False)
    _range_revision: int = field(init=False, default=0, repr=False)
    _writer: JsonlAuditWriter | None = field(init=False, default=None, repr=False)
    _snapshot_store: RangeSnapshotStore | None = field(
        init=False, default=None, repr=False
    )
    _processed_event_ids: dict[str, str] = field(
        init=False, default_factory=dict, repr=False
    )
    _range_snapshot_id: str = field(init=False, default="", repr=False)
    _opponent_model: OpponentActionModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.equity_simulations, self.rollout_simulations)
        ):
            raise TypeError("simulation counts must be integers")
        if self.equity_simulations <= 0 or self.rollout_simulations <= 1:
            raise ValueError("simulation counts must be positive (rollouts greater than one)")
        validate_master_seed(self.policy_hand_seed, "policy_hand_seed")
        if (
            isinstance(self.risk_aversion, bool)
            or not isinstance(self.risk_aversion, (int, float))
            or not math.isfinite(float(self.risk_aversion))
            or not 0.0 <= float(self.risk_aversion) <= 1.0
        ):
            raise ValueError("risk_aversion must be between zero and one")
        self._opponent_model = OpponentActionModel(self.opponent_style)
        self.opponent_style = self._opponent_model.name
        self.opponent_range = UniformRange()
        self._range_snapshot_id = snapshot_content_id(self.opponent_range)
        if self.audit_path is not None:
            self._writer = JsonlAuditWriter(self.audit_path)
            if self.snapshot_root is None:
                self.snapshot_root = Path(str(self.audit_path) + ".snapshots")
        if self.snapshot_root is not None:
            self._snapshot_store = RangeSnapshotStore(self.snapshot_root, self.policy_id)

    def reset_hand(self) -> None:
        self.opponent_range = UniformRange()
        self._range_revision = 0
        self._processed_event_ids.clear()
        self._range_snapshot_id = snapshot_content_id(self.opponent_range)

    @property
    def range_version(self) -> str:
        return (
            f"{self.opponent_style}:r{self._range_revision}:"
            f"{self._range_snapshot_id[:16]}"
        )

    def observe_opponent_action(
        self,
        action: CandidateAction,
        *,
        event_id: str,
        hole_cards: tuple[str, str],
        board: tuple[str, ...],
        pot: int,
        call_cost: int,
        own_street_contribution: int,
        candidate_actions: Sequence[CandidateAction],
    ) -> bool:
        """Apply one fully described public opponent action idempotently.

        Normal play calls :meth:`decide`, which consumes ``public_history``
        automatically.  This explicit method remains for import/replay tools.
        """

        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id is required")
        candidates = tuple(candidate_actions)
        event_payload = {
            "event_id": event_id,
            "kind": action.kind,
            "raise_to": action.raise_to,
            "hole_cards": hole_cards,
            "board": board,
            "pot": pot,
            "call_cost": call_cost,
            "own_street_contribution": own_street_contribution,
            "candidate_actions": tuple(
                action_candidate_id(candidate) for candidate in candidates
            ),
        }
        event_fingerprint = hashlib.sha256(repr(event_payload).encode("utf-8")).hexdigest()
        previous = self._processed_event_ids.get(event_id)
        if previous == event_fingerprint:
            return False
        if previous is not None:
            raise ValueError("event_id was reused for different action data")
        profile = make_range_profile(
            self.opponent_style,
            board_cards=board,
            pot=pot,
            call_cost=call_cost,
            own_street_contribution=own_street_contribution,
            candidate_actions=candidates,
        )
        self.opponent_range = bayes_update(
            self.opponent_range,
            action_candidate_id(action),
            profile,
            dead_cards=(*hole_cards, *board),
        )
        self._range_revision += 1
        self._range_snapshot_id = snapshot_content_id(self.opponent_range)
        self._processed_event_ids[event_id] = event_fingerprint
        return True

    def _condition_visible_cards(
        self,
        hole_cards: tuple[str, str],
        board: tuple[str, ...],
    ) -> None:
        dead = frozenset(parse_cards((*hole_cards, *board), field_name="visible_cards"))
        if any(card in dead for combo in self.opponent_range.masses for card in combo):
            self.opponent_range = self.opponent_range.conditioned(tuple(dead))
            self._range_revision += 1
            self._range_snapshot_id = snapshot_content_id(self.opponent_range)

    def _process_public_history(self, observation: Observation) -> None:
        for event in observation.public_history:
            fingerprint = hashlib.sha256(repr(event).encode("utf-8")).hexdigest()
            previous = self._processed_event_ids.get(event.event_id)
            if previous == fingerprint:
                continue
            if previous is not None:
                raise ValueError("event_id was reused for different public event data")

            if isinstance(event, BoardEvent):
                self._condition_visible_cards(observation.hole_cards, event.board)
            elif isinstance(event, ActionEvent) and event.actor_index != observation.hero_index:
                action = CandidateAction(event.action, event.raise_to)
                pre_state = event.pre_state
                profile = make_range_profile(
                    self.opponent_style,
                    board_cards=pre_state.board,
                    pot=pre_state.pot,
                    call_cost=pre_state.call_cost,
                    own_street_contribution=(
                        pre_state.street_contributions[event.actor_index]
                    ),
                    candidate_actions=pre_state.candidate_actions,
                )
                self.opponent_range = bayes_update(
                    self.opponent_range,
                    action_candidate_id(action),
                    profile,
                    dead_cards=(*observation.hole_cards, *pre_state.board),
                )
                self._range_revision += 1
                self._range_snapshot_id = snapshot_content_id(self.opponent_range)
            self._processed_event_ids[event.event_id] = fingerprint

        self._condition_visible_cards(observation.hole_cards, observation.board)

    def decide(self, observation: Observation) -> AgentDecision:
        if observation.actor_index != observation.hero_index:
            raise ValueError("the agent may decide only when hero_index is the current actor")
        self._process_public_history(observation)
        active_opponents = [
            index
            for index, status in enumerate(observation.statuses)
            if index != observation.hero_index and status != "folded"
        ]
        if len(observation.stacks) != 2 or len(active_opponents) != 1:
            raise ValueError("ResearchPokerAgent currently supports heads-up observations only")

        self._range_snapshot_id = snapshot_content_id(self.opponent_range)
        if self._snapshot_store is not None:
            stored_id = self._snapshot_store.write(self.opponent_range)
            if stored_id != self._range_snapshot_id:  # pragma: no cover - integrity guard
                raise RuntimeError("stored range snapshot ID changed unexpectedly")

        started = perf_counter()
        try:
            decision_index = int(observation.decision_id.rsplit(":d", 1)[1])
        except (IndexError, ValueError) as exc:  # Observation normally guarantees this.
            raise ValueError("observation has an invalid decision_id") from exc
        equity_seed = derive_analysis_seed(
            self.policy_hand_seed,
            purpose="equity",
            decision_index=decision_index,
            candidate_id="none",
            sample_index=0,
        )
        equity = monte_carlo_equity(
            observation.hole_cards,
            observation.board,
            opponents=1,
            simulations=self.equity_simulations,
            seed=equity_seed,
            opponent_range=self.opponent_range,
        )
        rollout = BaselineRolloutEstimator(
            opponent_style=self.opponent_style,
            simulations=self.rollout_simulations,
            policy_hand_seed=self.policy_hand_seed,
            decision_index=decision_index,
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
                "equity_seed": equity_seed,
                "policy_hand_seed": self.policy_hand_seed,
                "opponent_style": self.opponent_style,
                "opponent_model_parameters": self._opponent_model.parameters.as_dict(),
                "decision_index": decision_index,
                "equity_simulations": self.equity_simulations,
                "rollout_simulations": self.rollout_simulations,
                "risk_aversion": self.risk_aversion,
                "range_snapshot_id": self._range_snapshot_id,
                "processed_event_ids": sorted(self._processed_event_ids),
            },
        )
        if self._writer is not None:
            self._writer.write(audit_record)
        return AgentDecision(decision=decision, equity=equity, audit_record=audit_record)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "ResearchPokerAgent":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["AgentDecision", "ResearchPokerAgent"]
