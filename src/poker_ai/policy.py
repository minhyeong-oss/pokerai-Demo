"""Expected-value policy for the public poker observation boundary.

This module intentionally does not inspect a game engine.  It combines a
public :class:`~poker_ai.observation.Observation`, a range-weighted equity
result, and optional rollout estimates.  When a rollout is unavailable it uses
small, documented showdown proxies so every candidate still has an auditable
EV and confidence interval.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, Union

from .actions import CandidateAction, generate_candidate_actions
from .observation import Observation


Number = Union[int, float]


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError("{} must be numeric, not bool".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("{} must be numeric".format(name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _probability(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError("{} must be between 0 and 1".format(name))
    return result


def _optional_value(source: object, names: Sequence[str]) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return None
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return None


@dataclass(frozen=True)
class RolloutEstimate:
    """A sampled action value and the information needed for a CI.

    Callers may provide either an explicit ``ci_low``/``ci_high`` pair or a
    standard error.  A scalar rollout is also accepted by :func:`choose_action`
    and is treated as a deterministic estimate.
    """

    expected_value: float
    standard_error: Optional[float] = None
    samples: Optional[int] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def __post_init__(self) -> None:
        expected_value = _finite(self.expected_value, "expected_value")
        object.__setattr__(self, "expected_value", expected_value)

        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("ci_low and ci_high must either both be set or both be None")
        if self.ci_low is not None:
            low = _finite(self.ci_low, "ci_low")
            high = _finite(self.ci_high, "ci_high")
            if low > high:
                raise ValueError("ci_low cannot exceed ci_high")
            object.__setattr__(self, "ci_low", low)
            object.__setattr__(self, "ci_high", high)

        if self.standard_error is not None:
            standard_error = _finite(self.standard_error, "standard_error")
            if standard_error < 0.0:
                raise ValueError("standard_error cannot be negative")
            object.__setattr__(self, "standard_error", standard_error)

        if self.samples is not None:
            if isinstance(self.samples, bool) or not isinstance(self.samples, int):
                raise TypeError("samples must be an integer")
            if self.samples <= 0:
                raise ValueError("samples must be positive")

    def confidence_interval(self, z_score: float) -> Tuple[float, float]:
        if self.ci_low is not None:
            return float(self.ci_low), float(self.ci_high)
        if self.standard_error is None:
            return self.expected_value, self.expected_value
        margin = z_score * self.standard_error
        return self.expected_value - margin, self.expected_value + margin


@dataclass(frozen=True)
class SimpleResponseModel:
    """Heads-up response proxy for raises that have no sampled rollout.

    ``fold_probability`` is the chance the opponent folds immediately.  Every
    non-fold response is approximated as a call to showdown.  An optional
    ``equity_when_called`` captures range strengthening after the call; when it
    is omitted, the supplied unconditional equity is used.
    """

    fold_probability: float = 0.0
    equity_when_called: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fold_probability",
            _probability(self.fold_probability, "fold_probability"),
        )
        if self.equity_when_called is not None:
            object.__setattr__(
                self,
                "equity_when_called",
                _probability(self.equity_when_called, "equity_when_called"),
            )


@dataclass(frozen=True)
class ActionEvaluation:
    """One action's net-chip EV measured from the decision point."""

    action: CandidateAction
    expected_value: float
    ci_low: float
    ci_high: float
    selection_score: float
    source: str
    samples: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, CandidateAction):
            raise TypeError("action must be a CandidateAction")
        expected = _finite(self.expected_value, "expected_value")
        low = _finite(self.ci_low, "ci_low")
        high = _finite(self.ci_high, "ci_high")
        score = _finite(self.selection_score, "selection_score")
        if low > high:
            raise ValueError("ci_low cannot exceed ci_high")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        if self.samples is not None:
            if isinstance(self.samples, bool) or not isinstance(self.samples, int):
                raise TypeError("samples must be an integer")
            if self.samples <= 0:
                raise ValueError("samples must be positive")
        object.__setattr__(self, "expected_value", expected)
        object.__setattr__(self, "ci_low", low)
        object.__setattr__(self, "ci_high", high)
        object.__setattr__(self, "selection_score", score)


@dataclass(frozen=True)
class Decision:
    """Chosen action together with the complete comparison that produced it."""

    action: CandidateAction
    evaluations: Tuple[ActionEvaluation, ...]
    equity: float
    equity_ci_low: float
    equity_ci_high: float
    confidence_level: float
    selection_rule: str

    def __post_init__(self) -> None:
        evaluations = tuple(self.evaluations)
        if not evaluations:
            raise ValueError("a decision requires at least one action evaluation")
        if self.action not in tuple(item.action for item in evaluations):
            raise ValueError("chosen action must appear in evaluations")
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "equity", _probability(self.equity, "equity"))
        low = _probability(self.equity_ci_low, "equity_ci_low")
        high = _probability(self.equity_ci_high, "equity_ci_high")
        if low > high:
            raise ValueError("equity_ci_low cannot exceed equity_ci_high")
        object.__setattr__(self, "equity_ci_low", low)
        object.__setattr__(self, "equity_ci_high", high)
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level"),
        )
        if not self.selection_rule:
            raise ValueError("selection_rule cannot be empty")

    @property
    def chosen_evaluation(self) -> ActionEvaluation:
        return next(item for item in self.evaluations if item.action == self.action)


class ActionEstimator(Protocol):
    """Optional richer response/rollout model accepted by the policy."""

    def estimate(
        self,
        observation: Observation,
        action: CandidateAction,
        equity_result: object,
    ) -> Optional[object]:
        ...


RolloutSource = Union[
    Mapping[object, object],
    Callable[[Observation, CandidateAction, object], Optional[object]],
]


def _equity_view(
    equity_result: object,
    z_score: float,
) -> Tuple[float, float, float, Optional[int]]:
    if isinstance(equity_result, (int, float)) and not isinstance(equity_result, bool):
        equity = _probability(equity_result, "equity")
        return equity, equity, equity, None

    raw_equity = _optional_value(equity_result, ("equity",))
    if raw_equity is None:
        raise TypeError("equity_result must be numeric or expose an equity value")
    equity = _probability(raw_equity, "equity")

    raw_low = _optional_value(equity_result, ("ci95_low", "ci_low", "equity_ci_low"))
    raw_high = _optional_value(equity_result, ("ci95_high", "ci_high", "equity_ci_high"))
    if raw_low is None and raw_high is None:
        interval = _optional_value(
            equity_result,
            ("confidence_interval_95", "confidence_interval"),
        )
        if interval is not None and not callable(interval):
            try:
                raw_low, raw_high = tuple(interval)
            except (TypeError, ValueError) as exc:
                raise ValueError("equity confidence interval must contain two values") from exc
    if (raw_low is None) != (raw_high is None):
        raise ValueError("equity confidence interval requires both lower and upper bounds")

    raw_samples = _optional_value(
        equity_result, ("simulations", "samples", "trials", "n")
    )
    samples = None
    if raw_samples is not None:
        if isinstance(raw_samples, bool) or not isinstance(raw_samples, int):
            raise TypeError("equity sample count must be an integer")
        if raw_samples <= 0:
            raise ValueError("equity sample count must be positive")
        samples = raw_samples

    if raw_low is not None:
        low = _probability(raw_low, "equity ci lower bound")
        high = _probability(raw_high, "equity ci upper bound")
    else:
        raw_standard_error = _optional_value(
            equity_result, ("standard_error", "stderr", "std_error")
        )
        if raw_standard_error is not None:
            standard_error = _finite(raw_standard_error, "equity standard error")
            if standard_error < 0.0:
                raise ValueError("equity standard error cannot be negative")
        elif samples is not None:
            standard_error = math.sqrt(equity * (1.0 - equity) / samples)
        else:
            standard_error = 0.0
        low = max(0.0, equity - z_score * standard_error)
        high = min(1.0, equity + z_score * standard_error)

    if low > equity or high < equity:
        raise ValueError("equity confidence interval must contain equity")
    return equity, low, high, samples


def _coerce_rollout_estimate(value: object) -> RolloutEstimate:
    if isinstance(value, RolloutEstimate):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return RolloutEstimate(float(value))

    expected = _optional_value(value, ("expected_value", "ev", "mean", "value"))
    if expected is None:
        raise TypeError("rollout estimate must expose expected_value, ev, mean, or value")
    low = _optional_value(value, ("ci95_low", "ci_low"))
    high = _optional_value(value, ("ci95_high", "ci_high"))
    standard_error = _optional_value(value, ("standard_error", "stderr", "std_error"))
    samples = _optional_value(value, ("simulations", "samples", "trials", "n"))
    return RolloutEstimate(
        expected_value=expected,
        standard_error=standard_error,
        samples=samples,
        ci_low=low,
        ci_high=high,
    )


def _mapping_estimate(
    estimates: Mapping[object, object], action: CandidateAction
) -> Optional[object]:
    keys = [action, (action.kind, action.raise_to)]
    if action.raise_to is None:
        keys.append(action.kind)
    else:
        keys.extend(
            (
                "raise:{:g}".format(action.raise_to),
                "raise@{:g}".format(action.raise_to),
            )
        )
    for key in keys:
        if key in estimates:
            return estimates[key]
    return None


def _external_estimate(
    source: Optional[object],
    observation: Observation,
    action: CandidateAction,
    equity_result: object,
) -> Optional[RolloutEstimate]:
    if source is None:
        return None
    raw = None
    if isinstance(source, Mapping):
        raw = _mapping_estimate(source, action)
    elif callable(source):
        raw = source(observation, action, equity_result)
    elif hasattr(source, "estimate"):
        raw = source.estimate(observation, action, equity_result)
    else:
        raise TypeError("estimate source must be a mapping, callable, or expose estimate()")
    return None if raw is None else _coerce_rollout_estimate(raw)


def _rollout_evaluation(
    action: CandidateAction,
    estimate: RolloutEstimate,
    z_score: float,
    risk_aversion: float,
    source: str,
) -> ActionEvaluation:
    low, high = estimate.confidence_interval(z_score)
    score = estimate.expected_value - risk_aversion * (estimate.expected_value - low)
    return ActionEvaluation(
        action=action,
        expected_value=estimate.expected_value,
        ci_low=low,
        ci_high=high,
        selection_score=score,
        source=source,
        samples=estimate.samples,
    )


def _simple_model_values(
    response_model: Optional[object],
) -> Tuple[float, Optional[float]]:
    if response_model is None:
        return 0.0, None
    if isinstance(response_model, SimpleResponseModel):
        return response_model.fold_probability, response_model.equity_when_called
    if isinstance(response_model, Mapping):
        raw_fold = response_model.get("fold_probability", 0.0)
        raw_called = response_model.get("equity_when_called")
        fold_probability = _probability(raw_fold, "fold_probability")
        called_equity = (
            None
            if raw_called is None
            else _probability(raw_called, "equity_when_called")
        )
        return fold_probability, called_equity
    return 0.0, None


def _showdown_proxy(
    observation: Observation,
    action: CandidateAction,
    equity: float,
    equity_low: float,
    equity_high: float,
    response_model: Optional[object],
    risk_aversion: float,
    equity_samples: Optional[int],
) -> ActionEvaluation:
    pot = observation.pot
    source = "showdown_proxy"

    if action.kind == "fold":
        expected = low = high = 0.0
        source = "deterministic"
    elif action.kind == "check":
        expected = equity * pot
        low = equity_low * pot
        high = equity_high * pot
    elif action.kind == "call":
        final_pot = pot + observation.call_cost
        expected = equity * final_pot - observation.call_cost
        low = equity_low * final_pot - observation.call_cost
        high = equity_high * final_pot - observation.call_cost
    else:
        assert action.kind == "raise" and action.raise_to is not None
        hero_street = observation.street_contributions[observation.hero_index]
        full_hero_increment = min(
            observation.stacks[observation.hero_index],
            max(0.0, action.raise_to - hero_street),
        )

        opponents = [
            index
            for index, status in enumerate(observation.statuses)
            if index != observation.hero_index and status != "folded"
        ]
        if len(opponents) != 1:
            raise ValueError(
                "the analytic raise proxy is heads-up only; provide a rollout for multiway play"
            )
        opponent_index = opponents[0]
        opponent_street = observation.street_contributions[opponent_index]
        opponent_increment = max(0.0, action.raise_to - opponent_street)
        opponent_increment = min(
            opponent_increment, observation.stacks[opponent_index]
        )
        opponent_final = opponent_street + opponent_increment
        hero_increment = min(
            full_hero_increment,
            max(0.0, min(action.raise_to, opponent_final) - hero_street),
        )
        called_pot = pot + hero_increment + opponent_increment

        fold_probability, called_equity = _simple_model_values(response_model)
        if called_equity is None:
            called_equity = equity
            called_low = equity_low
            called_high = equity_high
        else:
            lower_margin = equity - equity_low
            upper_margin = equity_high - equity
            called_low = max(0.0, called_equity - lower_margin)
            called_high = min(1.0, called_equity + upper_margin)

        def combined(value: float) -> float:
            called_ev = value * called_pot - hero_increment
            return fold_probability * pot + (1.0 - fold_probability) * called_ev

        expected = combined(called_equity)
        low = combined(called_low)
        high = combined(called_high)
        source = "simple_response_model"

    score = expected - risk_aversion * (expected - low)
    return ActionEvaluation(
        action=action,
        expected_value=expected,
        ci_low=low,
        ci_high=high,
        selection_score=score,
        source=source,
        samples=equity_samples,
    )


def _tie_break_key(evaluation: ActionEvaluation, observation: Observation) -> tuple:
    # Higher priority wins exact numerical ties.  Smaller raises take less model
    # risk when their evaluated score is identical.
    priority = {"check": 4, "fold": 3, "call": 2, "raise": 1}
    incremental_cost = 0.0
    if evaluation.action.kind == "call":
        incremental_cost = observation.call_cost
    elif evaluation.action.kind == "raise":
        incremental_cost = max(
            0.0,
            float(evaluation.action.raise_to)
            - observation.street_contributions[observation.hero_index],
        )
    return (
        evaluation.selection_score,
        evaluation.expected_value,
        evaluation.ci_low,
        priority[evaluation.action.kind],
        -incremental_cost,
    )


def choose_action(
    observation: Observation,
    equity_result: object,
    response_model: Optional[object] = None,
    rollout_estimates: Optional[RolloutSource] = None,
    *,
    candidates: Optional[Sequence[CandidateAction]] = None,
    confidence_level: float = 0.95,
    risk_aversion: float = 0.0,
) -> Decision:
    """Evaluate every legal candidate and choose the highest scored action.

    Rollout estimates have first priority.  A richer ``response_model`` may
    expose ``estimate(observation, action, equity_result)`` for selected
    actions.  Remaining candidates use transparent showdown proxies.  By
    default selection maximizes mean EV; setting ``risk_aversion=1`` compares
    CI lower bounds, and values in between blend the two.
    """

    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")
    confidence_level = _probability(confidence_level, "confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    risk_aversion = _probability(risk_aversion, "risk_aversion")
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    equity, equity_low, equity_high, equity_samples = _equity_view(
        equity_result, z_score
    )

    action_candidates = (
        generate_candidate_actions(observation)
        if candidates is None
        else tuple(candidates)
    )
    if not action_candidates:
        raise ValueError("candidates cannot be empty")
    if len(set(action_candidates)) != len(action_candidates):
        raise ValueError("candidates cannot contain duplicates")
    if not all(isinstance(item, CandidateAction) for item in action_candidates):
        raise TypeError("every candidate must be a CandidateAction")

    evaluations = []
    for action in action_candidates:
        estimate = _external_estimate(
            rollout_estimates, observation, action, equity_result
        )
        source = "rollout"
        if estimate is None and response_model is not None and not isinstance(
            response_model, (SimpleResponseModel, Mapping)
        ):
            estimate = _external_estimate(
                response_model, observation, action, equity_result
            )
            source = "response_model"
        if estimate is not None:
            evaluation = _rollout_evaluation(
                action, estimate, z_score, risk_aversion, source
            )
        else:
            evaluation = _showdown_proxy(
                observation,
                action,
                equity,
                equity_low,
                equity_high,
                response_model,
                risk_aversion,
                equity_samples,
            )
        evaluations.append(evaluation)

    chosen = max(evaluations, key=lambda item: _tie_break_key(item, observation))
    selection_rule = (
        "max_expected_value"
        if math.isclose(risk_aversion, 0.0)
        else "confidence_adjusted_ev(risk_aversion={:g})".format(risk_aversion)
    )
    return Decision(
        action=chosen.action,
        evaluations=tuple(evaluations),
        equity=equity,
        equity_ci_low=equity_low,
        equity_ci_high=equity_high,
        confidence_level=confidence_level,
        selection_rule=selection_rule,
    )


decide = choose_action


__all__ = [
    "ActionEstimator",
    "ActionEvaluation",
    "Decision",
    "RolloutEstimate",
    "SimpleResponseModel",
    "choose_action",
    "decide",
]
