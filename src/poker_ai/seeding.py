"""Domain-separated reproducible seed derivation."""

from __future__ import annotations

import hashlib
import hmac
from typing import Iterable

from .actions import CandidateAction


_MASTER_PREFIX = "pokerai-master-v1:"
_PURPOSES = frozenset(("equity", "rollout_response", "rollout_runout", "actual_action"))


def validate_master_seed(value: object, name: str = "master_seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _encode_domain(fields: Iterable[object]) -> bytes:
    encoded = bytearray()
    for field in fields:
        text = str(field).encode("utf-8")
        encoded.extend(len(text).to_bytes(4, "big"))
        encoded.extend(text)
    return bytes(encoded)


def derive_seed(master_seed: int, *domain_fields: object) -> int:
    """Derive one unsigned 64-bit seed with the PRD HMAC-SHA-256 rule."""

    master_seed = validate_master_seed(master_seed)
    if not domain_fields:
        raise ValueError("at least one seed domain field is required")
    key = hashlib.sha256(f"{_MASTER_PREFIX}{master_seed}".encode("utf-8")).digest()
    digest = hmac.new(key, _encode_domain(domain_fields), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def action_candidate_id(action: CandidateAction | None) -> str:
    if action is None:
        return "none"
    if not isinstance(action, CandidateAction):
        raise TypeError("action must be CandidateAction or None")
    if action.kind == "raise":
        return f"raise:{action.raise_to}"
    return action.kind


def derive_analysis_seed(
    policy_hand_seed: int,
    *,
    purpose: str,
    decision_index: int,
    candidate_id: str,
    sample_index: int,
) -> int:
    """Derive an isolated policy-analysis stream coordinate."""

    validate_master_seed(policy_hand_seed, "policy_hand_seed")
    if purpose not in _PURPOSES:
        raise ValueError(f"unsupported seed purpose: {purpose!r}")
    for name, value in (("decision_index", decision_index), ("sample_index", sample_index)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    return derive_seed(
        policy_hand_seed,
        purpose,
        decision_index,
        candidate_id,
        sample_index,
    )


__all__ = [
    "action_candidate_id",
    "derive_analysis_seed",
    "derive_seed",
    "validate_master_seed",
]
