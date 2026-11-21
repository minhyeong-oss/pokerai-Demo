"""Boundary for future visual inputs.

The MVP intentionally does not capture screens or click poker clients.  A
future adapter can implement this protocol for an explicitly permitted
practice application while the strategy continues to consume the same public
``Observation`` type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .observation import Observation


class InconsistentSnapshotError(ValueError):
    """Raised when recognized fields cannot describe one legal game state."""


@dataclass(frozen=True)
class SnapshotIdentity:
    """Identity used to reject stale or duplicate visual events."""

    hand_id: str
    revision: int

    def __post_init__(self) -> None:
        if not self.hand_id:
            raise ValueError("hand_id cannot be empty")
        if self.revision < 0:
            raise ValueError("revision cannot be negative")


@runtime_checkable
class ScreenAdapter(Protocol):
    """Convert a permitted practice-screen snapshot into public information."""

    def observe(self) -> tuple[SnapshotIdentity, Observation]:
        """Return the latest validated snapshot and its public observation."""

    def confirm_current(self, identity: SnapshotIdentity) -> bool:
        """Return whether the same hand/revision is still current.

        A future executor must call this immediately before applying an action.
        The protocol deliberately contains no click or real-money-site API.
        """


__all__ = [
    "InconsistentSnapshotError",
    "ScreenAdapter",
    "SnapshotIdentity",
]
