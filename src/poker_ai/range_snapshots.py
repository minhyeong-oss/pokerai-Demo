"""Canonical, content-addressed gzip storage for quantized hand ranges."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from .cards import card_code
from .ranges import HandRange, QUANTIZATION_SCALE, WeightedRange
from .versions import RANGE_SNAPSHOT_SCHEMA_VERSION


_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class RangeSnapshotError(RuntimeError):
    """Raised when a persisted range snapshot fails integrity validation."""


def snapshot_payload(hand_range: HandRange) -> dict[str, Any]:
    if not isinstance(hand_range, HandRange):
        raise TypeError("hand_range must be a HandRange")
    combinations = [
        [card_code(combo[0]), card_code(combo[1]), mass]
        for combo, mass in hand_range.masses.items()
    ]
    return {
        "schema_version": RANGE_SNAPSHOT_SCHEMA_VERSION,
        "quantization_scale": QUANTIZATION_SCALE,
        "combinations": combinations,
        "total_mass": hand_range.total_mass,
    }


def canonical_snapshot_bytes(hand_range: HandRange) -> bytes:
    return json.dumps(
        snapshot_payload(hand_range),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot_content_id(hand_range: HandRange) -> str:
    return hashlib.sha256(canonical_snapshot_bytes(hand_range)).hexdigest()


class RangeSnapshotStore:
    """Policy-isolated storage rooted at ``<root>/<policy_id>``."""

    def __init__(self, root: str | Path, policy_id: str) -> None:
        self.root = Path(root)
        if not isinstance(policy_id, str) or _POLICY_ID_RE.fullmatch(policy_id) is None:
            raise ValueError("policy_id contains unsupported characters")
        self.policy_id = policy_id

    def path_for(self, content_id: str) -> Path:
        if not isinstance(content_id, str) or _HASH_RE.fullmatch(content_id) is None:
            raise ValueError("content_id must be a lowercase 64-character SHA-256 hex string")
        return self.root / self.policy_id / content_id[:2] / f"{content_id}.json.gz"

    def write(self, hand_range: HandRange) -> str:
        raw = canonical_snapshot_bytes(hand_range)
        content_id = hashlib.sha256(raw).hexdigest()
        target = self.path_for(content_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            restored = self.load(content_id)
            if canonical_snapshot_bytes(restored) != raw:
                raise RangeSnapshotError("existing snapshot content does not match its ID")
            return content_id

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{content_id}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=handle, mtime=0
                ) as compressed:
                    compressed.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return content_id

    def load(self, content_id: str) -> WeightedRange:
        path = self.path_for(content_id)
        if not path.is_file():
            raise RangeSnapshotError(f"range snapshot is missing: {content_id}")
        try:
            with gzip.open(path, "rb") as compressed:
                raw = compressed.read()
        except (OSError, EOFError) as exc:
            raise RangeSnapshotError("range snapshot gzip data is invalid") from exc
        actual_id = hashlib.sha256(raw).hexdigest()
        if actual_id != content_id:
            raise RangeSnapshotError("range snapshot content hash does not match its path")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RangeSnapshotError("range snapshot is not valid UTF-8 JSON") from exc
        restored = _range_from_payload(payload)
        if canonical_snapshot_bytes(restored) != raw:
            raise RangeSnapshotError("range snapshot is not in canonical JSON form")
        return restored


def _range_from_payload(payload: object) -> WeightedRange:
    if not isinstance(payload, Mapping):
        raise RangeSnapshotError("range snapshot root must be an object")
    if payload.get("schema_version") != RANGE_SNAPSHOT_SCHEMA_VERSION:
        raise RangeSnapshotError("unsupported range snapshot schema_version")
    if payload.get("quantization_scale") != QUANTIZATION_SCALE:
        raise RangeSnapshotError("range snapshot quantization_scale mismatch")
    combinations = payload.get("combinations")
    if not isinstance(combinations, list):
        raise RangeSnapshotError("range snapshot combinations must be a list")
    masses = []
    for entry in combinations:
        if not isinstance(entry, list) or len(entry) != 3:
            raise RangeSnapshotError("range snapshot combination entries are invalid")
        first, second, mass = entry
        masses.append(((first, second), mass))
    try:
        hand_range = WeightedRange.from_masses(masses)
    except (TypeError, ValueError) as exc:
        raise RangeSnapshotError("range snapshot contains invalid cards or masses") from exc
    total_mass = payload.get("total_mass")
    if isinstance(total_mass, bool) or not isinstance(total_mass, int):
        raise RangeSnapshotError("range snapshot total_mass must be an integer")
    if hand_range.total_mass != total_mass:
        raise RangeSnapshotError("range snapshot total_mass mismatch")
    if len(hand_range) != len(combinations):
        raise RangeSnapshotError("range snapshot contains duplicate or zero-mass combos")
    return hand_range


__all__ = [
    "RangeSnapshotError",
    "RangeSnapshotStore",
    "canonical_snapshot_bytes",
    "snapshot_content_id",
    "snapshot_payload",
]
