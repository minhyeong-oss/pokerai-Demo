"""JSON-serializable decision audit records and append-only JSONL output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
import threading
from typing import Any, IO, Optional, Union

from .observation import Observation
from .policy import Decision
from .versions import AUDIT_SCHEMA_VERSION, PROGRAM_VERSION


PathLike = Union[str, Path]


def validate_audit_schema(record: Mapping[str, Any]) -> None:
    """Reject records that claim an unsupported decision-log schema."""

    if not isinstance(record, Mapping):
        raise TypeError("audit record must be a mapping")
    if "schema_version" not in record:
        return
    version = record["schema_version"]
    if version != AUDIT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported audit schema {version!r}; expected {AUDIT_SCHEMA_VERSION}"
        )


def to_jsonable(value: Any) -> Any:
    """Recursively convert project values into standard JSON data types."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("audit records cannot contain NaN or infinity")
        return value
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            converted[key] = to_jsonable(item)
        return converted
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted_items = [to_jsonable(item) for item in value]
        return sorted(converted_items, key=lambda item: repr(item))
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return to_jsonable(value.as_dict())
    raise TypeError("value of type {} is not JSON serializable".format(type(value).__name__))


def _timestamp_text(timestamp: Optional[Union[str, datetime]]) -> str:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if isinstance(timestamp, str):
        if not timestamp.strip():
            raise ValueError("timestamp cannot be empty")
        return timestamp
    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a string, datetime, or None")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_audit_record(
    observation: Observation,
    decision: Decision,
    *,
    timestamp: Optional[Union[str, datetime]] = None,
    model_version: Optional[str] = None,
    range_version: Optional[str] = None,
    calculation_ms: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build one self-contained, directly ``json.dumps``-able record."""

    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")
    if not isinstance(decision, Decision):
        raise TypeError("decision must be a Decision")
    if calculation_ms is not None:
        if isinstance(calculation_ms, bool):
            raise TypeError("calculation_ms must be numeric")
        calculation_ms = float(calculation_ms)
        if not math.isfinite(calculation_ms) or calculation_ms < 0.0:
            raise ValueError("calculation_ms must be finite and non-negative")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    record = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "program_version": PROGRAM_VERSION,
        "timestamp": _timestamp_text(timestamp),
        "model_version": model_version,
        "range_version": range_version,
        "calculation_ms": calculation_ms,
        "observation": observation,
        "decision": decision,
        "metadata": metadata,
    }
    converted = to_jsonable(record)
    # Fail here, close to construction, rather than later in a long simulation.
    json.dumps(converted, ensure_ascii=False, allow_nan=False)
    return converted


class JsonlAuditWriter:
    """Thread-safe append-only writer for one JSON object per line.

    The writer opens lazily, can be used as a context manager, and flushes each
    record by default so a stopped simulation retains its latest decisions.
    """

    def __init__(self, path: PathLike, *, flush: bool = True) -> None:
        self.path = Path(path)
        self.flush = bool(flush)
        self._file = None  # type: Optional[IO[str]]
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._file is None or self._file.closed

    def _open(self) -> IO[str]:
        if self._file is None or self._file.closed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8", newline="\n")
        return self._file

    def write(self, record: object) -> None:
        """Serialize and append one record."""

        payload = to_jsonable(record)
        if isinstance(payload, Mapping):
            validate_audit_schema(payload)
        line = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            handle = self._open()
            handle.write(line + "\n")
            if self.flush:
                handle.flush()

    def write_decision(
        self,
        observation: Observation,
        decision: Decision,
        **audit_fields: Any,
    ) -> dict:
        """Build, write, and return one decision audit record."""

        record = make_audit_record(observation, decision, **audit_fields)
        self.write(record)
        return record

    def close(self) -> None:
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.close()

    def __enter__(self) -> "JsonlAuditWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def append_jsonl(path: PathLike, record: object) -> None:
    """Append one record without retaining an open writer."""

    with JsonlAuditWriter(path) as writer:
        writer.write(record)


# Common capitalization variants kept as aliases at the I/O boundary.
JSONLAuditWriter = JsonlAuditWriter
JSONLWriter = JsonlAuditWriter
JsonlWriter = JsonlAuditWriter


__all__ = [
    "JSONLAuditWriter",
    "JSONLWriter",
    "JsonlAuditWriter",
    "JsonlWriter",
    "append_jsonl",
    "make_audit_record",
    "to_jsonable",
    "validate_audit_schema",
]
