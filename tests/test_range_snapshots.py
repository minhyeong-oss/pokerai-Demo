import gzip
import hashlib
import json

import pytest

from poker_ai.range_snapshots import (
    RangeSnapshotError,
    RangeSnapshotStore,
    canonical_snapshot_bytes,
    snapshot_content_id,
)
from poker_ai.ranges import WeightedRange


def _write_raw(store: RangeSnapshotStore, raw: bytes) -> str:
    content_id = hashlib.sha256(raw).hexdigest()
    path = store.path_for(content_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0))
    return content_id


def test_snapshot_canonical_bytes_and_content_id_are_stable() -> None:
    hand_range = WeightedRange({("As", "Kd"): 3, ("Qc", "Qd"): 1})
    raw = canonical_snapshot_bytes(hand_range)
    assert b"\n" not in raw
    assert raw == (
        b'{"combinations":[["Qc","Qd",250000000000000],'
        b'["Kd","As",750000000000000]],'
        b'"quantization_scale":1000000000000000,'
        b'"schema_version":1,"total_mass":1000000000000000}'
    )
    assert snapshot_content_id(hand_range) == (
        "8e76b083ca84526ec781c79e50b0368b03d7b05cdf0458201abf112808d3f274"
    )
    payload = json.loads(raw)
    assert payload["schema_version"] == 1
    assert payload["quantization_scale"] == 10**15
    assert payload["total_mass"] == 10**15


def test_snapshot_round_trip_is_policy_isolated_and_deterministic(tmp_path) -> None:
    hand_range = WeightedRange({("As", "Kd"): 3, ("Qc", "Qd"): 1})
    store = RangeSnapshotStore(tmp_path, "hero")
    content_id = store.write(hand_range)
    path = store.path_for(content_id)

    assert path == tmp_path / "hero" / content_id[:2] / f"{content_id}.json.gz"
    assert int.from_bytes(path.read_bytes()[4:8], "little") == 0
    first_bytes = path.read_bytes()
    assert store.write(hand_range) == content_id
    assert path.read_bytes() == first_bytes
    restored = store.load(content_id)
    assert restored.masses == hand_range.masses
    assert restored.total_mass == hand_range.total_mass


def test_snapshot_load_rejects_missing_corrupt_and_total_mismatch(tmp_path) -> None:
    store = RangeSnapshotStore(tmp_path, "hero")
    with pytest.raises(RangeSnapshotError, match="missing"):
        store.load("0" * 64)

    corrupt_id = "1" * 64
    corrupt_path = store.path_for(corrupt_id)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"not-gzip")
    with pytest.raises(RangeSnapshotError, match="gzip"):
        store.load(corrupt_id)

    hand_range = WeightedRange({("As", "Kd"): 1})
    payload = json.loads(canonical_snapshot_bytes(hand_range))
    payload["total_mass"] += 1
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    mismatch_id = _write_raw(store, raw)
    with pytest.raises(RangeSnapshotError, match="total_mass"):
        store.load(mismatch_id)
