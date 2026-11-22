from poker_ai import __version__
from poker_ai.agent import ResearchPokerAgent
from poker_ai.versions import (
    AUDIT_SCHEMA_VERSION,
    POLICY_MODEL_VERSION,
    PROGRAM_VERSION,
    RANGE_SNAPSHOT_SCHEMA_VERSION,
)


def test_v1_3_version_contract() -> None:
    assert __version__ == PROGRAM_VERSION == "0.2.0"
    assert AUDIT_SCHEMA_VERSION == 2
    assert RANGE_SNAPSHOT_SCHEMA_VERSION == 1
    assert POLICY_MODEL_VERSION == "baseline-ev-v2"
    assert ResearchPokerAgent().model_version == POLICY_MODEL_VERSION
