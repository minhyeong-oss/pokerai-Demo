import hashlib
import hmac

import pytest

from poker_ai.actions import CandidateAction
from poker_ai.seeding import action_candidate_id, derive_analysis_seed, derive_seed


def _reference_seed(master_seed: int, *fields: object) -> int:
    key = hashlib.sha256(f"pokerai-master-v1:{master_seed}".encode()).digest()
    message = b"".join(
        len(str(field).encode()).to_bytes(4, "big") + str(field).encode()
        for field in fields
    )
    return int.from_bytes(hmac.new(key, message, hashlib.sha256).digest()[:8], "big")


def test_seed_derivation_matches_documented_hmac_encoding() -> None:
    expected = _reference_seed(17, "deck", "single", 3)
    assert derive_seed(17, "deck", "single", 3) == expected
    assert derive_seed(17, "deck", "single", 3) == expected
    assert derive_seed(17, "deck", "single", 4) != expected


def test_analysis_domains_and_candidates_are_independent() -> None:
    equity = derive_analysis_seed(
        99,
        purpose="equity",
        decision_index=2,
        candidate_id="none",
        sample_index=0,
    )
    response = derive_analysis_seed(
        99,
        purpose="rollout_response",
        decision_index=2,
        candidate_id="raise:400",
        sample_index=0,
    )
    later_sample = derive_analysis_seed(
        99,
        purpose="rollout_response",
        decision_index=2,
        candidate_id="raise:400",
        sample_index=1,
    )
    assert len({equity, response, later_sample}) == 3
    assert action_candidate_id(CandidateAction("raise", 400)) == "raise:400"


@pytest.mark.parametrize("bad_seed", [-1, True, 1.5])
def test_master_seeds_must_be_non_negative_integers(bad_seed) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_seed(bad_seed, "deck")
