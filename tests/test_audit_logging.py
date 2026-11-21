from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from poker_ai.actions import LegalActions
from poker_ai.logging import JsonlAuditWriter, make_audit_record
from poker_ai.observation import Observation
from poker_ai.policy import choose_action


def make_observation():
    return Observation(
        hero_cards=("As", "Qh"),
        board=(),
        street="preflop",
        hero_index=0,
        button_index=0,
        actor_index=0,
        pot=30.0,
        to_call=10.0,
        stacks=(90.0, 90.0),
        street_contributions=(10.0, 10.0),
        hand_contributions=(10.0, 10.0),
        player_statuses=("active", "active"),
        action_history=({"actor_index": 1, "kind": "bet", "amount": 10.0},),
        legal_actions=LegalActions(False, True, True),
    )


class AuditLoggingTests(unittest.TestCase):
    def test_record_is_directly_json_serializable(self):
        observation = make_observation()
        decision = choose_action(
            observation,
            {"equity": 0.35, "ci95_low": 0.30, "ci95_high": 0.40, "simulations": 10000},
        )
        record = make_audit_record(
            observation,
            decision,
            timestamp=datetime(2026, 10, 14, tzinfo=timezone.utc),
            model_version="baseline-v1",
            range_version="uniform-v1",
            calculation_ms=12.5,
            metadata={"seed": 42},
        )

        json.dumps(record, allow_nan=False)
        self.assertEqual(record["timestamp"], "2026-10-14T00:00:00Z")
        self.assertEqual(record["observation"]["hero_cards"], ["As", "Qh"])
        self.assertEqual(record["decision"]["action"]["kind"], "call")
        self.assertEqual(record["metadata"], {"seed": 42})

    def test_jsonl_writer_appends_one_valid_object_per_line(self):
        observation = make_observation()
        decision = choose_action(observation, 0.35)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "audit.jsonl"
            with JsonlAuditWriter(path) as writer:
                writer.write_decision(
                    observation,
                    decision,
                    timestamp="2026-10-14T00:00:00Z",
                )
                writer.write({"event": "hand_end", "net_bb": 4.0})

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["schema_version"], 1)
            self.assertEqual(json.loads(lines[1])["event"], "hand_end")
            self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
