from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from poker_ai.actions import LegalActions
from poker_ai.logging import JsonlAuditWriter, make_audit_record, validate_audit_schema
from poker_ai.observation import Observation
from poker_ai.policy import choose_action


def make_observation():
    return Observation(
        hand_id="audit:h0",
        decision_id="audit:h0:d0",
        hole_cards=("As", "Qh"),
        board=(),
        street="preflop",
        hero_index=0,
        button_index=0,
        actor_index=0,
        pot=3000,
        call_cost=1000,
        stacks=(9000, 9000),
        street_contributions=(1000, 1000),
        hand_contributions=(1000, 1000),
        statuses=("active", "active"),
        public_history=(),
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
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["program_version"], "0.2.0")
        self.assertEqual(record["observation"]["hole_cards"], ["As", "Qh"])
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
            self.assertEqual(json.loads(lines[0])["schema_version"], 2)
            self.assertEqual(json.loads(lines[1])["event"], "hand_end")
            self.assertTrue(writer.closed)

    def test_unsupported_audit_schema_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "unsupported audit schema"):
            validate_audit_schema({"schema_version": 1})
        with self.assertRaisesRegex(ValueError, "unsupported audit schema"):
            JsonlAuditWriter("unused.jsonl").write({"schema_version": 99})


if __name__ == "__main__":
    unittest.main()
