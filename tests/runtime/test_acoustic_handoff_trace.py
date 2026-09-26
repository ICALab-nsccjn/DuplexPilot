import json
import tempfile
import unittest
from pathlib import Path

from lychee_fd.runtime.acoustic_handoff_trace import describe_value, write_event


class AcousticHandoffTraceTests(unittest.TestCase):
    def test_describe_value_preserves_small_sequence_contract(self):
        described = describe_value([158359, 158354])

        self.assertEqual(described["container"], "list")
        self.assertEqual(described["length"], 2)
        self.assertEqual(described["values"], [158359, 158354])

    def test_write_event_is_structured_and_fail_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "handoff.jsonl"

            write_event(
                {"probe": "P1", "request_id": "req-a", "payload": {"value": 1}},
                path=str(path),
            )

            record = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["schema"], "lychee-acoustic-handoff-v1")
            self.assertEqual(record["probe"], "P1")
            self.assertEqual(record["request_id"], "req-a")
            self.assertGreater(record["timestamp_monotonic_ns"], 0)
