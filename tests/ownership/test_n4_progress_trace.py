import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class N4ProgressTraceTests(unittest.TestCase):
    def _state(self, request_id="A", **changes):
        values = {
            "session_id": request_id,
            "text_input_ids": [1, 2],
            "stoken_input_ids": [3, 4],
            "control_input_ids": [5, 6],
            "phase": "listening",
            "interrupt_active": False,
            "finished": False,
            "row_aware_enabled": True,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    def test_writer_is_disabled_without_explicit_path(self):
        from lychee_fd.runtime.n4_progress_trace import write_event

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(write_event("TEST_EVENT", request_id="A"))

    def test_writer_emits_monotonic_jsonl_without_mutating_payload(self):
        from lychee_fd.runtime.n4_progress_trace import write_event

        payload = {"request_id": "A", "selected_request_ids": ["A", "B"]}
        original = json.loads(json.dumps(payload))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with mock.patch.dict(
                os.environ,
                {"LYCHEEFD_N4_PROGRESS_TRACE_PATH": str(path)},
                clear=True,
            ):
                self.assertTrue(write_event("ADMISSION_OPPORTUNITY", **payload))
                self.assertTrue(write_event("PHYSICAL_BATCH", **payload))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(payload, original)
        self.assertEqual([row["event_type"] for row in rows], [
            "ADMISSION_OPPORTUNITY", "PHYSICAL_BATCH"
        ])
        self.assertLessEqual(
            rows[0]["timestamp_monotonic_ns"],
            rows[1]["timestamp_monotonic_ns"],
        )

    def test_exact_compatibility_uses_frozen_state_contract(self):
        from lychee_fd.runtime.n4_progress_trace import classify_pair

        equal = classify_pair(self._state("A"), self._state("B"))
        divergent = classify_pair(
            self._state("A"),
            self._state("B", stoken_input_ids=[3], phase="speaking"),
        )
        self.assertTrue(equal["exact_compatible"])
        self.assertFalse(divergent["exact_compatible"])
        self.assertEqual(
            divergent["left_state"]["lengths"],
            {"text": 2, "stoken": 2, "control": 2},
        )
        self.assertEqual(
            divergent["right_state"]["lengths"],
            {"text": 2, "stoken": 1, "control": 2},
        )

    def test_virtualizable_and_restored_are_direct_pair_results(self):
        from lychee_fd.runtime.n4_progress_trace import classify_pair

        result = classify_pair(
            self._state("A"),
            self._state("B", text_input_ids=[1, 2, 3]),
        )
        self.assertFalse(result["exact_compatible"])
        self.assertTrue(result["virtualizable_compatible"])
        self.assertTrue(result["restored_opportunity"])
        self.assertEqual(result["policy_can_pack_result"], "NOT_OBSERVABLE")

    def test_virtualizable_rejects_duplicate_or_incomplete_rows(self):
        from lychee_fd.runtime.n4_progress_trace import classify_pair

        duplicate = classify_pair(self._state("A"), self._state("A"))
        incomplete = classify_pair(
            self._state("A"),
            self._state("B", control_input_ids=None),
        )
        self.assertFalse(duplicate["virtualizable_compatible"])
        self.assertFalse(incomplete["virtualizable_compatible"])
        self.assertFalse(incomplete["restored_opportunity"])


if __name__ == "__main__":
    unittest.main()
