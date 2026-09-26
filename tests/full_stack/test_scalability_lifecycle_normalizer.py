import tempfile
import unittest
from pathlib import Path


class ScalabilityLifecycleNormalizerTests(unittest.TestCase):
    def test_missing_stages_are_explicitly_not_observable(self):
        from tools.scalability_debug.normalize_lifecycle import normalize_attempt

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "client_progress_trace.jsonl").write_text(
                '{"kind":"session_start","session_id":"s1","timestamp_perf_ns":10}\n'
                '{"kind":"sse_event","event_type":"done","session_id":"s1","timestamp_perf_ns":30}\n',
                encoding="utf-8",
            )
            rows = normalize_attempt(
                attempt_dir=root,
                manifest={"run_id": "r1", "attempt_id": "a1", "concurrency": 1},
            )

        by_stage = {row["stage"]: row for row in rows}
        self.assertTrue(by_stage["REQUEST_CREATED"]["observed"])
        self.assertEqual(by_stage["REQUEST_CREATED"]["timestamp_monotonic_ns"], 10)
        self.assertFalse(by_stage["MODEL_FORWARD_START"]["observed"])
        self.assertIsNone(by_stage["MODEL_FORWARD_START"]["timestamp_monotonic_ns"])
        self.assertEqual(by_stage["MODEL_FORWARD_START"]["observation"], "NOT_OBSERVABLE")

    def test_physical_batch_and_p3_are_keyed_to_the_same_request(self):
        from tools.scalability_debug.normalize_lifecycle import normalize_attempt

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "client_progress_trace.jsonl").write_text(
                '{"kind":"session_start","session_id":"s1","timestamp_perf_ns":1}\n',
                encoding="utf-8",
            )
            (root / "n4_progress_trace.jsonl").write_text(
                '{"event_type":"PHYSICAL_BATCH_MEMBERSHIP","timestamp_monotonic_ns":20,"physical_batch_size":2,"physical_rows":{"0":"s1","1":"s2"}}\n',
                encoding="utf-8",
            )
            (root / "acoustic_handoff_trace.jsonl").write_text(
                '{"probe":"P3","request_id":"s1","timestamp_monotonic_ns":40}\n',
                encoding="utf-8",
            )
            rows = normalize_attempt(
                attempt_dir=root,
                manifest={"run_id": "r1", "attempt_id": "a1", "concurrency": 2},
            )

        by_stage = {row["stage"]: row for row in rows if row["session_id"] == "s1"}
        self.assertEqual(by_stage["PHYSICAL_BATCH_ASSIGNED"]["physical_batch_size"], 2)
        self.assertEqual(by_stage["STREAMING_DECODER_START"]["timestamp_monotonic_ns"], 40)


if __name__ == "__main__":
    unittest.main()
