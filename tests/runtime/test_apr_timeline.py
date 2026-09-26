import json
import tempfile
import unittest
from pathlib import Path


class APRTimelineTests(unittest.TestCase):
    def _sink(self, callback=None):
        from profiling.apr_timeline import TimelineSink

        return TimelineSink(
            enabled=True,
            run_context={
                "run_id": "run-1",
                "system": "apr",
                "workload": "A",
                "concurrency": 8,
                "repeat": 1,
            },
            callback=callback,
        )

    def test_span_emits_canonical_start_end_and_duration(self):
        sink = self._sink()
        with sink.span("APR_checkpoint_start", "APR_checkpoint_end", session_id="s1", worker_id=0):
            pass
        self.assertEqual([event.event for event in sink.events], ["APR_checkpoint_start", "APR_checkpoint_end"])
        self.assertGreaterEqual(sink.events[1].duration_ns, 0)
        self.assertEqual(sink.events[1].session_id, "s1")

    def test_sink_is_in_memory_until_explicit_flush_and_has_identity(self):
        sink = self._sink()
        sink.emit("request_arrival", session_id="s1", timestamp_monotonic_ns=10)
        sink.emit("request_finish", session_id="s1", timestamp_monotonic_ns=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "apr_full_timeline.json"
            self.assertFalse(path.exists())
            sink.flush_json(path)
            records = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(records[0]["timeline_schema_version"], "apr-full-timeline-v1")
        self.assertEqual(records[0]["run_id"], "run-1")
        self.assertEqual(records[-1]["event"], "request_finish")

    def test_callback_failure_does_not_change_serving_event(self):
        sink = self._sink(callback=lambda _: (_ for _ in ()).throw(RuntimeError("sink down")))
        event = sink.emit("token_ready", session_id="s1")
        self.assertEqual(len(sink.events), 1)
        self.assertEqual(event.event, "token_ready")

    def test_invalid_event_and_timestamp_regression_fail_closed(self):
        sink = self._sink()
        with self.assertRaises(ValueError):
            sink.emit("not-an-event", session_id="s1")
        sink.emit("request_arrival", session_id="s1", timestamp_monotonic_ns=20)
        sink.emit("model_start", session_id="s1", timestamp_monotonic_ns=10)
        with self.assertRaises(ValueError):
            sink.validate()


if __name__ == "__main__":
    unittest.main()
