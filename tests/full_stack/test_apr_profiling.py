import json
import tempfile
import unittest
from pathlib import Path


class StageProfilerTests(unittest.TestCase):
    def _context(self):
        return {
            "run_id": "run-1",
            "system": "apr",
            "workload": "A",
            "concurrency": 4,
            "repeat": 1,
        }

    def test_span_records_context_identity_and_monotonic_duration(self):
        from lychee_fd.runtime.apr.profiling import StageProfiler

        profiler = StageProfiler(enabled=True, run_context=self._context())
        with profiler.span(
            "STATE_RESTORE",
            session_id="s1",
            generation_id=0,
            sequence_no=2,
            state_version=2,
            worker_id=0,
        ):
            pass

        self.assertEqual(len(profiler.spans), 1)
        span = profiler.spans[0]
        self.assertEqual(span.stage, "STATE_RESTORE")
        self.assertEqual(span.session_id, "s1")
        self.assertEqual(span.run_id, "run-1")
        self.assertGreaterEqual(span.end_monotonic_ns, span.start_monotonic_ns)

    def test_invalid_stage_and_missing_session_fail_closed(self):
        from lychee_fd.runtime.apr.profiling import ProfileContractError, StageProfiler

        profiler = StageProfiler(enabled=True, run_context=self._context())
        with self.assertRaises(ProfileContractError):
            with profiler.span("NOT_A_STAGE", session_id="s1"):
                pass
        with self.assertRaises(ProfileContractError):
            with profiler.span("STATE_RESTORE"):
                pass

    def test_exception_is_reraised_and_span_records_error_type(self):
        from lychee_fd.runtime.apr.profiling import StageProfiler

        profiler = StageProfiler(enabled=True, run_context=self._context())
        with self.assertRaises(RuntimeError):
            with profiler.span("BACKEND_PROCESS", session_id="s1"):
                raise RuntimeError("backend failed")

        self.assertEqual(profiler.spans[0].error_type, "RuntimeError")

    def test_disabled_profiler_and_sink_failure_do_not_change_callers(self):
        from lychee_fd.runtime.apr.profiling import StageProfiler

        disabled = StageProfiler(enabled=False, run_context=self._context())
        with disabled.span("STATE_RESTORE", session_id="s1"):
            pass
        self.assertEqual(disabled.spans, [])

        def failing_sink(_span):
            raise RuntimeError("trace sink down")

        profiler = StageProfiler(
            enabled=True,
            run_context=self._context(),
            sink=failing_sink,
        )
        with profiler.span("STATE_RESTORE", session_id="s1"):
            pass
        self.assertEqual(len(profiler.spans), 1)

    def test_flush_jsonl_writes_versioned_valid_records(self):
        from lychee_fd.runtime.apr.profiling import StageProfiler

        profiler = StageProfiler(enabled=True, run_context=self._context())
        with profiler.span("PCM_COMMIT", session_id="s1", queue_depth=3):
            pass
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spans.jsonl"
            profiler.flush_jsonl(path)
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["profile_schema_version"], "apr-profile-v1")
        self.assertEqual(record["event_type"], "PCM_COMMIT")
        self.assertEqual(record["queue_depth"], 3)

    def test_apr_trace_exposes_optional_profile_span_without_changing_events(self):
        from lychee_fd.runtime.apr.profiling import StageProfiler
        from lychee_fd.runtime.apr.trace import APRTrace

        profiler = StageProfiler(enabled=True, run_context=self._context())
        trace = APRTrace(profiler=profiler)
        event = trace.emit("APR_ACOUSTIC_START", request_id="s1")

        self.assertEqual(event["event_type"], "APR_ACOUSTIC_START")
        with trace.profile_span("STATE_RESTORE", session_id="s1"):
            pass
        self.assertEqual(profiler.spans[0].stage, "STATE_RESTORE")


if __name__ == "__main__":
    unittest.main()
