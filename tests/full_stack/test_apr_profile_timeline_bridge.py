import unittest


class APRProfileTimelineBridgeTests(unittest.TestCase):
    def test_stage_spans_are_projected_to_canonical_timeline(self):
        from profiling.apr_timeline import TimelineSink
        from lychee_fd.runtime.apr.profiling import StageProfiler

        timeline = TimelineSink(
            enabled=True,
            run_context={
                "run_id": "run-1",
                "system": "apr",
                "workload": "A",
                "concurrency": 8,
                "repeat": 1,
            },
        )
        profiler = StageProfiler(
            enabled=True,
            run_context=timeline.run_context,
            timeline=timeline,
        )
        with profiler.span("STATE_RESTORE", session_id="s1", worker_id=0):
            pass
        names = [event.event for event in timeline.events]
        self.assertEqual(names, ["restore_start", "restore_end"])
        self.assertGreaterEqual(timeline.events[1].duration_ns, 0)

    def test_external_acoustic_span_carries_identity_and_state_size(self):
        from profiling.apr_timeline import TimelineSink
        from lychee_fd.runtime.apr.profiling import StageProfiler

        timeline = TimelineSink(
            enabled=True,
            run_context={
                "run_id": "run-1",
                "system": "apr",
                "workload": "A",
                "concurrency": 8,
                "repeat": 1,
            },
        )
        profiler = StageProfiler(
            enabled=True,
            run_context=timeline.run_context,
            timeline=timeline,
        )
        profiler.record_external_span(
            "TOKEN2WAV_FLOW",
            start_monotonic_ns=100,
            end_monotonic_ns=130,
            session_id="s1",
            sequence_no=3,
            state_size_bytes=4096,
        )
        self.assertEqual(profiler.spans[0].state_size_bytes, 4096)
        self.assertEqual([event.event for event in timeline.events], ["token2wav_start", "token2wav_end"])
        self.assertEqual(timeline.events[1].sequence_no, 3)

    def test_model_forward_projects_model_step_marker(self):
        from profiling.apr_timeline import TimelineSink
        from lychee_fd.runtime.apr.profiling import StageProfiler

        timeline = TimelineSink(
            enabled=True,
            run_context={"run_id": "run-1", "system": "apr", "workload": "A", "concurrency": 8, "repeat": 1},
        )
        profiler = StageProfiler(enabled=True, run_context=timeline.run_context, timeline=timeline)
        with profiler.span("MODEL_FORWARD"):
            pass
        self.assertEqual([event.event for event in timeline.events], ["model_start", "model_step", "model_end"])


if __name__ == "__main__":
    unittest.main()
