import unittest

from workloads.apr_benchmark import (
    BASELINES,
    EVENT_TYPES,
    generate_workload,
    progression_rounds,
    summarize_trace,
    trace_hash,
    write_trace_jsonl,
)


class APRWorkloadSuiteTests(unittest.TestCase):
    def test_generated_trace_is_deterministic_and_complete(self):
        first = generate_workload("A", concurrency=4, seed=17)
        second = generate_workload("A", concurrency=4, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first.concurrency, 4)
        self.assertEqual(
            {event.event_type for event in first.events},
            set(EVENT_TYPES),
        )
        self.assertEqual(
            [event.timestamp_s for event in first.events],
            sorted(event.timestamp_s for event in first.events),
        )

    def test_production_workload_has_heavy_tail_and_selectable_arrival_process(self):
        trace = generate_workload(
            "B", concurrency=16, seed=23, arrival_process="burst"
        )
        summary = summarize_trace(trace)

        self.assertEqual(summary["arrival_process"], "burst")
        self.assertEqual(summary["session_count"], 16)
        self.assertIn("short", summary["duration_classes"])
        self.assertIn("medium", summary["duration_classes"])
        self.assertIn("long", summary["duration_classes"])
        self.assertGreater(summary["max_session_duration_s"], summary["median_session_duration_s"])
        self.assertGreater(summary["duration_variance_s"], 0.0)
        self.assertGreaterEqual(summary["arrival_span_s"], 0.0)
        self.assertGreater(summary["arrival_interarrival_cv"], 0.0)

    def test_all_workloads_have_eight_progression_opportunities_per_session(self):
        for workload in ("A", "B", "C"):
            trace = generate_workload(workload, concurrency=4, seed=17)
            summary = summarize_trace(trace)
            self.assertGreaterEqual(
                summary["min_progression_opportunities"],
                8,
                msg=f"{workload} has insufficient progression opportunities",
            )

    def test_fragmentation_workload_starts_long_sessions_before_late_short_sessions(self):
        trace = generate_workload("C", concurrency=8, seed=3)
        by_session = {}
        for event in trace.events:
            by_session.setdefault(event.session_id, []).append(event)

        starts = {
            session_id: events[0].timestamp_s
            for session_id, events in by_session.items()
        }
        long_starts = [starts[sid] for sid in trace.metadata["long_session_ids"]]
        short_starts = [starts[sid] for sid in trace.metadata["late_short_session_ids"]]
        self.assertLess(max(long_starts), min(short_starts))
        self.assertEqual(trace.metadata["purpose"], "resource_fragmentation_stress")

    def test_experiment_systems_are_frozen(self):
        self.assertEqual(
            BASELINES,
            ("original_affinity", "apr", "apr_no_migration", "no_rsv_dsv_apr"),
        )

    def test_progression_rounds_preserve_arrival_boundary(self):
        trace = generate_workload("C", concurrency=8, seed=3)
        rounds = progression_rounds(trace, quantum_s=0.25)

        self.assertTrue(rounds)
        self.assertNotIn("session-7", rounds[0])
        self.assertIn("session-0", rounds[0])
        self.assertEqual(trace_hash(trace), trace_hash(trace))

    def test_trace_jsonl_contains_manifest_and_canonical_events(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            write_trace_jsonl(generate_workload("A", concurrency=4, seed=1), path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[0].split('"type":')[1].split(",", 1)[0].strip(), '"manifest"')
        self.assertTrue(all('"type": "event"' in line for line in lines[1:]))


if __name__ == "__main__":
    unittest.main()
