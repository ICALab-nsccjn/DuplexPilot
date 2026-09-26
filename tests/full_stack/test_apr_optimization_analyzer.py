import unittest


REQUIRED_STAGES = (
    "APR_SCHEDULE_WAIT",
    "STATE_ACQUIRE",
    "STATE_RESTORE",
    "BACKEND_PROCESS",
    "PCM_COMMIT",
    "STATE_CAPTURE",
    "STATE_COMMIT",
)


def _span(stage, *, run_id="run-1", start=100, duration=10, worker_id=0):
    return {
        "profile_schema_version": "apr-profile-v1",
        "run_id": run_id,
        "system": "apr",
        "workload": "A",
        "concurrency": 4,
        "repeat": 1,
        "session_id": "session-0",
        "generation_id": 0,
        "sequence_no": 0,
        "state_version": 0,
        "worker_id": worker_id,
        "event_type": stage,
        "start_monotonic_ns": start,
        "end_monotonic_ns": start + duration,
        "duration_ns": duration,
        "queue_depth": 2,
    }


def _complete_run(run_id="run-1", base=100):
    return [
        _span(stage, run_id=run_id, start=base + index * 20, duration=10 + index)
        for index, stage in enumerate(REQUIRED_STAGES)
    ]


class APROptimizationAnalyzerTests(unittest.TestCase):
    def test_analyzer_computes_stage_and_migration_metrics_without_filling_missing_gpu(self):
        from tools.apr.analyze_apr_optimization import analyze_profile_records

        report = analyze_profile_records(
            _complete_run(),
            gpu_records=[
                {"gpu_id": "0", "gpu_utilization": 40.0, "memory_used": 100.0},
            ],
        )

        self.assertEqual(report["valid_run_ids"], ["run-1"])
        self.assertEqual(report["invalid_run_ids"], [])
        self.assertEqual(report["stages"]["BACKEND_PROCESS"]["count"], 1)
        self.assertEqual(report["stages"]["BACKEND_PROCESS"]["total_ns"], 13)
        self.assertEqual(report["migration"]["count"], 1)
        self.assertGreater(report["migration"]["latency_ns"], 0)
        self.assertEqual(report["scheduler"]["queue_wait_ns"], 10)
        self.assertGreater(report["worker_residency_ns"]["0"], 0)
        self.assertGreater(report["backend_share"], 0.0)
        self.assertEqual(report["gpu"]["GPU0"]["mean_utilization"], 40.0)
        self.assertIsNone(report["gpu"]["GPU1"]["mean_utilization"])

    def test_run_missing_restore_is_invalid_and_excluded_from_aggregates(self):
        from lychee_fd.runtime.apr.profiling import PROFILE_STAGES
        from tools.apr.analyze_apr_optimization import analyze_profile_records

        invalid = [record for record in _complete_run("run-2", base=500) if record["event_type"] != "STATE_RESTORE"]
        report = analyze_profile_records(_complete_run() + invalid)

        self.assertEqual(report["valid_run_ids"], ["run-1"])
        self.assertEqual(report["invalid_run_ids"], ["run-2"])
        self.assertEqual(report["stages"]["STATE_RESTORE"]["count"], 1)
        self.assertNotIn("run-2", report["aggregated_run_ids"])
        self.assertEqual(report["optimization_decision"]["selected_candidate"], "none")
        self.assertIn("evidence", report["optimization_decision"]["reason"])
        self.assertTrue(PROFILE_STAGES)


if __name__ == "__main__":
    unittest.main()
