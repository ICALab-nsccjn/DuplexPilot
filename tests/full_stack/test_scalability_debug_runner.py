import unittest


class ScalabilityDebugRunnerTests(unittest.TestCase):
    def test_build_matrix_has_five_attempts_for_each_frozen_concurrency(self):
        from tools.scalability_debug.run_scalability_diagnostics import (
            build_diagnostic_matrix,
        )

        rows = build_diagnostic_matrix(
            baseline="B5",
            workload="W1",
            concurrency_values=(2, 3, 4),
            attempts=5,
        )

        self.assertEqual(len(rows), 15)
        self.assertEqual(
            [(row["concurrency"], row["attempt"]) for row in rows[:5]],
            [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
        )
        self.assertTrue(all(row["diagnostic_only"] for row in rows))

    def test_first_missing_stage_reports_acoustic_boundary(self):
        from tools.scalability_debug.run_scalability_diagnostics import (
            first_missing_stage,
        )

        reached = {
            "REQUEST_CREATED",
            "INPUT_RECEIVED",
            "PENDING",
            "ADMISSION_OPPORTUNITY",
            "ADMITTED",
            "PHYSICAL_BATCH_ASSIGNED",
            "MODEL_FORWARD_START",
            "MODEL_FORWARD_END",
            "SAMPLER_START",
            "SAMPLER_END",
        }

        self.assertEqual(first_missing_stage(reached), "ACOUSTIC_OUTPUT_CREATED")

    def test_attempt_summary_is_not_clean_when_any_session_lacks_pcm(self):
        from tools.scalability_debug.run_scalability_diagnostics import (
            summarize_attempt,
        )

        summary = summarize_attempt(
            baseline="B5",
            workload="W1",
            concurrency=4,
            attempt=1,
            sessions=[
                {"session_id": "a", "pcm_events": 2, "done": True},
                {"session_id": "b", "pcm_events": 0, "done": False},
            ],
        )

        self.assertFalse(summary["clean"])
        self.assertEqual(summary["failed_session_ids"], ["b"])
        self.assertEqual(summary["failure_stage"], "ACOUSTIC_OUTPUT_CREATED")

    def test_attempt_summary_accepts_life_runner_done_event(self):
        from tools.scalability_debug.run_scalability_diagnostics import (
            summarize_attempt,
        )

        summary = summarize_attempt(
            baseline="B5",
            workload="W1",
            concurrency=2,
            attempt=1,
            sessions=[
                {"session_id": "a", "pcm_events": 11, "done_event": True},
                {"session_id": "b", "pcm_events": 8, "done_event": True},
            ],
        )

        self.assertTrue(summary["clean"])
        self.assertEqual(summary["failed_session_ids"], [])
        self.assertEqual(summary["failure_stage"], "NONE")


if __name__ == "__main__":
    unittest.main()
