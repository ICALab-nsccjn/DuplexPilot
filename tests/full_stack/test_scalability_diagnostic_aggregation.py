import unittest


class ScalabilityDiagnosticAggregationTests(unittest.TestCase):
    def test_session_clean_uses_done_event_and_pcm_count(self):
        from tools.scalability_debug.aggregate_diagnostics import session_is_clean

        self.assertTrue(session_is_clean({"status": "ok", "pcm_events": 3, "done_event": True}))
        self.assertFalse(session_is_clean({"status": "ok", "pcm_events": 0, "done_event": True}))
        self.assertFalse(session_is_clean({"status": "ok", "pcm_events": 3, "done_event": False}))

    def test_boundary_row_keeps_diagnostic_only_and_counts_clean_attempts(self):
        from tools.scalability_debug.aggregate_diagnostics import summarize_root

        rows = summarize_root([
            {"concurrency": 3, "attempt": 1, "inner_manifest": {"session_results": [
                {"status": "ok", "pcm_events": 1, "done_event": True},
            ]}},
            {"concurrency": 3, "attempt": 2, "inner_manifest": {"session_results": [
                {"status": "ok", "pcm_events": 0, "done_event": False},
            ]}},
        ])

        self.assertEqual(rows[0]["clean_attempts"], 1)
        self.assertEqual(rows[0]["attempts"], 2)
        self.assertTrue(rows[0]["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
