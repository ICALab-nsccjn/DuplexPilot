import json
import tempfile
import unittest
from pathlib import Path


class APRFullStackProfileReportTests(unittest.TestCase):
    def test_report_generator_writes_diagnostic_reports_from_raw_evidence(self):
        from tools.apr.analyze_apr_full_stack_profile import write_reports

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "raw"
            attempt = root / "apr-run" / "attempts" / "apr" / "N8" / "r1"
            attempt.mkdir(parents=True)
            (attempt / "apr_full_timeline.json").write_text(
                json.dumps([
                    {"event": "request_arrival", "session_id": "s1", "timestamp_monotonic_ns": 10},
                    {"event": "APR_checkpoint_end", "session_id": "s1", "timestamp_monotonic_ns": 30, "duration_ns": 20},
                    {"event": "token2wav_end", "session_id": "s1", "timestamp_monotonic_ns": 60, "duration_ns": 30},
                ]),
                encoding="utf-8",
            )
            (attempt / "pipeline_spans.jsonl").write_text(
                "\n".join([
                    json.dumps({"event_type": "STATE_RESTORE", "duration_ns": 20, "session_id": "s1"}),
                    json.dumps({"event_type": "TOKEN2WAV_FLOW", "duration_ns": 30, "session_id": "s1"}),
                ]) + "\n",
                encoding="utf-8",
            )
            (attempt / "attempt_result.json").write_text(
                json.dumps({"valid": True, "elapsed_s": 1.0, "system": "apr", "concurrency": 8}),
                encoding="utf-8",
            )
            (attempt / "profiling_metadata.json").write_text(
                json.dumps({"diagnostic_only": True, "formal_aggregate_eligible": False}),
                encoding="utf-8",
            )
            out = Path(tmpdir) / "reports"
            paths = write_reports(root, out)
            self.assertEqual(len(paths), 5)
            text = (out / "APR_FULL_STACK_PROFILE_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("diagnostic-only", text)
            self.assertIn("TOKEN2WAV_FLOW", text)
            self.assertNotIn("speedup claim", text.lower())

    def test_empty_input_keeps_optimization_unselected(self):
        from tools.apr.analyze_apr_full_stack_profile import summarize_root

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = summarize_root(Path(tmpdir))
        self.assertEqual(summary["decision"], "NO_SAFE_OPTIMIZATION_SELECTED")
        self.assertFalse(summary["attribution_complete"])


if __name__ == "__main__":
    unittest.main()
