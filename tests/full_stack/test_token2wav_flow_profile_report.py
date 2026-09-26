import tempfile
import unittest
from pathlib import Path

from profiling.token2wav_flow_profile.contracts import FlowOperatorRecord
from tools.apr.analyze_token2wav_flow_profile import generate_flow_reports


class Token2WavFlowProfileReportTests(unittest.TestCase):
    def _record(self, operator_name, cpu_time_us, *, cuda_time_us=None):
        return FlowOperatorRecord(
            operator_name=operator_name,
            cpu_time_us=cpu_time_us,
            cuda_time_us=cuda_time_us,
            memory_bytes=128,
            kernel_count=2,
            input_shapes=((2, 4),),
            output_shapes=((2, 8),),
            request_id="request-a",
            generation_id=1,
            batch_signature="flow-v1",
        )

    def test_operator_report_ranks_top_ten_and_retains_shapes(self):
        records = tuple(
            self._record(
                f"operator-{index:02d}",
                1000 - index,
                cuda_time_us=2000 - index,
            )
            for index in range(11)
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = generate_flow_reports(records, temporary)
            report = Path(paths["operator_profile"]).read_text(encoding="utf-8")

        self.assertLess(report.index("operator-00"), report.index("operator-09"))
        self.assertNotIn("operator-10", report)
        self.assertIn("input_shapes=((2, 4),)", report)
        self.assertIn("output_shapes=((2, 8),)", report)

    def test_invalid_attempts_are_excluded_from_aggregates(self):
        records = (self._record("valid-op", 100, cuda_time_us=80),)
        invalid = ({"attempt_id": "bad-1", "reason": "invalid PCM"},)
        with tempfile.TemporaryDirectory() as temporary:
            paths = generate_flow_reports(
                records, temporary, invalid_attempts=invalid
            )
            report = Path(paths["operator_profile"]).read_text(encoding="utf-8")

        self.assertIn("invalid attempts: 1", report)
        self.assertIn("valid-op", report)
        self.assertNotIn("invalid PCM", report)

    def test_missing_cuda_is_reported_without_fabricated_cuda_time(self):
        records = (self._record("cpu-only-op", 100),)
        with tempfile.TemporaryDirectory() as temporary:
            paths = generate_flow_reports(records, temporary)
            report = Path(paths["operator_profile"]).read_text(encoding="utf-8")

        self.assertIn("CUDA attribution: unavailable", report)
        self.assertIn("cuda_time_us: unknown", report)
        self.assertNotIn("cuda_time_us: 0", report)

    def test_decision_report_never_claims_e2e_gain_from_microbenchmark_only(self):
        records = (self._record("flow-op", 100, cuda_time_us=80),)
        with tempfile.TemporaryDirectory() as temporary:
            paths = generate_flow_reports(records, temporary)
            report = Path(paths["decision"]).read_text(encoding="utf-8")

        self.assertIn("DIAGNOSTIC_ONLY", report)
        self.assertIn("E2E_GAIN_ESTABLISHED: NO", report)
        self.assertNotIn("E2E_GAIN_ESTABLISHED: YES", report)


if __name__ == "__main__":
    unittest.main()
