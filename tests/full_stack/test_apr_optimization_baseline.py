import json
import tempfile
import unittest
from pathlib import Path


class AprOptimizationBaselineTests(unittest.TestCase):
    def _write_fixture(self, root: Path, *, complete_identity: bool = True) -> tuple[Path, list[Path]]:
        report = root / "APR_WORKLOAD_ADVANTAGE_REPORT.md"
        report.write_text(
            "\n".join(
                [
                    "# APR Workload Advantage Report",
                    "- Valid counted runs: `135 / 135` runnable expected runs",
                    "- Invalid counted runs: `0`",
                ]
            ),
            encoding="utf-8",
        )
        manifests = []
        for system in ("apr", "original_affinity"):
            payload = {
                "system": system,
                "concurrency": 4,
                "source_commit": "be0dadcd217a756d8e0cf1ca8ca1fd2b8869f8dc",
                "model_checkpoint": "/models/lychee",
                "token2wav_checkpoint": "/models/token2wav",
                "gpu_mapping": {"model": 0, "acoustic": 1},
                "worker_count": 2,
                "measurement_schema_version": "apr-e2e-v1",
                "metrics": {
                    "throughput_sps": 1.0,
                    "checkpoint_overhead_s": "NOT_MEASURED",
                },
            }
            if not complete_identity:
                payload.pop("token2wav_checkpoint")
            path = root / f"{system}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifests.append(path)
        return report, manifests

    def test_build_baseline_summary_preserves_scope_and_unmeasured_metrics(self):
        from tools.apr.prepare_apr_optimization_baseline import build_baseline_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            report, manifests = self._write_fixture(Path(tmpdir))
            summary = build_baseline_summary(report, manifests)

        self.assertEqual(summary["systems"], ["apr", "original_affinity"])
        self.assertEqual(summary["concurrency"], [4, 8, 16])
        self.assertEqual(
            summary["source_report"],
            "reports/apr_workload_suite/APR_WORKLOAD_ADVANTAGE_REPORT.md",
        )
        self.assertEqual(summary["valid_counted_runs"], 135)
        self.assertEqual(summary["invalid_counted_runs"], 0)
        self.assertEqual(summary["rows"][0]["checkpoint_overhead_s"], "NOT_MEASURED")

    def test_missing_manifest_identity_fails_closed(self):
        from tools.apr.prepare_apr_optimization_baseline import (
            BaselineContractError,
            build_baseline_summary,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report, manifests = self._write_fixture(Path(tmpdir), complete_identity=False)
            with self.assertRaises(BaselineContractError):
                build_baseline_summary(report, manifests)


if __name__ == "__main__":
    unittest.main()
