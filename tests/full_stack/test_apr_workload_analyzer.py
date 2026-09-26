import csv
import tempfile
import unittest
from pathlib import Path

from tools.apr.analyze_apr_workload_suite import analyze_suite, write_outputs


PRIMARY_FIELDS = (
    "run_id",
    "system",
    "workload",
    "concurrency",
    "repeat",
    "elapsed_s",
    "completed_sessions",
    "pcm_chunks",
    "pcm_bytes",
    "session_throughput_sps",
    "useful_audio_throughput_sps",
    "time_to_first_audio_s",
    "completion_latency_s",
    "checkpoint_count",
    "restore_count",
    "worker_switch_count",
    "migration_time_s",
)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class APRWorkloadAnalyzerTests(unittest.TestCase):
    def test_analyzer_pairs_apr_and_original_without_inventing_unmeasured_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for system, value in (("apr", 2.0), ("original_affinity", 1.0)):
                _write_csv(
                    root / "runs" / "C" / "N4" / system / "attempt-01" / "formal_primary_metrics.csv",
                    [
                        {
                            "run_id": f"{system}-r1",
                            "system": system,
                            "workload": "C",
                            "concurrency": 4,
                            "repeat": 1,
                            "elapsed_s": 4,
                            "completed_sessions": 4,
                            "pcm_chunks": 4,
                            "pcm_bytes": 100,
                            "session_throughput_sps": value,
                            "useful_audio_throughput_sps": value,
                            "time_to_first_audio_s": 0.5,
                            "completion_latency_s": 2.0,
                            "checkpoint_count": 8,
                            "restore_count": 4,
                            "worker_switch_count": 2,
                            "migration_time_s": "",
                        }
                    ],
                )
            result = analyze_suite(root, repeats=1)

            cell = next(
                row
                for row in result["matrix"]
                if row["workload"] == "C"
                and row["concurrency"] == 4
                and row["system"] == "apr"
            )
            self.assertEqual(cell["valid_count"], 1)
            gain = next(
                row
                for row in result["gain"]
                if row["workload"] == "C" and row["concurrency"] == 4
            )
            self.assertEqual(gain["session_throughput_ratio"], 2.0)
            self.assertEqual(gain["session_ratio_mean"], 2.0)
            self.assertEqual(gain["session_ratio_median"], 2.0)
            self.assertEqual(gain["session_ratio_min"], 2.0)
            self.assertEqual(gain["session_ratio_max"], 2.0)
            self.assertEqual(gain["positive_repeat_count"], 1)
            self.assertEqual(gain["negative_repeat_count"], 0)
            self.assertEqual(gain["direction_consistency"], "CONSISTENT_POSITIVE")
            self.assertEqual(cell["checkpoint_overhead_s"], "NOT_MEASURED")

    def test_write_outputs_creates_report_and_csv_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = analyze_suite(root, repeats=1)
            outputs = write_outputs(root, result)

            self.assertTrue(outputs["report"].is_file())
            self.assertTrue(outputs["primary"].is_file())
            self.assertTrue(outputs["matrix"].is_file())
            self.assertIn("NOT_MEASURED", outputs["report"].read_text(encoding="utf-8"))
            self.assertIn("Advantage region", outputs["report"].read_text(encoding="utf-8"))
            self.assertIn("Experimental setup", outputs["report"].read_text(encoding="utf-8"))
            self.assertIn("Fragmentation analysis", outputs["report"].read_text(encoding="utf-8"))

    def test_primary_rows_with_same_runner_id_from_different_workloads_are_retained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for workload in ("A", "B"):
                _write_csv(
                    root / "runs" / workload / "N4" / "apr" / "attempt-01" / "formal_primary_metrics.csv",
                    [
                        {
                            "run_id": "apr-real-apr-n4-r1-counted",
                            "system": "apr",
                            "workload": workload,
                            "concurrency": 4,
                            "repeat": 1,
                            "elapsed_s": 4,
                            "completed_sessions": 4,
                            "pcm_chunks": 4,
                            "pcm_bytes": 100,
                            "session_throughput_sps": 1.0,
                            "useful_audio_throughput_sps": 1.0,
                            "time_to_first_audio_s": 0.5,
                            "completion_latency_s": 2.0,
                            "checkpoint_count": 8,
                            "restore_count": 4,
                            "worker_switch_count": 2,
                            "migration_time_s": "",
                        }
                    ],
                )
            result = analyze_suite(root, repeats=1)

            self.assertEqual(len(result["primary"]), 2)


if __name__ == "__main__":
    unittest.main()
