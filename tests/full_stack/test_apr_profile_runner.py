import tempfile
import unittest
from pathlib import Path


class APRProfileManifestTests(unittest.TestCase):
    def test_build_profile_manifest_records_frozen_runtime_identity(self):
        from tools.apr.real_e2e_config import RealE2EConfig, build_profile_manifest

        config = RealE2EConfig(
            model_path=Path("/models/lychee"),
            token2wav_path=Path("/models/token2wav"),
            prompt_wav=Path("/assets/prompt.wav"),
            worker_count=2,
        )

        manifest = build_profile_manifest(
            config,
            run_id="profile-apr-a-n4-r1",
            system="apr",
            workload="A",
            concurrency=4,
            repeat=1,
            warmup=False,
            source_commit="abc123",
            workload_trace_hash="trace-hash",
            profiling_enabled=True,
            torch_profiler_enabled=False,
            nsight_enabled=False,
        )

        for key in (
            "profile_schema_version",
            "profiling_enabled",
            "torch_profiler_enabled",
            "nsight_enabled",
            "source_commit",
            "model_checkpoint",
            "token2wav_checkpoint",
            "gpu_mapping",
            "worker_count",
            "workload_trace_hash",
            "measurement_schema_version",
        ):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["run_id"], "profile-apr-a-n4-r1")
        self.assertEqual(manifest["source_commit"], "abc123")
        self.assertEqual(manifest["worker_count"], 2)
        self.assertFalse(manifest["warmup"])


class APRProfileModeTests(unittest.TestCase):
    def test_correctness_mode_runs_without_creating_torch_trace(self):
        from tools.apr.torch_profile_apr import run_torch_profile

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "torch_profile_trace.json"
            calls = []

            run_torch_profile(
                lambda: calls.append("model-step"),
                output,
                enabled=False,
                step_count=3,
            )

            self.assertEqual(calls, ["model-step"] * 3)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
