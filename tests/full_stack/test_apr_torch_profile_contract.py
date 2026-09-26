import tempfile
import unittest
from pathlib import Path


class APRTorchProfileContractTests(unittest.TestCase):
    def test_enabled_profile_writes_trace_for_each_existing_model_step(self):
        from tools.apr.torch_profile_apr import run_torch_profile

        class FakeActivity:
            CPU = "cpu"
            CUDA = "cuda"

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeProfiler:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def step(self):
                return None

            def export_chrome_trace(self, path):
                Path(path).write_text("{}", encoding="utf-8")

        class FakeProfilerModule:
            ProfilerActivity = FakeActivity

            @staticmethod
            def schedule(**kwargs):
                return kwargs

            @staticmethod
            def profile(**kwargs):
                return FakeProfiler()

        class FakeTorch:
            profiler = FakeProfilerModule
            cuda = FakeCuda()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "torch_profile_trace.json"
            calls = []

            result = run_torch_profile(
                lambda: calls.append(len(calls)),
                output,
                enabled=True,
                step_count=3,
                torch_module=FakeTorch,
            )

            self.assertEqual(calls, [0, 1, 2])
            self.assertTrue(output.is_file())
            self.assertEqual(result["enabled"], True)
            self.assertEqual(result["model_steps"], 3)
            self.assertEqual(result["profiler_steps"], 3)


if __name__ == "__main__":
    unittest.main()
