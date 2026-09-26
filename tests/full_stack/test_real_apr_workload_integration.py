import unittest
from pathlib import Path

from tools.apr.run_real_apr_e2e import (
    build_run_manifest,
    build_workload_rounds,
    run_real_correctness_attempt,
)
from tools.apr.real_e2e_runner import RealModelAcousticHandoff
from tools.apr.real_e2e_config import RealE2EConfig
from workloads.apr_benchmark import generate_workload, trace_hash


class RealAPRWorkloadIntegrationTests(unittest.TestCase):
    def test_runtime_rounds_follow_workload_arrival_and_have_unique_sessions(self):
        trace = generate_workload("C", concurrency=4, seed=3)
        rounds = build_workload_rounds(trace)

        self.assertTrue(rounds)
        self.assertGreaterEqual(len(rounds), 8)
        self.assertNotIn("session-7", rounds[0])
        self.assertTrue(all(len(set(round_ids)) == len(round_ids) for round_ids in rounds))
        counts = {sid: sum(sid in round_ids for round_ids in rounds) for sid in trace.metadata["durations_s"]}
        self.assertGreaterEqual(min(counts.values()), 8)

    def test_manifest_records_workload_identity_and_trace_hash(self):
        trace = generate_workload("B", concurrency=4, seed=19)
        config = RealE2EConfig(
            model_path=Path("/model"),
            token2wav_path=Path("/token2wav"),
            prompt_wav=Path("/prompt.wav"),
        )
        manifest = build_run_manifest(
            config,
            system="apr",
            concurrency=4,
            repeat=1,
            warmup=False,
            rounds=4,
            performance=True,
            workload_trace=trace,
        )

        self.assertEqual(manifest["workload"], "B")
        self.assertEqual(manifest["workload_trace_hash"], trace_hash(trace))
        self.assertEqual(manifest["workload_summary"]["session_count"], 4)

    def test_fake_runtime_admits_late_sessions_and_completes_trace(self):
        import tempfile

        from tests.test_real_apr_e2e_correctness import (
            FakeModelRunner,
            config_for_test,
            lane_for_test,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_real_correctness_attempt(
                config=config_for_test(),
                system="apr",
                concurrency=8,
                repeat=1,
                out_root=Path(tmpdir),
                model_runner=FakeModelRunner(),
                lane=lane_for_test(),
                workload_trace=generate_workload("C", concurrency=8, seed=3),
            )

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["failed_sessions"]), 0)
        self.assertTrue(all(row["accepted"] for row in result["sessions"]))

    def test_finish_flush_is_committed_as_owned_acoustic_output(self):
        import tempfile

        from tests.test_real_apr_e2e_correctness import (
            FakeModelRunner,
            config_for_test,
            lane_for_test,
        )

        session_ids = tuple(f"session-{index}" for index in range(4))
        handoff = RealModelAcousticHandoff(
            FakeModelRunner(),
            stream_ids={sid: f"stream-{sid}" for sid in session_ids},
            generation_ids={sid: 0 for sid in session_ids},
            acoustic_chunk_size=3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_real_correctness_attempt(
                config=config_for_test(),
                system="apr",
                concurrency=4,
                repeat=1,
                out_root=Path(tmpdir),
                model_runner=FakeModelRunner(),
                lane=lane_for_test(),
                workload_trace=generate_workload("C", concurrency=4, seed=3),
                handoff=handoff,
            )

        self.assertTrue(result["valid"])
        self.assertGreater(result["pcm_chunks"], 0)


if __name__ == "__main__":
    unittest.main()
