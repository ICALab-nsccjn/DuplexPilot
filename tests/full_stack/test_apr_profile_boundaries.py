import unittest

from lychee_fd.runtime.apr.acoustic_backend import AcousticBackend
from lychee_fd.runtime.apr.contracts import AcousticPcmRecord, AcousticTokenBatch
from lychee_fd.runtime.apr.profiling import StageProfiler
from lychee_fd.runtime.apr.runtime import AprRuntime
from tools.apr.real_acoustic_lanes import APRLocalAcousticLane
from tools.apr.real_e2e_runner import RealModelAcousticHandoff


class FakeAcousticBackend(AcousticBackend):
    def __init__(self, request_id: str, *, fail: bool = False) -> None:
        self.request_id = request_id
        self.fail = fail
        self.steps = 0
        self.pending_pcm: list[AcousticPcmRecord] = []

    def capture_state(self, request_id: str):
        if request_id != self.request_id:
            raise AssertionError("backend captured foreign request")
        return {"steps": self.steps}

    def restore_state(self, state):
        self.steps = int(state.get("steps", 0))

    def resume(self):
        return None

    def process(self, tokens: tuple[int, ...]) -> None:
        if self.fail:
            raise RuntimeError("synthetic backend failure")
        self.steps += len(tokens)
        self.pending_pcm = [
            AcousticPcmRecord(
                request_id=self.request_id,
                stream_id=f"stream-{self.request_id}",
                generation_id=0,
                sequence_no=0,
                pcm_bytes=b"pcm",
                sample_rate=24000,
                pcm_seq=0,
            )
        ]

    def commit_pcm(self):
        records = tuple(self.pending_pcm)
        self.pending_pcm = []
        return records

    def cancel(self):
        return None


def _batch(request_id: str = "request-a") -> AcousticTokenBatch:
    return AcousticTokenBatch(
        request_id=request_id,
        stream_id=f"stream-{request_id}",
        generation_id=0,
        sequence_no=0,
        stoken_ids=(7,),
        source_execution_id="exec-0",
        state_version=0,
        created_monotonic_ns=1,
    )


def _profiler() -> StageProfiler:
    return StageProfiler(
        enabled=True,
        run_context={
            "run_id": "profile-run",
            "system": "apr",
            "workload": "A",
            "concurrency": 1,
            "repeat": 1,
        },
    )


class APRProfileBoundaryTests(unittest.TestCase):
    def test_one_apr_step_emits_all_boundary_spans_with_one_identity(self):
        profiler = _profiler()
        runtime = AprRuntime(enabled=True, profiler=profiler, worker_count=1)
        runtime.start_request("request-a", FakeAcousticBackend("request-a"))
        runtime.enqueue_token_batch(_batch())

        result = runtime.process_one()

        self.assertIsNotNone(result)
        stages = [span.stage for span in profiler.spans]
        self.assertEqual(
            stages,
            [
                "APR_SCHEDULE_WAIT",
                "STATE_ACQUIRE",
                "STATE_RESTORE",
                "BACKEND_PROCESS",
                "PCM_COMMIT",
                "STATE_CAPTURE",
                "STATE_COMMIT",
            ],
        )
        for span in profiler.spans:
            self.assertEqual(span.session_id, "request-a")
            self.assertEqual(span.generation_id, 0)
            self.assertEqual(span.sequence_no, 0)
            self.assertEqual(span.state_version, 0)
            self.assertIsNone(span.error_type)

    def test_worker_error_closes_active_span_without_success_span(self):
        profiler = _profiler()
        runtime = AprRuntime(enabled=True, profiler=profiler, worker_count=1)
        runtime.start_request("request-a", FakeAcousticBackend("request-a", fail=True))
        runtime.enqueue_token_batch(_batch())

        with self.assertRaises(RuntimeError):
            runtime.process_one()

        by_stage = {span.stage: span for span in profiler.spans}
        self.assertEqual(by_stage["BACKEND_PROCESS"].error_type, "RuntimeError")
        self.assertNotIn("STATE_CAPTURE", by_stage)
        self.assertNotIn("STATE_COMMIT", by_stage)
        self.assertNotIn("PCM_COMMIT", by_stage)
        self.assertTrue(all(span.end_monotonic_ns >= span.start_monotonic_ns for span in profiler.spans))

    def test_real_model_handoff_profiles_model_and_token_transfer(self):
        class FakePlan:
            row_to_request = ("request-a",)

        class FakeModelRunner:
            def run_plan(self, _store, _plan, **_kwargs):
                return {
                    "outputs": [
                        {
                            "request_id": "request-a",
                            "stoken": 3,
                            "generation_id": 0,
                            "sequence_no": 0,
                            "state_version": 0,
                        }
                    ]
                }

        profiler = _profiler()
        handoff = RealModelAcousticHandoff(FakeModelRunner(), profiler=profiler)

        batches = handoff.run_plan(object(), FakePlan())

        self.assertEqual(len(batches), 1)
        stages = [span.stage for span in profiler.spans]
        self.assertEqual(stages, ["MODEL_FORWARD", "TOKEN_TRANSFER"])
        transfer = profiler.spans[-1]
        self.assertEqual(transfer.session_id, "request-a")
        self.assertEqual(transfer.generation_id, 0)
        self.assertEqual(transfer.sequence_no, 0)
        self.assertEqual(transfer.state_version, 0)

    def test_real_acoustic_lane_profiles_admission_pcm_egress_and_cleanup(self):
        profiler = _profiler()
        lane = APRLocalAcousticLane(
            worker_count=1,
            backend_factory=lambda request_id: FakeAcousticBackend(request_id),
            profiler=profiler,
        )
        lane.start(("request-a",))
        lane.submit(_batch())

        progress = lane.process_one()
        lane.close()

        self.assertIsNotNone(progress)
        stages = [span.stage for span in profiler.spans]
        self.assertIn("REQUEST_ADMISSION", stages)
        self.assertIn("PCM_EGRESS", stages)
        self.assertIn("CLEANUP", stages)


if __name__ == "__main__":
    unittest.main()
