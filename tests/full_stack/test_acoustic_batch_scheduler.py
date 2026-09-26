import unittest

import torch

from profiling.token2wav_flow_profile.batching import (
    AcousticBatchScheduler,
    IncompatibleFlowState,
)
from profiling.token2wav_flow_profile.contracts import FlowStep


class RecordingBackend:
    def __init__(self, *, fail=False, fail_on_call=None, on_execute=None):
        self.calls = []
        self.fail = fail
        self.fail_on_call = fail_on_call
        self.on_execute = on_execute

    def execute(self, packed_step):
        self.calls.append(tuple(packed_step.request_ids))
        if self.on_execute is not None:
            self.on_execute(packed_step)
        if self.fail or self.fail_on_call == len(self.calls):
            raise RuntimeError("synthetic Flow backend failure")
        output_mel = packed_step.tokens.to(torch.float32)
        output_cache = {
            key: value.clone() for key, value in packed_step.flow_cache.items()
        }
        return output_mel, output_cache


class AcousticBatchSchedulerTests(unittest.TestCase):
    def _step(
        self,
        request_id,
        *,
        generation_id=1,
        model_identity="flow-v1",
        token_value=0,
    ):
        return FlowStep(
            request_id=request_id,
            generation_id=generation_id,
            sequence_no=0,
            tokens=torch.tensor(
                [[token_value, token_value + 1]], dtype=torch.int64
            ),
            speaker=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            flow_cache={
                "flow_h": torch.tensor([[token_value + 0.1]], dtype=torch.float32),
                "flow_k": torch.tensor(
                    [[[token_value + 0.2]]], dtype=torch.float32
                ),
            },
            last_chunk=False,
            n_timesteps=4,
            model_identity=model_identity,
        )

    def test_scheduler_groups_only_matching_signatures(self):
        backend = RecordingBackend()
        scheduler = AcousticBatchScheduler(backend=backend, max_batch_size=4)
        steps = (
            self._step("request-a", token_value=1),
            self._step("request-b", token_value=10),
            self._step("request-c", model_identity="flow-v2", token_value=20),
        )

        results = scheduler.submit(steps)

        self.assertEqual(backend.calls, [("request-a", "request-b"), ("request-c",)])
        self.assertEqual(
            [result.request_ids for result in results],
            [("request-a", "request-b"), ("request-c",)],
        )
        self.assertTrue(all(result.status == "COMMITTED" for result in results))

    def test_results_and_trace_preserve_generation_identity(self):
        backend = RecordingBackend()
        scheduler = AcousticBatchScheduler(backend=backend, max_batch_size=4)
        steps = (
            self._step("request-a", generation_id=3, token_value=1),
            self._step("request-b", generation_id=4, token_value=10),
        )

        results = scheduler.submit(steps)

        self.assertEqual(results[0].generation_ids, (3, 4))
        submit_event = next(
            event for event in scheduler.trace() if event["event"] == "BATCH_SUBMIT"
        )
        self.assertEqual(submit_event["generation_ids"], (3, 4))

    def test_scheduler_respects_max_batch_size_and_ready_order(self):
        backend = RecordingBackend()
        scheduler = AcousticBatchScheduler(backend=backend, max_batch_size=2)
        steps = tuple(
            self._step(f"request-{index}", token_value=index) for index in range(5)
        )

        results = scheduler.submit(steps)

        self.assertEqual(
            backend.calls,
            [("request-0", "request-1"), ("request-2", "request-3"), ("request-4",)],
        )
        self.assertEqual(
            [request_id for result in results for request_id in result.request_ids],
            [f"request-{index}" for index in range(5)],
        )

    def test_incompatible_steps_run_as_explicit_singletons(self):
        backend = RecordingBackend()
        scheduler = AcousticBatchScheduler(backend=backend, max_batch_size=4)
        steps = (
            self._step("request-a", token_value=1),
            self._step("request-b", model_identity="flow-v2", token_value=2),
        )

        results = scheduler.submit(steps)

        self.assertEqual(backend.calls, [("request-a",), ("request-b",)])
        self.assertTrue(all(result.batch_size == 1 for result in results))
        self.assertTrue(all(result.status == "COMMITTED" for result in results))

    def test_cancelled_generation_is_not_submitted(self):
        backend = RecordingBackend()
        scheduler = AcousticBatchScheduler(backend=backend, max_batch_size=4)
        steps = (self._step("request-a"), self._step("request-b", token_value=2))
        scheduler.cancel("request-b", generation_id=1)

        results = scheduler.submit(steps)

        self.assertEqual(backend.calls, [("request-a",)])
        self.assertEqual([result.request_ids for result in results], [("request-a",)])
        self.assertTrue(any(event["event"] == "CANCEL" for event in scheduler.trace()))

    def test_backend_failure_is_invalid_and_does_not_commit_partial_state(self):
        backend = RecordingBackend(fail=True)
        scheduler = AcousticBatchScheduler(backend=backend, max_batch_size=4)
        steps = (self._step("request-a"), self._step("request-b", token_value=2))

        results = scheduler.submit(steps)

        self.assertEqual(backend.calls, [("request-a", "request-b")])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "INVALID")
        self.assertEqual(results[0].request_ids, ("request-a", "request-b"))
        self.assertIn("synthetic Flow backend failure", results[0].reason)
        self.assertTrue(all(event["event"] != "BATCH_COMMIT" for event in scheduler.trace()))

    def test_late_chunk_failure_invalidates_entire_compatibility_group(self):
        backend = RecordingBackend(fail_on_call=2)
        scheduler = AcousticBatchScheduler(backend=backend, max_batch_size=2)
        steps = tuple(
            self._step(f"request-{index}", token_value=index) for index in range(5)
        )

        results = scheduler.submit(steps)

        self.assertEqual(
            backend.calls,
            [("request-0", "request-1"), ("request-2", "request-3")],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "INVALID")
        self.assertEqual(
            results[0].request_ids,
            tuple(f"request-{index}" for index in range(5)),
        )
        self.assertTrue(all(event["event"] != "BATCH_COMMIT" for event in scheduler.trace()))

    def test_non_iterable_submission_is_rejected_with_contract_error(self):
        scheduler = AcousticBatchScheduler(backend=RecordingBackend())

        with self.assertRaises(IncompatibleFlowState) as context:
            scheduler.submit(123)

        self.assertIn("sequence", str(context.exception))

    def test_cancellation_during_execution_cannot_commit(self):
        backend = RecordingBackend()
        scheduler = AcousticBatchScheduler(backend=backend, max_batch_size=4)
        backend.on_execute = lambda packed: scheduler.cancel(
            packed.request_ids[0], generation_id=1
        )

        results = scheduler.submit((self._step("request-a"),))

        self.assertEqual(results[0].status, "CANCELLED")
        self.assertTrue(
            all(event["event"] != "BATCH_COMMIT" for event in scheduler.trace())
        )


if __name__ == "__main__":
    unittest.main()
