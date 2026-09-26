import unittest

from lychee_fd.runtime.apr.contracts import AcousticPcmRecord, AcousticTokenBatch
from lychee_fd.runtime.apr.state_store import AcousticStateStore
from lychee_fd.runtime.apr.worker import APRWorkerError, AcousticBackend, AcousticWorker


def batch(request_id="request-a", sequence_no=0, generation_id=0, state_version=0):
    return AcousticTokenBatch(
        request_id=request_id,
        stream_id=f"stream-{request_id}",
        generation_id=generation_id,
        sequence_no=sequence_no,
        stoken_ids=(sequence_no + 1,),
        source_execution_id=f"exec-{request_id}-{sequence_no}",
        state_version=state_version,
        created_monotonic_ns=sequence_no + 1,
    )


class FakeBackend(AcousticBackend):
    def __init__(self, request_id, fail=False):
        self.request_id = request_id
        self.fail = fail
        self.restored = []
        self.tokens = []
        self.steps = 0

    def restore(self, state):
        self.restored.append(dict(state))

    def process(self, tokens):
        if self.fail:
            raise RuntimeError("synthetic backend failure")
        self.tokens.append(tokens)
        return {"ignored_transition_metadata": True}, [
            AcousticPcmRecord(
                request_id=self.request_id,
                stream_id=f"stream-{self.request_id}",
                generation_id=0,
                sequence_no=len(self.tokens) - 1,
                pcm_bytes=b"\x00\x01",
                sample_rate=24000,
                pcm_seq=len(self.tokens) - 1,
            )
        ]

    def capture(self):
        self.steps += 1
        return {"steps": self.steps}


class AcousticWorkerTests(unittest.TestCase):
    def test_restore_process_capture_and_atomic_pcm_commit(self):
        store = AcousticStateStore()
        events = []
        backend = FakeBackend("request-a")
        worker = AcousticWorker(store, backend, worker_slot=2, event_sink=events.append)

        result = worker.process(batch())

        self.assertEqual(result.state_version, 1)
        self.assertEqual(len(result.pcm_records), 1)
        self.assertEqual(backend.restored, [{}])
        self.assertEqual(backend.tokens, [(1,)])
        self.assertEqual(store.snapshot("request-a")["state"], {"steps": 1})
        self.assertFalse(store.snapshot("request-a")["active_lease"])
        self.assertEqual([event["event_type"] for event in events], ["APR_ACOUSTIC_START", "APR_ACOUSTIC_END"])

    def test_stale_generation_is_rejected_and_reported(self):
        store = AcousticStateStore()
        lease = store.acquire("request-a", generation_id=0, expected_version=0)
        store.release("request-a")
        store.cancel("request-a", generation_id=0)
        events = []
        worker = AcousticWorker(store, FakeBackend("request-a"), event_sink=events.append)

        with self.assertRaises(APRWorkerError):
            worker.process(batch())
        self.assertEqual(events[-1]["event_type"], "APR_ERROR")
        self.assertEqual(events[-1]["request_id"], "request-a")
        self.assertIsNotNone(lease)

    def test_backend_failure_releases_slot_and_does_not_poison_other_request(self):
        store = AcousticStateStore()
        failed_events = []
        failed_worker = AcousticWorker(store, FakeBackend("request-a", fail=True), event_sink=failed_events.append)
        with self.assertRaises(APRWorkerError):
            failed_worker.process(batch())
        self.assertFalse(store.snapshot("request-a")["active_lease"])
        self.assertEqual(failed_events[-1]["event_type"], "APR_ERROR")

        success_worker = AcousticWorker(store, FakeBackend("request-b"))
        result = success_worker.process(batch(request_id="request-b"))
        self.assertEqual(result.request_id, "request-b")


if __name__ == "__main__":
    unittest.main()
