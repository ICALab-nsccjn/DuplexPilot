import unittest

from lychee_fd.runtime.apr.contracts import AcousticTokenBatch
from lychee_fd.runtime.apr.queue import AcousticIngressQueue
from lychee_fd.runtime.apr.scheduler import APRSchedulerError, AcousticScheduler


def batch(sequence_no, request_id):
    return AcousticTokenBatch(
        request_id=request_id,
        stream_id=f"stream-{request_id}",
        generation_id=0,
        sequence_no=sequence_no,
        stoken_ids=(sequence_no + 1,),
        source_execution_id=f"{request_id}-exec-{sequence_no}",
        state_version=sequence_no,
        created_monotonic_ns=sequence_no + 1,
    )


class AcousticSchedulerTests(unittest.TestCase):
    def test_round_robin_selection_and_worker_lease_exclusion(self):
        scheduler = AcousticScheduler(worker_count=1)
        queues = {}
        for request_id in ("request-a", "request-b"):
            queues[request_id] = AcousticIngressQueue(request_id=request_id)
            queues[request_id].put(batch(0, request_id))
            scheduler.register(request_id, queues[request_id])
            scheduler.mark_ready(request_id)

        self.assertEqual(scheduler.next_ready(), "request-a")
        self.assertIsNone(scheduler.next_ready())
        queues["request-a"].get()
        scheduler.release("request-a")
        self.assertEqual(scheduler.next_ready(), "request-b")

    def test_ready_requests_rotate_in_registration_order(self):
        scheduler = AcousticScheduler(worker_count=2)
        queues = {}
        for request_id in ("request-a", "request-b", "request-c"):
            queues[request_id] = AcousticIngressQueue(request_id=request_id)
            queues[request_id].put(batch(0, request_id))
            queues[request_id].put(batch(1, request_id))
            scheduler.register(request_id, queues[request_id])
            scheduler.mark_ready(request_id)

        self.assertEqual(scheduler.next_ready(), "request-a")
        scheduler.release("request-a")
        self.assertEqual(scheduler.next_ready(), "request-b")
        scheduler.release("request-b")
        self.assertEqual(scheduler.next_ready(), "request-c")

    def test_unregister_removes_ready_and_leased_request(self):
        scheduler = AcousticScheduler(worker_count=1)
        queue = AcousticIngressQueue(request_id="request-a")
        queue.put(batch(0, "request-a"))
        scheduler.register("request-a", queue)
        scheduler.mark_ready("request-a")
        self.assertEqual(scheduler.next_ready(), "request-a")
        scheduler.unregister("request-a")
        self.assertIsNone(scheduler.next_ready())
        with self.assertRaises(APRSchedulerError):
            scheduler.mark_ready("request-a")

    def test_duplicate_registration_and_unknown_ready_are_rejected(self):
        scheduler = AcousticScheduler()
        queue = AcousticIngressQueue(request_id="request-a")
        scheduler.register("request-a", queue)
        with self.assertRaises(APRSchedulerError):
            scheduler.register("request-a", queue)
        with self.assertRaises(APRSchedulerError):
            scheduler.mark_ready("missing")


if __name__ == "__main__":
    unittest.main()
