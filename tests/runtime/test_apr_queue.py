import unittest

from lychee_fd.runtime.apr.contracts import AcousticTokenBatch
from lychee_fd.runtime.apr.queue import APRBackpressure, APRQueueError, AcousticIngressQueue


def batch(sequence_no, generation_id=0, request_id="request-a"):
    return AcousticTokenBatch(
        request_id=request_id,
        stream_id="stream-a",
        generation_id=generation_id,
        sequence_no=sequence_no,
        stoken_ids=(sequence_no + 1,),
        source_execution_id=f"exec-{generation_id}-{sequence_no}",
        state_version=sequence_no,
        created_monotonic_ns=sequence_no + 1,
    )


class AcousticIngressQueueTests(unittest.TestCase):
    def test_fifo_order_and_depth(self):
        queue = AcousticIngressQueue(request_id="request-a", capacity=3)
        queue.put(batch(0))
        queue.put(batch(1))
        self.assertEqual(queue.depth(), 2)
        self.assertEqual(queue.get().sequence_no, 0)
        self.assertEqual(queue.get().sequence_no, 1)
        self.assertIsNone(queue.get())

    def test_duplicate_and_gap_are_rejected_without_mutating_queue(self):
        queue = AcousticIngressQueue(request_id="request-a", capacity=3)
        queue.put(batch(0))
        with self.assertRaises(APRQueueError):
            queue.put(batch(0))
        with self.assertRaises(APRQueueError):
            queue.put(batch(2))
        self.assertEqual(queue.depth(), 1)

    def test_overflow_is_explicit_backpressure(self):
        queue = AcousticIngressQueue(request_id="request-a", capacity=1)
        queue.put(batch(0))
        with self.assertRaises(APRBackpressure):
            queue.put(batch(1))
        self.assertEqual(queue.depth(), 1)

    def test_generation_cancellation_drains_old_entries_and_allows_new_generation(self):
        queue = AcousticIngressQueue(request_id="request-a", capacity=3)
        queue.put(batch(0))
        queue.put(batch(1))
        self.assertEqual(queue.cancel_generation(0), 2)
        self.assertEqual(queue.depth(), 0)
        queue.put(batch(0, generation_id=1))
        self.assertEqual(queue.get().generation_id, 1)

    def test_request_mismatch_and_closed_queue_fail_closed(self):
        queue = AcousticIngressQueue(request_id="request-a", capacity=2)
        with self.assertRaises(APRQueueError):
            queue.put(batch(0, request_id="request-b"))
        queue.close()
        with self.assertRaises(APRQueueError):
            queue.put(batch(0))
        self.assertIsNone(queue.get())


if __name__ == "__main__":
    unittest.main()
