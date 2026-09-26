import unittest

from lychee_fd.runtime.apr.contracts import AcousticPcmRecord
from lychee_fd.runtime.apr.state_store import APRStateError, AcousticStateStore


def pcm(request_id="request-a", generation_id=0, sequence_no=0):
    return AcousticPcmRecord(
        request_id=request_id,
        stream_id="stream-a",
        generation_id=generation_id,
        sequence_no=sequence_no,
        pcm_bytes=b"\x00\x01",
        sample_rate=24000,
        pcm_seq=sequence_no,
    )


class AcousticStateStoreTests(unittest.TestCase):
    def test_acquire_commit_snapshot_and_release(self):
        store = AcousticStateStore()
        lease = store.acquire("request-a", generation_id=0, expected_version=0)

        result = store.commit(lease, {"decoder": "v1"}, [pcm()])
        self.assertEqual(result.state_version, 1)
        snapshot = store.snapshot("request-a")
        self.assertEqual(snapshot["state_version"], 1)
        self.assertEqual(snapshot["state"], {"decoder": "v1"})
        self.assertEqual(snapshot["pcm_records"], 1)

        store.release("request-a")
        self.assertFalse(store.snapshot("request-a")["active_lease"])
        store.remove("request-a")
        self.assertIsNone(store.snapshot("request-a"))

    def test_only_one_lease_and_stale_lease_are_rejected(self):
        store = AcousticStateStore()
        first = store.acquire("request-a", generation_id=0, expected_version=0)
        with self.assertRaises(APRStateError):
            store.acquire("request-a", generation_id=0, expected_version=0)

        store.commit(first, {"decoder": "v1"}, [])
        store.release("request-a")
        second = store.acquire("request-a", generation_id=0, expected_version=1)
        with self.assertRaises(APRStateError):
            store.commit(first, {"decoder": "stale"}, [])
        store.release("request-a")
        self.assertIsNotNone(second)

    def test_expected_version_must_match_and_generation_cancel_is_fail_closed(self):
        store = AcousticStateStore()
        lease = store.acquire("request-a", generation_id=0, expected_version=0)
        with self.assertRaises(APRStateError):
            store.acquire("request-a", generation_id=0, expected_version=1)

        store.cancel("request-a", generation_id=0)
        with self.assertRaises(APRStateError):
            store.commit(lease, {"decoder": "cancelled"}, [pcm()])
        store.release("request-a")

    def test_pcm_owner_and_generation_must_match_the_lease(self):
        store = AcousticStateStore()
        lease = store.acquire("request-a", generation_id=0, expected_version=0)
        with self.assertRaises(APRStateError):
            store.commit(lease, {}, [pcm(request_id="request-b")])
        with self.assertRaises(APRStateError):
            store.commit(lease, {}, [pcm(generation_id=1)])
        store.release("request-a")


if __name__ == "__main__":
    unittest.main()
