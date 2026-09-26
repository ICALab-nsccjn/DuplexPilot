import unittest

from reference_token2wav_backend import (
    DeterministicReferenceToken2Wav,
)


class Token2WavReferenceBackendTests(unittest.TestCase):
    def test_checkpoint_restore_preserves_exact_pcm_and_event_order(self):
        source = DeterministicReferenceToken2Wav(
            request_id="request-a", stream_id="stream-a", generation_id=2
        )
        source.process((10, 20, 30))
        checkpoint = source.capture_state()
        source.process((40, 50))
        expected_pcm = source.commit_pcm()
        expected_events = source.events

        migrated = DeterministicReferenceToken2Wav(
            request_id="request-a", stream_id="stream-a", generation_id=2
        )
        migrated.restore_state(checkpoint)
        migrated.process((40, 50))

        self.assertEqual(migrated.commit_pcm(), expected_pcm)
        self.assertEqual(migrated.events, expected_events)

    def test_identity_and_cancel_semantics_are_strict(self):
        source = DeterministicReferenceToken2Wav(
            request_id="request-a", stream_id="stream-a", generation_id=2
        )
        checkpoint = source.capture_state()
        wrong_owner = DeterministicReferenceToken2Wav(
            request_id="request-b", stream_id="stream-a", generation_id=2
        )
        with self.assertRaises(ValueError):
            wrong_owner.restore_state(checkpoint)

        source.process((10,))
        source.cancel()
        self.assertEqual(source.commit_pcm(), b"")
        with self.assertRaises(RuntimeError):
            source.process((20,))


if __name__ == "__main__":
    unittest.main()
