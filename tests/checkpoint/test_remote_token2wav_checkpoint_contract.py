import copy
import unittest

from lychee_fd.runtime.remote_token2wav_contract import (
    RemoteToken2WavCheckpoint,
    RemoteToken2WavStateContract,
)


class _MockRemoteToken2Wav(RemoteToken2WavStateContract):
    """In-memory stand-in; production remote service is intentionally untouched."""

    def __init__(self, request_id, stream_id, generation_id):
        self.request_id = request_id
        self.stream_id = stream_id
        self.generation_id = generation_id
        self.position = 0
        self.pending_pcm = []
        self.cancelled = False

    def export_state(self):
        return RemoteToken2WavCheckpoint(
            request_id=self.request_id,
            stream_id=self.stream_id,
            generation_id=self.generation_id,
            token_position=self.position,
            pending_pcm=tuple(self.pending_pcm),
            cancelled=self.cancelled,
        )

    def validate_state(self, state):
        if not isinstance(state, RemoteToken2WavCheckpoint):
            raise TypeError("invalid remote checkpoint type")
        if state.request_id != self.request_id:
            raise ValueError("request owner mismatch")
        if state.stream_id != self.stream_id:
            raise ValueError("stream owner mismatch")
        if state.generation_id != self.generation_id:
            raise ValueError("generation owner mismatch")
        if state.token_position < 0:
            raise ValueError("invalid token position")

    def import_state(self, state):
        self.validate_state(state)
        self.position = state.token_position
        self.pending_pcm = list(state.pending_pcm)
        self.cancelled = state.cancelled

    def resume_stream(self, tokens):
        if self.cancelled:
            raise RuntimeError("cancelled stream cannot resume")
        self.position += len(tokens)
        self.pending_pcm.append(
            f"{self.request_id}:{self.generation_id}:{self.position}".encode()
        )

    def cancel_stream(self):
        self.cancelled = True
        self.pending_pcm.clear()

    def commit_pcm(self):
        if self.cancelled:
            return ()
        output = tuple(self.pending_pcm)
        self.pending_pcm.clear()
        return output


class RemoteToken2WavCheckpointContractTests(unittest.TestCase):
    def test_export_transfer_validate_import_and_resume_preserve_identity_and_output(self):
        source = _MockRemoteToken2Wav("request-a", "stream-a", 3)
        source.resume_stream((1, 2))
        checkpoint = source.export_state()

        transferred = copy.deepcopy(checkpoint)
        target = _MockRemoteToken2Wav("request-a", "stream-a", 3)
        target.validate_state(transferred)
        target.import_state(transferred)
        target.resume_stream((3, 4))

        self.assertEqual(target.commit_pcm(), (
            b"request-a:3:2",
            b"request-a:3:4",
        ))

    def test_wrong_owner_and_generation_are_rejected_before_import(self):
        source = _MockRemoteToken2Wav("request-a", "stream-a", 3)
        checkpoint = source.export_state()

        wrong_owner = _MockRemoteToken2Wav("request-b", "stream-a", 3)
        with self.assertRaises(ValueError):
            wrong_owner.import_state(checkpoint)

        wrong_generation = _MockRemoteToken2Wav("request-a", "stream-a", 4)
        with self.assertRaises(ValueError):
            wrong_generation.import_state(checkpoint)

    def test_cancel_rejects_late_resume_and_discards_stale_output(self):
        backend = _MockRemoteToken2Wav("request-a", "stream-a", 3)
        backend.resume_stream((1,))
        backend.cancel_stream()

        self.assertEqual(backend.commit_pcm(), ())
        with self.assertRaises(RuntimeError):
            backend.resume_stream((2,))


if __name__ == "__main__":
    unittest.main()
