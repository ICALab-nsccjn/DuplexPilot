import unittest

import torch

from lychee_fd.runtime.token2wav_checkpoint import (
    LocalToken2WavCheckpointAdapter,
)
from lychee_fd.runtime.acoustic_backend_checkpoint import AcousticBackendCheckpoint


class _FakeToken2Wav:
    def stream_with_state(self, tokens, prompt_wav, state, last_chunk=False):
        self._assert_state_shape(state)
        position = int(state["flow_cache"]["position"].item())
        position += len(tokens)
        state["flow_cache"]["position"] = torch.tensor(position)
        state["hift_cache"]["speech"] = torch.arange(position, dtype=torch.float32)
        suffix = b"F" if last_chunk else b"P"
        return f"{prompt_wav}:{position}:".encode("ascii") + suffix

    @staticmethod
    def _assert_state_shape(state):
        assert set(state) == {"flow_cache", "hift_cache"}
        assert "position" in state["flow_cache"]
        assert "speech" in state["hift_cache"]


def _initial_state():
    return {
        "flow_cache": {"position": torch.tensor(0)},
        "hift_cache": {"speech": torch.zeros(0)},
    }


class Token2WavCheckpointTests(unittest.TestCase):
    def test_local_adapter_implements_checkpoint_backend_contract(self):
        self.assertTrue(issubclass(LocalToken2WavCheckpointAdapter, AcousticBackendCheckpoint))

    def test_continuous_and_migrated_processing_have_identical_pcm_and_state(self):
        model = _FakeToken2Wav()
        continuous = LocalToken2WavCheckpointAdapter(
            model,
            request_id="request-a",
            stream_id="stream-a",
            generation_id=7,
            prompt_wav="prompt.wav",
            stream_state=_initial_state(),
        )
        continuous.process((1, 2))
        continuous.process((3, 4), last_chunk=True)
        continuous_pcm = continuous.commit_output()
        continuous_state = continuous.capture_state().restore()

        first = LocalToken2WavCheckpointAdapter(
            model,
            request_id="request-a",
            stream_id="stream-a",
            generation_id=7,
            prompt_wav="prompt.wav",
            stream_state=_initial_state(),
        )
        first.process((1, 2))
        checkpoint = first.capture_state()
        second = LocalToken2WavCheckpointAdapter(
            model,
            request_id="request-a",
            stream_id="stream-a",
            generation_id=7,
            prompt_wav="prompt.wav",
            stream_state=_initial_state(),
        )
        second.restore_state(checkpoint)
        second.process((3, 4), last_chunk=True)
        migrated_pcm = second.commit_output()
        migrated_state = second.capture_state().restore()

        self.assertEqual(migrated_pcm, continuous_pcm)
        self.assertTrue(torch.equal(
            migrated_state["flow_cache"]["position"],
            continuous_state["flow_cache"]["position"],
        ))
        self.assertTrue(torch.equal(
            migrated_state["hift_cache"]["speech"],
            continuous_state["hift_cache"]["speech"],
        ))
        self.assertEqual(second.capture_state().generation_id, 7)

    def test_cancel_discards_pending_pcm_and_rejects_future_processing(self):
        adapter = LocalToken2WavCheckpointAdapter(
            _FakeToken2Wav(),
            request_id="request-a",
            stream_id="stream-a",
            generation_id=7,
            prompt_wav="prompt.wav",
            stream_state=_initial_state(),
        )
        adapter.process((1,))
        adapter.cancel()

        self.assertEqual(adapter.commit_output(), ())
        with self.assertRaises(RuntimeError):
            adapter.process((2,))

    def test_restore_rejects_a_different_logical_owner(self):
        source = LocalToken2WavCheckpointAdapter(
            _FakeToken2Wav(),
            request_id="request-a",
            stream_id="stream-a",
            generation_id=7,
            prompt_wav="prompt.wav",
            stream_state=_initial_state(),
        )
        checkpoint = source.capture_state()
        target = LocalToken2WavCheckpointAdapter(
            _FakeToken2Wav(),
            request_id="request-b",
            stream_id="stream-a",
            generation_id=7,
            prompt_wav="prompt.wav",
            stream_state=_initial_state(),
        )

        with self.assertRaises(ValueError):
            target.restore_state(checkpoint)


if __name__ == "__main__":
    unittest.main()
