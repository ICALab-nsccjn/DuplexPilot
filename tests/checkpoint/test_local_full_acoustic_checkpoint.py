import copy
import os
import unittest

import numpy as np
import torch

from lychee_fd.runtime.acoustic_equivalence import compare_pcm_equivalence
from lychee_fd.runtime.acoustic_checkpoint_state import AcousticCheckpointState
from lychee_fd.runtime.token2wav_checkpoint import LocalToken2WavCheckpointAdapter


class _DeterministicAcousticModel:
    def stream_with_state(self, tokens, state, last_chunk=False):
        output = bytearray()
        for token in tokens:
            state["position"] += 1
            state["events"].append({
                "event_id": state["position"],
                "token": int(token),
                "flush": bool(last_chunk and token == tokens[-1]),
            })
            output.extend(
                int((int(token) * 31 + state["position"] * 7) % 32768)
                .to_bytes(2, "little", signed=True)
            )
        return bytes(output)


class _AdapterAcousticModel:
    def stream_with_state(self, tokens, prompt_wav, state, last_chunk=False):
        output = bytearray()
        for token in tokens:
            state["position"] += 1
            output.extend(
                int((int(token) * 13 + state["position"]) % 32768)
                .to_bytes(2, "little", signed=True)
            )
        return bytes(output)


class LocalFullAcousticCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RUN_REAL_LOCAL_FULL_ACOUSTIC_CHECKPOINT") != "1":
            cls.real_model = None
            return
        from token2wav import Token2wav

        model_path = os.environ.get(
            "LYCHEEFD_REAL_T2W_MODEL", "/mnt/DuplexPilot/data/models/token2wav"
        )
        prompt_wav = os.environ.get(
            "LYCHEEFD_REAL_T2W_PROMPT",
            "/mnt/DuplexPilot/data/DuplexPilot/lychee_dsv_closure_worktree/frontend/public/clone_24k_mono/default_male.wav",
        )
        if not os.path.isdir(model_path) or not os.path.isfile(prompt_wav):
            raise unittest.SkipTest("real Token2Wav assets are unavailable")
        cls.real_model = Token2wav(model_path, float16=False)
        cls.real_prompt_wav = prompt_wav

    def test_capture_serialize_restore_continue_passes_all_declared_levels(self):
        model = _DeterministicAcousticModel()
        source_state = {"position": 0, "events": []}
        decoder_state = {
            "event_ids": [1, 2],
            "flush_boundary": "not_flushed",
            "owner": "request-a",
        }
        source_state["pcm_prefix"] = model.stream_with_state(
            (10, 20), source_state, last_chunk=False
        )
        decoder_state["event_ids"].append(3)
        checkpoint = AcousticCheckpointState.capture(
            request_id="request-a",
            generation_id=4,
            stream_id="stream-a",
            version=1,
            decoder_state=decoder_state,
            token2wav_state=source_state,
            token_buffer=(10, 20),
            flush_state={"last_chunk": False, "boundary": "open"},
            pending_output=(source_state["pcm_prefix"],),
            cpu_rng_state=torch.get_rng_state(),
            cuda_rng_state=(),
            explicit_generator_state={
                "reference": torch.Generator(device="cpu").get_state()
            },
        )
        payload = checkpoint.serialize()
        restored = AcousticCheckpointState.restore(payload)
        restored.validate()

        self.assertEqual(restored.request_id, "request-a")
        self.assertEqual(restored.generation_id, 4)
        self.assertEqual(restored.stream_id, "stream-a")
        self.assertEqual(tuple(restored.decoder_state["event_ids"]), (1, 2, 3))
        self.assertEqual(restored.flush_state["boundary"], "open")

        continued_source = copy.deepcopy(source_state)
        continued_restored = copy.deepcopy(dict(restored.token2wav_state))
        pcm_a = model.stream_with_state((30, 40), continued_source, last_chunk=True)
        pcm_b = model.stream_with_state((30, 40), continued_restored, last_chunk=True)

        self.assertEqual(pcm_a, pcm_b)
        self.assertEqual(continued_source["events"], continued_restored["events"])
        self.assertEqual(continued_source["position"], continued_restored["position"])

    def test_restore_rejects_identity_mismatch_and_snapshot_is_copy_safe(self):
        source_state = {"cache": {"value": 1}}
        checkpoint = AcousticCheckpointState.capture(
            request_id="request-a",
            generation_id=1,
            stream_id="stream-a",
            version=1,
            decoder_state={"events": [1]},
            token2wav_state=source_state,
            token_buffer=(1,),
            flush_state={"last_chunk": False},
            pending_output=(b"pcm",),
            cpu_rng_state=torch.get_rng_state(),
            cuda_rng_state=(),
            explicit_generator_state={},
        )
        source_state["cache"]["value"] = 99
        self.assertEqual(checkpoint.token2wav_state["cache"]["value"], 1)

        with self.assertRaises(ValueError):
            checkpoint.validate(
                request_id="request-b",
                generation_id=1,
                stream_id="stream-a",
            )

    def test_local_adapter_exposes_full_acoustic_checkpoint_with_rng(self):
        model = _AdapterAcousticModel()
        initial = {"position": 0, "flow_cache": {}, "hift_cache": {}}
        adapter = LocalToken2WavCheckpointAdapter(
            model,
            request_id="request-a",
            stream_id="stream-a",
            generation_id=2,
            prompt_wav="prompt.wav",
            stream_state=initial,
        )
        adapter.process((10, 20))
        checkpoint = adapter.capture_acoustic_state(
            decoder_state={"event_ids": [1, 2]},
            token_buffer=(10, 20),
            flush_state={"last_chunk": False},
        )
        serialized = checkpoint.serialize()

        restored = LocalToken2WavCheckpointAdapter(
            model,
            request_id="request-a",
            stream_id="stream-a",
            generation_id=2,
            prompt_wav="prompt.wav",
            stream_state=initial,
        )
        decoder_and_flush = restored.restore_acoustic_state(
            AcousticCheckpointState.restore(serialized)
        )
        restored.process((30,), last_chunk=True)

        self.assertEqual(decoder_and_flush["decoder_state"]["event_ids"], [1, 2])
        self.assertEqual(restored.commit_output(), checkpoint.pending_output + (b"\x89\x01",))

    def test_real_token2wav_full_checkpoint_passes_declared_levels(self):
        if self.real_model is None:
            self.skipTest("set RUN_REAL_LOCAL_FULL_ACOUSTIC_CHECKPOINT=1")
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            tokens = ((1493, 4299, 4218, 2049, 528, 2752, 4850, 4569) * 8)[:50]
            initial_state = self.real_model.create_stream_state(self.real_prompt_wav)
            source = LocalToken2WavCheckpointAdapter(
                self.real_model,
                request_id="real-request",
                stream_id="real-stream",
                generation_id=1,
                prompt_wav=self.real_prompt_wav,
                stream_state=initial_state,
            )
            source.process(tokens[:25], last_chunk=False)
            checkpoint = source.capture_acoustic_state(
                decoder_state={"event_ids": list(range(25)), "flush": False},
                token_buffer=tokens[:25],
                flush_state={"last_chunk": False},
            )
            restored_checkpoint = AcousticCheckpointState.restore(checkpoint.serialize())

            source.process(tokens[25:], last_chunk=True)
            continuous_pcm = b"".join(source.commit_output())

            migrated = LocalToken2WavCheckpointAdapter(
                self.real_model,
                request_id="real-request",
                stream_id="real-stream",
                generation_id=1,
                prompt_wav=self.real_prompt_wav,
                stream_state=self.real_model.create_stream_state(self.real_prompt_wav),
            )
            decoder_state = migrated.restore_acoustic_state(restored_checkpoint)
            migrated.process(tokens[25:], last_chunk=True)
            migrated_pcm = b"".join(migrated.commit_output())
        finally:
            torch.use_deterministic_algorithms(False)

        metrics = compare_pcm_equivalence(
            continuous_pcm, migrated_pcm, sample_rate=24000, migrated_sample_rate=24000
        )
        self.assertEqual(tuple(decoder_state["decoder_state"]["event_ids"]), tuple(range(25)))
        self.assertTrue(metrics.passed)
        self.assertTrue(metrics.bitwise_equal)


if __name__ == "__main__":
    unittest.main()
