import os
import unittest

import numpy as np
import torch

from lychee_fd.runtime.token2wav_checkpoint import Token2WavCheckpoint
from token2wav import Token2wav


MODEL_PATH = os.environ.get(
    "LYCHEEFD_REAL_T2W_MODEL",
    "/mnt/DuplexPilot/data/models/token2wav",
)
PROMPT_WAV = os.environ.get(
    "LYCHEEFD_REAL_T2W_PROMPT",
    "/mnt/DuplexPilot/data/DuplexPilot/lychee_dsv_closure_worktree/frontend/public/clone_24k_mono/default_male.wav",
)


class Token2WavDeterministicReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RUN_REAL_TOKEN2WAV_DETERMINISTIC") != "1":
            raise unittest.SkipTest(
                "set RUN_REAL_TOKEN2WAV_DETERMINISTIC=1 for the real reference"
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for the real reference")
        if not os.path.isdir(MODEL_PATH):
            raise unittest.SkipTest(f"Token2Wav model not found: {MODEL_PATH}")
        if not os.path.isfile(PROMPT_WAV):
            raise unittest.SkipTest(f"prompt wav not found: {PROMPT_WAV}")
        cls.model = Token2wav(MODEL_PATH, float16=False)

    def test_strict_deterministic_checkpoint_continuation_is_bitwise_equal(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
            tokens = ((1493, 4299, 4218, 2049, 528, 2752, 4850, 4569) * 8)[:50]
            split = 25

            continuous_state = self.model.create_stream_state(PROMPT_WAV)
            migrated_state = self.model.create_stream_state(PROMPT_WAV)
            self.model.stream_with_state(
                tokens[:split], PROMPT_WAV, continuous_state, last_chunk=False
            )
            self.model.stream_with_state(
                tokens[:split], PROMPT_WAV, migrated_state, last_chunk=False
            )
            checkpoint = Token2WavCheckpoint.capture(
                request_id="deterministic-reference",
                stream_id="deterministic-stream",
                generation_id=1,
                prompt_wav=PROMPT_WAV,
                stream_state=migrated_state,
            )
            restored_state = checkpoint.restore(
                request_id="deterministic-reference",
                stream_id="deterministic-stream",
            )

            control_pcm = self.model.stream_with_state(
                tokens[split:], PROMPT_WAV, continuous_state, last_chunk=True
            )
            migrated_pcm = self.model.stream_with_state(
                tokens[split:], PROMPT_WAV, restored_state, last_chunk=True
            )
        except RuntimeError as exc:
            if "deterministic" in str(exc).lower():
                self.skipTest(
                    "DETERMINISTIC_REFERENCE_UNAVAILABLE: " + str(exc)
                )
            raise
        finally:
            torch.use_deterministic_algorithms(False)

        control = np.frombuffer(control_pcm, dtype="<i2")
        migrated = np.frombuffer(migrated_pcm, dtype="<i2")
        self.assertEqual(control.shape, migrated.shape)
        self.assertTrue(np.array_equal(control, migrated))


if __name__ == "__main__":
    unittest.main()
