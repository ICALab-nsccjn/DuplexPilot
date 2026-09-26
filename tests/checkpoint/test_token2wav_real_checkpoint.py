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


class RealToken2WavCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RUN_REAL_TOKEN2WAV_CHECKPOINT") != "1":
            raise unittest.SkipTest("set RUN_REAL_TOKEN2WAV_CHECKPOINT=1 for the real backend gate")
        if not os.path.isdir(MODEL_PATH):
            raise unittest.SkipTest(f"Token2Wav model not found: {MODEL_PATH}")
        if not os.path.isfile(PROMPT_WAV):
            raise unittest.SkipTest(f"prompt wav not found: {PROMPT_WAV}")
        cls.model = Token2wav(MODEL_PATH, float16=False)

    def _assert_state_equal(self, left, right):
        if isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self._assert_state_equal(left[key], right[key])
            return
        if isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self._assert_state_equal(left_item, right_item)
            return
        if hasattr(left, "shape") and hasattr(left, "dtype"):
            self.assertTrue(torch.equal(left, right))
            self.assertEqual(left.device, right.device)
            self.assertEqual(left.dtype, right.dtype)
            return
        self.assertEqual(left, right)

    def test_real_stream_state_capture_restore_matches_continuous_pcm(self):
        seed_tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569]
        tokens = (seed_tokens * 8)[:50]

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        control_state = self.model.create_stream_state(PROMPT_WAV)
        self.model.stream_with_state(
            tokens[:25], PROMPT_WAV, control_state, last_chunk=False
        )
        checkpoint = Token2WavCheckpoint.capture(
            request_id="real-request",
            stream_id="real-stream",
            generation_id=1,
            prompt_wav=PROMPT_WAV,
            stream_state=control_state,
        )
        restored_state = checkpoint.restore(
            request_id="real-request", stream_id="real-stream"
        )
        self.assertTrue(
            torch.equal(
                restored_state["flow_cache"]["estimator_att_cache"],
                checkpoint.stream_state["flow_cache"]["estimator_att_cache"],
            )
        )
        self._assert_state_equal(control_state, restored_state)
        continuous_last = self.model.stream_with_state(
            tokens[25:], PROMPT_WAV, control_state, last_chunk=True
        )
        migrated_last = self.model.stream_with_state(
            tokens[25:], PROMPT_WAV, restored_state, last_chunk=True
        )

        if migrated_last != continuous_last:
            a = np.frombuffer(continuous_last, dtype="<i2").astype(np.int32)
            b = np.frombuffer(migrated_last, dtype="<i2").astype(np.int32)
            self.fail(
                "continued PCM differs: lengths=%s/%s max_abs=%s differing=%s"
                % (len(a), len(b), np.max(np.abs(a - b)), np.count_nonzero(a != b))
            )
        self.assertEqual(
            restored_state["flow_cache"]["estimator_att_cache"].shape,
            control_state["flow_cache"]["estimator_att_cache"].shape,
        )


if __name__ == "__main__":
    unittest.main()
