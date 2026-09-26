import unittest
from types import SimpleNamespace
import threading

import torch

from lychee_fd.runtime.vllm_generation import _VLLMModelAdapter
from lychee_fd.vllm_integration.engine import _PatchedLycheeVLLMEngine
from lychee_fd.vllm_integration.model_lychee import LycheeDuplexState
from lychee_fd.vllm_integration.sampler import MultiHeadSamplingParams


class RowAwareOnlineAdapterTests(unittest.TestCase):
    def test_request_state_marks_logical_row_and_keeps_audio_context(self):
        engine = object.__new__(_PatchedLycheeVLLMEngine)
        engine._strict_native_side_tokens = False
        params = MultiHeadSamplingParams()

        state = engine._build_multihead_request_state(
            request_id="session-A",
            text_ids=torch.tensor([[11, 12]]),
            stoken_ids=torch.tensor([[152418, 152419]]),
            control_ids=torch.tensor([[153228, 153229]]),
            keep_alive=False,
            sampling_params=params,
            row_aware=True,
            audio_embeds=torch.zeros((1, 2, 4)),
            audio_feat_lens=torch.tensor([2]),
            audio_input_ids=torch.tensor([[151690, 151690]]),
            audio_patch_token_id=151690,
            prefix_input_len=3,
        )

        self.assertTrue(state.row_aware_enabled)
        payload = state.to_worker_payload()
        self.assertEqual(payload["request_id"], "session-A")
        self.assertEqual(payload["audio_input_ids"], [151690, 151690])
        self.assertEqual(payload["audio_feat_lens"], [2])

    def test_row_aware_side_tokens_never_fallback_to_global_state(self):
        LycheeDuplexState.out_stoken = 777
        LycheeDuplexState.out_control = 888

        with self.assertRaisesRegex(RuntimeError, "row-aware side output"):
            _PatchedLycheeVLLMEngine._extract_row_aware_tokens(
                SimpleNamespace(
                    request_id="session-A",
                    stoken_token_ids=None,
                    control_token_ids=None,
                ),
                expected_request_id="session-A",
            )

        LycheeDuplexState.reset()

    def test_row_aware_adapter_releases_execution_lock_between_steps(self):
        observed = {}

        class DummyEngine:
            def step_generate_stream(self, **kwargs):
                observed.update(kwargs)
                yield {"step": 0}
                yield {"step": 1}

        adapter = object.__new__(_VLLMModelAdapter)
        adapter._engine = DummyEngine()
        adapter.config = SimpleNamespace(audio_pad_token_id=None)
        adapter.device = "cpu"
        adapter._cached_audio_key = None
        adapter._cached_audio_context = None
        adapter._stream_lock = threading.RLock()
        adapter._lock_event_recorder = None
        adapter.runtime_mode = "dynamic_virtualized"
        adapter.row_aware_enabled = True

        stream = adapter.multi_head_generate_stream(
            request_id="session-A",
            input_ids=1,
            stoken_ids=2,
            control_input_ids=3,
            prefix_input_ids=4,
            audio_input_ids=5,
        )
        self.assertEqual(next(stream), {"step": 0})
        self.assertFalse(adapter._stream_lock._is_owned())
        self.assertEqual(observed["request_id"], "session-A")
        self.assertTrue(observed["row_aware_enabled"])

    def test_stream_lock_scope_without_recorder_acquires_shared_lock(self):
        adapter = object.__new__(_VLLMModelAdapter)
        adapter._stream_lock = threading.RLock()
        adapter._lock_event_recorder = None

        entered = threading.Event()
        release = threading.Event()

        def worker():
            with adapter._stream_lock_scope(("session-B",)):
                entered.set()
                release.wait(timeout=1.0)

        adapter._stream_lock.acquire()
        thread = threading.Thread(target=worker)
        thread.start()
        self.assertFalse(entered.wait(timeout=0.05))
        adapter._stream_lock.release()
        self.assertTrue(entered.wait(timeout=1.0))
        release.set()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
