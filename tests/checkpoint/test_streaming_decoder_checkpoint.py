import importlib.machinery
import sys
import types
import unittest
from unittest.mock import patch

# The decoder unit does not use datasets, but the production module imports it
# eagerly. Keep this test focused on decoder behavior in the pinned container.
for _optional_module in ("datasets", "librosa", "matplotlib", "matplotlib.pyplot"):
    _stub = sys.modules.setdefault(_optional_module, types.ModuleType(_optional_module))
    _stub.__spec__ = importlib.machinery.ModuleSpec(_optional_module, loader=None)
_peft_stub = types.ModuleType("peft")
_peft_stub.PeftModel = type("PeftModel", (), {})
_peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
sys.modules.setdefault("peft", _peft_stub)

from lychee_fd.runtime.vllm_generation import StreamingDecoder


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join({1: "A", 2: "B", 3: "C"}.get(int(token), "?") for token in token_ids)


class _Framework:
    text_pad_token_id = 999
    tts_pad_id = 998
    stoken_delay_token_id = 151695
    stoken_pad_token_id = 151694


def _events():
    return [
        {"type": "state_change", "from": "L", "to": "S"},
        {"type": "speaking_token", "text_token": 1, "stoken": 151697, "step": 1},
        {"type": "speaking_token", "text_token": 2, "stoken": 151698, "step": 2},
        {"type": "speaking_done"},
    ]


def _new_decoder():
    return StreamingDecoder(
        _Tokenizer(),
        _Framework(),
        tts_chunk_size=2,
        acoustic_trace_context={
            "request_id": "request-a",
            "generation_id": 7,
            "runtime_mode": "test",
        },
    )


class StreamingDecoderCheckpointTests(unittest.TestCase):
    def test_capture_restore_matches_continuous_event_and_audio_sequence(self):
        events = _events()
        with patch(
            "lychee_fd.runtime.vllm_generation.write_event",
            lambda event: None,
        ), patch(
            "lychee_fd.runtime.vllm_generation.write_ownership_event",
            lambda *args, **kwargs: None,
        ):
            continuous_decoder = _new_decoder()
            continuous = []
            for event in events:
                continuous.extend(continuous_decoder.feed(event))

            first_decoder = _new_decoder()
            migrated = []
            for event in events[:2]:
                migrated.extend(first_decoder.feed(event))
            checkpoint = first_decoder.capture_checkpoint()
            payload = checkpoint.serialize()

            restored_decoder = _new_decoder()
            restored_decoder.restore_checkpoint(payload)
            for event in events[2:]:
                migrated.extend(restored_decoder.feed(event))

        self.assertEqual(migrated, continuous)
        self.assertEqual(checkpoint.event_id, "evt-1")
        self.assertEqual(restored_decoder.capture_checkpoint().event_counter, 1)
        self.assertEqual(
            restored_decoder.capture_checkpoint().acoustic_trace_context["request_id"],
            "request-a",
        )

    def test_restore_rejects_incompatible_decoder_configuration(self):
        with patch(
            "lychee_fd.runtime.vllm_generation.write_event",
            lambda event: None,
        ), patch(
            "lychee_fd.runtime.vllm_generation.write_ownership_event",
            lambda *args, **kwargs: None,
        ):
            source = _new_decoder()
            source.feed(_events()[0])
            checkpoint = source.capture_checkpoint()
            target = StreamingDecoder(_Tokenizer(), _Framework(), tts_chunk_size=3)

        with self.assertRaises(ValueError):
            target.restore_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
