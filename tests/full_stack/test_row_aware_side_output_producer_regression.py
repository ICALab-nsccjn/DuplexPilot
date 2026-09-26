import unittest
from types import SimpleNamespace

import torch

from lychee_fd.vllm_integration.engine import _PatchedLycheeVLLMEngine
from lychee_fd.vllm_integration.model_lychee import LycheeDuplexForVLLM
from lychee_fd.vllm_integration.sampler import MultiHeadSamplingParams


class RowAwareSideOutputProducerRegressionTests(unittest.TestCase):
    def test_row_aware_text_head_applies_request_local_processor_with_batch_dim(self):
        model = object.__new__(LycheeDuplexForVLLM)
        model._sample_nan_guard = True
        model._project_text_logits = lambda hidden_row: torch.zeros(
            (1, 6), dtype=torch.float32
        )

        def constrain_text(input_ids, logits):
            self.assertEqual(tuple(input_ids.shape), (1, 2))
            self.assertEqual(input_ids.tolist(), [[41, 42]])
            constrained = torch.full_like(logits, float("-inf"))
            constrained[0, 5] = 20.0
            return constrained

        row = {
            "text_input_ids": torch.tensor([41, 42]),
            "text_processors": [constrain_text],
            "text_sampling": {
                "do_sample": False,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
        }

        self.assertEqual(
            model._sample_row_text_token(torch.zeros((1, 4)), row),
            5,
        )

    def test_row_aware_text_logits_handoff_applies_processor_with_batch_dim(self):
        model = object.__new__(LycheeDuplexForVLLM)
        model._logits_nan_guard = False
        model.logits_processor = None
        row = {
            "request_id": "A",
            "text_input_ids": torch.tensor([41, 42]),
            "text_processors": [],
        }

        def constrain_text(input_ids, logits):
            self.assertEqual(tuple(input_ids.shape), (1, 2))
            self.assertEqual(input_ids.tolist(), [[41, 42]])
            return logits

        row["text_processors"] = [constrain_text]
        model._row_aware_last_hidden_rows = lambda *args, **kwargs: torch.zeros(
            (1, 4), dtype=torch.float32
        )
        model._normalize_lychee_side_state = lambda *args, **kwargs: [row]
        model._row_aware_projection_metadata = lambda sampling_metadata, *args: sampling_metadata
        model._project_text_logits = lambda hidden_rows: torch.zeros(
            (1, 6), dtype=torch.float32
        )
        model._force_single_token_logits = lambda logits, token: logits

        sampling_metadata = SimpleNamespace(
            seq_groups=[SimpleNamespace(sample_indices=[0], prompt_logprob_indices=[])]
        )
        model._compute_row_aware_logits(
            torch.zeros((1, 4), dtype=torch.float32),
            sampling_metadata,
            [row],
            [{"request_id": "A", "text": 5, "stoken": 1, "control": 2}],
        )

    def test_row_aware_sampling_requires_real_model_side_output_consumer(self):
        engine = object.__new__(_PatchedLycheeVLLMEngine)
        engine._strict_native_side_tokens = False

        payload = engine._build_multihead_sampling_payload(
            MultiHeadSamplingParams(),
            require_model_side_tokens=True,
        )

        self.assertTrue(payload["stepaudio_require_model_side_tokens"])

    def test_row_aware_side_head_applies_request_local_processor_before_sampling(self):
        model = object.__new__(LycheeDuplexForVLLM)
        model._sample_nan_guard = True
        model.num_main_layers = 0
        model.num_stoken_layers = 0
        model.num_control_layers = 0
        model.num_merge_layers = 0
        model.total_num_layers = 0
        model.main_layers = []
        model.main_norm = lambda hidden: hidden
        model.embed_tokens = lambda token_ids: torch.zeros(
            (token_ids.reshape(-1).shape[0], 4), dtype=torch.float32
        )
        model._is_current_stream_capturing = lambda: False

        row = {
            "request_id": "A",
            "row_aware_enabled": True,
            "text_input_ids": [7],
            "stoken_input_ids": [8],
            "control_input_ids": [41, 42],
            "control_processors": [],
            "control_sampling": {
                "do_sample": False,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
            "finished": False,
        }

        def project_side_last_logits(hidden_row, *, branch, processors):
            logits = torch.zeros((1, 6), dtype=torch.float32)
            logits[0, 2] = 10.0
            return logits

        def constrain_control(input_ids, logits):
            assert tuple(input_ids.shape) == (1, 2)
            assert input_ids.tolist() == [[41, 42]]
            constrained = torch.full_like(logits, float("-inf"))
            constrained[0, 5] = 20.0
            return constrained

        row["control_processors"] = [constrain_control]
        model._project_side_last_logits = project_side_last_logits
        model._sample_row_text_token = lambda hidden_row, row: 1

        model._forward_row_aware(
            input_ids=torch.tensor([7]),
            positions=torch.tensor([0]),
            kv_caches=[],
            attn_metadata=SimpleNamespace(query_start_loc=[0, 1]),
            lychee_side_state=[row],
        )
        output = model.consume_lychee_row_side_outputs()[0]
        self.assertEqual(output["request_id"], "A")
        self.assertEqual(output["control"], 5)
