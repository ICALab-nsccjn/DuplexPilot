"""Regression for the Native selected-row sampler contract.

Native vLLM may execute multiple physical rows while only a subset produces
sampler logits in that step.  Row-aware forward must preserve side outputs
for every physical row but return only the rows selected by the sampler
metadata.
"""

import unittest
from types import SimpleNamespace

import torch

from lychee_fd.vllm_integration.model_lychee import LycheeDuplexForVLLM
from vllm.model_executor.layers.sampler import _apply_min_tokens_penalty


class RowAwareSamplerCardinalityRegressionTest(unittest.TestCase):

    def test_mixed_prompt_decode_returns_only_native_selected_rows(self):
        model = object.__new__(LycheeDuplexForVLLM)
        model.logits_processor = None
        model._logits_nan_guard = False
        model._project_text_logits = lambda hidden: hidden[:, :4]
        model._apply_processors = lambda logits, input_ids, processors: logits
        model._force_single_token_logits = lambda logits, token: logits

        sampling_metadata = SimpleNamespace(
            # Physical rows are [prompt-only, sampled].  The sampler sees
            # only the second row after Native selected-token pruning.
            seq_groups=[
                SimpleNamespace(
                    request_id="prompt-only",
                    seq_ids=[10],
                    sample_indices=[],
                    prompt_logprob_indices=[],
                    query_len=1,
                    do_sample=False,
                    sampling_params=SimpleNamespace(
                        min_tokens=0, all_stop_token_ids=[]
                    ),
                ),
                SimpleNamespace(
                    request_id="sampled",
                    seq_ids=[11],
                    sample_indices=[0],
                    prompt_logprob_indices=[],
                    query_len=1,
                    do_sample=True,
                    sampling_params=SimpleNamespace(
                        min_tokens=0, all_stop_token_ids=[]
                    ),
                ),
            ],
            selected_token_indices=torch.tensor([1], dtype=torch.long),
        )
        request_state = [
            {
                "request_id": "prompt-only",
                "row_aware_enabled": True,
                "finished": False,
                "text_processors": None,
                "text_input_ids": None,
            },
            {
                "request_id": "sampled",
                "row_aware_enabled": True,
                "finished": False,
                "text_processors": None,
                "text_input_ids": None,
            },
        ]
        row_side_outputs = [
            {"request_id": "prompt-only", "physical_row": 0, "text": 1},
            {"request_id": "sampled", "physical_row": 1, "text": 2},
        ]

        logits = LycheeDuplexForVLLM._compute_row_aware_logits(
            model,
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                dtype=torch.float32,
            ),
            sampling_metadata,
            request_state,
            row_side_outputs,
        )

        logits = _apply_min_tokens_penalty(logits, sampling_metadata)
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertTrue(torch.equal(logits[0], torch.tensor([5.0, 6.0, 7.0, 8.0])))


if __name__ == "__main__":
    unittest.main()
