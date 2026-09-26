"""Regression for the row-aware sampler input contract.

The Official scheduler can execute a chunked-prefill step with ``do_sample``
false.  Native vLLM then passes an empty logits batch to the sampler.  The
row-aware model path must preserve that contract even though it computes
transient row side outputs during the forward pass.
"""

import unittest
from types import SimpleNamespace

import torch

from lychee_fd.vllm_integration.model_lychee import LycheeDuplexForVLLM


class ProcessorStatePersistenceRegressionTest(unittest.TestCase):

    def test_row_aware_non_sampling_step_returns_no_sampler_rows(self):
        model = object.__new__(LycheeDuplexForVLLM)
        model.logits_processor = None
        model._logits_nan_guard = False
        model._project_text_logits = lambda hidden: torch.zeros(
            (hidden.shape[0], 4), dtype=torch.float32, device=hidden.device
        )
        model._apply_processors = lambda logits, input_ids, processors: logits
        model._force_single_token_logits = lambda logits, token: logits

        sampling_metadata = SimpleNamespace(
            seq_groups=[
                SimpleNamespace(
                    sample_indices=[],
                    prompt_logprob_indices=[],
                    query_len=None,
                )
            ],
            selected_token_indices=torch.empty(0, dtype=torch.long),
        )
        request_state = [{
            "request_id": "request-a",
            "row_aware_enabled": True,
            "finished": False,
            "text_processors": None,
            "text_input_ids": None,
        }]
        row_side_outputs = [{
            "request_id": "request-a",
            "physical_row": 0,
            "text": 158358,
            "stoken": 158359,
            "control": 158357,
        }]

        logits = LycheeDuplexForVLLM._compute_row_aware_logits(
            model,
            torch.zeros((1, 8), dtype=torch.float32),
            sampling_metadata,
            request_state,
            row_side_outputs,
        )

        self.assertEqual(tuple(logits.shape), (0, 4))

if __name__ == "__main__":
    unittest.main()
