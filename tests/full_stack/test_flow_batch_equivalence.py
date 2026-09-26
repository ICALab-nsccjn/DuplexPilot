import unittest

import torch

from profiling.token2wav_flow_profile.contracts import FlowStep
from profiling.token2wav_flow_profile.equivalence import (
    FlowBatchEquivalenceRunner,
    compare_single_and_batch,
)


class DeterministicBackend:
    def __init__(self, *, perturb_batched=False):
        self.perturb_batched = perturb_batched

    def execute(self, packed_step):
        output_mel = packed_step.tokens.to(torch.float32)
        output_mel = output_mel + packed_step.speaker[:, :1]
        output_cache = {
            key: value + packed_step.speaker[:, :1].reshape(-1, 1, 1)
            if value.ndim == 3
            else value + packed_step.speaker[:, :1]
            for key, value in packed_step.flow_cache.items()
        }
        if self.perturb_batched and len(packed_step.request_ids) > 1:
            output_mel = output_mel + 1.0
        return output_mel, output_cache


class FlowBatchEquivalenceTests(unittest.TestCase):
    def _step(self, request_id, *, generation_id, sequence_no, token_value):
        return FlowStep(
            request_id=request_id,
            generation_id=generation_id,
            sequence_no=sequence_no,
            tokens=torch.tensor(
                [[token_value, token_value + 1]], dtype=torch.int64
            ),
            speaker=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            flow_cache={
                "flow_h": torch.tensor([[token_value + 0.1]], dtype=torch.float32),
                "flow_k": torch.tensor(
                    [[[token_value + 0.2]]], dtype=torch.float32
                ),
            },
            last_chunk=False,
            n_timesteps=4,
            model_identity="flow-v1",
        )

    def _steps(self):
        return (
            self._step("request-a", generation_id=3, sequence_no=8, token_value=1),
            self._step("request-b", generation_id=4, sequence_no=9, token_value=10),
        )

    def test_batch_and_single_execution_preserve_per_request_mel_order(self):
        result = FlowBatchEquivalenceRunner().compare(
            self._steps(), DeterministicBackend()
        )

        self.assertTrue(result.output_order)
        self.assertEqual(result.mismatches, ())

    def test_batch_and_single_execution_preserve_cache_keys_shapes_and_values(self):
        result = FlowBatchEquivalenceRunner().compare(
            self._steps(), DeterministicBackend()
        )

        self.assertTrue(result.cache_equivalence)
        self.assertTrue(result.numerical_equivalence)

    def test_identity_sequence_and_generation_metadata_are_preserved(self):
        result = FlowBatchEquivalenceRunner().compare(
            self._steps(), DeterministicBackend()
        )

        self.assertTrue(result.state_integrity)

    def test_numerical_mismatch_is_reported_not_hidden(self):
        result = FlowBatchEquivalenceRunner().compare(
            self._steps(), DeterministicBackend(perturb_batched=True)
        )

        self.assertFalse(result.numerical_equivalence)
        self.assertTrue(any("mel" in mismatch for mismatch in result.mismatches))
        self.assertTrue(any("max_abs=" in mismatch for mismatch in result.mismatches))
        self.assertTrue(any("max_rel=" in mismatch for mismatch in result.mismatches))

    def test_empty_comparison_fails_closed(self):
        result = compare_single_and_batch([], [])

        self.assertFalse(result.state_integrity)
        self.assertFalse(result.output_order)
        self.assertFalse(result.cache_equivalence)
        self.assertFalse(result.numerical_equivalence)
        self.assertTrue(result.mismatches)


if __name__ == "__main__":
    unittest.main()
