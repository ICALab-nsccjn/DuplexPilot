import unittest
from types import SimpleNamespace

from vllm.model_executor.layers.sampler import (
    _get_stepaudio_model_side_output,
    _get_stepaudio_model_side_tokens,
)


def _metadata_with_physical_outputs():
    return SimpleNamespace(
        lychee_row_side_outputs=[
            {
                "request_id": "A",
                "physical_row": 0,
                "text": 10,
                "stoken": 100,
                "control": 1000,
            },
            {
                "request_id": "B",
                "physical_row": 1,
                "text": 20,
                "stoken": 200,
                "control": 2000,
            },
        ]
    )


class ModelToSamplerRequestIdentityAlignmentTests(unittest.TestCase):

    def test_sampler_request_id_resolves_physical_row_side_output_after_reorder(self):
        """Sampler row 0 can represent physical row 1 after logits projection."""
        metadata = _metadata_with_physical_outputs()

        # Physical side-output order is [A, B], while the selected sampler
        # order is [B, A].  The desired contract is identity based.
        output = _get_stepaudio_model_side_output(metadata, "B")
        self.assertEqual(
            (output["request_id"], output["text"], output["stoken"], output["control"]),
            ("B", 20, 200, 2000),
        )
        self.assertEqual(
            _get_stepaudio_model_side_tokens(metadata, 0, "B"),
            (200, 2000),
        )

    def test_sampler_request_id_resolves_same_order_without_positional_assumption(self):
        metadata = _metadata_with_physical_outputs()

        self.assertEqual(
            _get_stepaudio_model_side_tokens(metadata, 0, "A"),
            (100, 1000),
        )

    def test_b1_request_id_lookup_remains_valid(self):
        metadata = SimpleNamespace(
            lychee_row_side_outputs=[
                {"request_id": "A", "stoken": 100, "control": 1000}
            ]
        )
        self.assertEqual(
            _get_stepaudio_model_side_tokens(metadata, 0, "A"),
            (100, 1000),
        )

    def test_missing_or_duplicate_request_id_fails_closed(self):
        missing = SimpleNamespace(
            lychee_row_side_outputs=[
                {"stoken": 100, "control": 1000}
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "missing request identity"):
            _get_stepaudio_model_side_output(missing, "A")

        duplicate = SimpleNamespace(
            lychee_row_side_outputs=[
                {"request_id": "A", "stoken": 100, "control": 1000},
                {"request_id": "A", "stoken": 101, "control": 1001},
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _get_stepaudio_model_side_output(duplicate, "A")
