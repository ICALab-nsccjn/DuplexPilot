import pytest
import torch

from lychee_fd.vllm_integration.model_lychee import LycheeDuplexForVLLM


def _row(request_id, *, finished=False):
    return {
        "request_id": request_id,
        "row_aware_enabled": True,
        "text_input_ids": [1],
        "stoken_input_ids": [2],
        "control_input_ids": [3],
        "finished": finished,
    }


@pytest.mark.parametrize(
    "rows,physical_rows,needle",
    [
        ([_row("")], 1, "request"),
        ([_row("A"), _row("A")], 2, "duplicate"),
        ([_row("A", finished=True)], 1, "finished"),
        ([_row("A")], 2, "row"),
    ],
)
def test_invalid_explicit_state_fails_closed(rows, physical_rows, needle):
    model = object.__new__(LycheeDuplexForVLLM)

    with pytest.raises(RuntimeError, match=needle):
        model._normalize_lychee_side_state(
            rows, physical_rows=physical_rows, device=torch.device("cpu")
        )


def test_mixed_explicit_and_legacy_path_is_rejected_by_forward():
    model = object.__new__(LycheeDuplexForVLLM)
    with pytest.raises(RuntimeError, match="row-aware"):
        model.forward(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            kv_caches=[],
            attn_metadata=None,
            lychee_side_state=[_row("A")],
        )
