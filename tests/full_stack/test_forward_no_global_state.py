import torch

from lychee_fd.vllm_integration.model_lychee import (
    LycheeDuplexForVLLM,
    LycheeDuplexState,
)


def test_explicit_row_normalization_does_not_consume_class_level_request_state():
    model = object.__new__(LycheeDuplexForVLLM)
    LycheeDuplexState.text_input_ids = torch.tensor([[999]])
    LycheeDuplexState.stoken_input_ids = torch.tensor([[998]])
    LycheeDuplexState.control_input_ids = torch.tensor([[997]])

    row = {
        "request_id": "A",
        "row_aware_enabled": True,
        "text_input_ids": [1, 2],
        "stoken_input_ids": [3],
        "control_input_ids": [4],
        "phase": "listening",
        "finished": False,
    }
    normalized = model._normalize_lychee_side_state(
        [row], physical_rows=1, device=torch.device("cpu")
    )

    assert normalized[0]["text_input_ids"].tolist() == [1, 2]
    assert normalized[0]["stoken_input_ids"].tolist() == [3]
    assert normalized[0]["control_input_ids"].tolist() == [4]

    LycheeDuplexState.reset()
