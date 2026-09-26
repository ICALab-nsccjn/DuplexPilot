from lychee_fd.vllm_integration.model_lychee import LycheeDuplexForVLLM


def test_row_side_output_producer_keeps_request_identity_and_heads_aligned():
    rows = [
        {"request_id": "A", "physical_row": 0, "phase": "speaking", "interrupt_active": False},
    ]

    outputs = LycheeDuplexForVLLM._build_row_side_outputs(
        rows, text_tokens=[301], stoken_tokens=[401], control_tokens=[501]
    )

    assert outputs == [
        {
            "request_id": "A",
            "physical_row": 0,
            "text": 301,
            "stoken": 401,
            "control": 501,
            "phase": "speaking",
            "interrupt_active": False,
        }
    ]
