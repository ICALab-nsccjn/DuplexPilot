from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "Step-Audio2"))

from cosyvoice2.flow.decoder_dit import DiT
from cosyvoice2.flow.flow_matching import CausalCFMStepState, CausalConditionalCFM


def _state(
    request_id: str,
    *,
    step_index: int,
    current_length: int,
    attention_length: int,
    value: float,
) -> CausalCFMStepState:
    t_span = torch.tensor([0.0, 0.2, 0.6, 1.0])
    cnn = tuple(torch.full((1, 2, 8, 2), value + i) for i in range(3))
    att = tuple(
        torch.full((1, 2, 1, attention_length, 4), value + i)
        for i in range(3)
    )
    return CausalCFMStepState(
        x=torch.full((1, 2, current_length), value),
        t=t_span[step_index].reshape(1),
        dt=(t_span[step_index + 1] - t_span[step_index]).reshape(()),
        step_index=step_index,
        t_span=t_span,
        mu=torch.full((1, 2, current_length), value + 1),
        speaker=torch.full((1, 1), value + 2),
        condition=torch.full((1, 2, current_length), value + 3),
        input_cnn_cache=cnn,
        input_att_cache=att,
        request_id=request_id,
        generation_id=int(value),
        sequence_no=int(value),
        version=0,
        _cnn_history=cnn,
        _att_history=att,
    )


def _make_cfm() -> CausalConditionalCFM:
    estimator = DiT(
        in_channels=7,
        out_channels=2,
        depth=1,
        num_heads=1,
        head_dim=2,
        hidden_size=4,
    )
    estimator.eval()
    # Keep the miniature test's worker-local buffers consistent with its
    # hidden size.  The production model uses the corresponding 1024-channel
    # buffers; without this override the legacy B=1 control intentionally
    # assumes production dimensions.
    estimator.cnn_cache_buffer = torch.zeros((1, 2, 8, 2))
    estimator.att_cache_buffer = torch.zeros((1, 2, 1, 20, 4))
    cfm = CausalConditionalCFM(estimator)
    cfm.eval()
    return cfm


def test_real_dit_combined_path_masks_current_and_attention_padding():
    cfm = _make_cfm()
    first = _state(
        "first", step_index=0, current_length=3, attention_length=2, value=1.0
    )
    second = _state(
        "second", step_index=1, current_length=5, attention_length=5, value=2.0
    )

    updated = cfm.advance_chunk_step_variable_mixed_chunk_padding_batch(
        (first, second)
    )

    assert [state.request_id for state in updated] == ["first", "second"]
    assert [state.step_index for state in updated] == [1, 2]
    assert [int(state.x.shape[-1]) for state in updated] == [3, 5]
    assert [int(state.completed_att_cache.shape[3]) for state in updated] == [5, 10]
    assert all(torch.isfinite(state.x).all() for state in updated)


def test_real_dit_combined_valid_rows_match_independent_b1_execution():
    combined = _make_cfm()
    control = _make_cfm()
    control.load_state_dict(combined.state_dict())
    first = _state(
        "first", step_index=0, current_length=3, attention_length=2, value=1.0
    )
    second = _state(
        "second", step_index=1, current_length=5, attention_length=5, value=2.0
    )
    combined_result = combined.advance_chunk_step_variable_mixed_chunk_padding_batch(
        (deepcopy(first), deepcopy(second))
    )
    expected_first = control.advance_chunk_step(deepcopy(first))
    expected_second = control.advance_chunk_step(deepcopy(second))

    for actual, expected in zip(combined_result, (expected_first, expected_second)):
        assert torch.allclose(actual.x, expected.x, rtol=1e-4, atol=1e-5)
        assert torch.allclose(
            actual.completed_att_cache, expected.completed_att_cache,
            rtol=1e-4, atol=1e-5,
        )
        assert torch.allclose(
            actual.completed_cnn_cache, expected.completed_cnn_cache,
            rtol=1e-4, atol=1e-5,
        )
