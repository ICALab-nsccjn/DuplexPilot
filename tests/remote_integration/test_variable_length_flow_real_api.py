from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "Step-Audio2"))

from cosyvoice2.flow.decoder_dit import DiT
from cosyvoice2.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice2.flow.flow_matching import (
    CausalCFMStepState,
    CausalConditionalCFM,
)


class RowEncodingEstimator:
    """Fake estimator that makes physical CFG row routing observable."""

    out_channels = 2

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def forward_chunk(self, *, x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
        # B=1 control path only.
        self.calls.append({"kind": "single", "x": x.detach().clone()})
        batch = x.shape[0]
        dphi = torch.zeros_like(x)
        cnn = torch.zeros((1, batch, 1, 1), dtype=x.dtype, device=x.device)
        att = torch.zeros((1, batch, 1, x.shape[-1], 2), dtype=x.dtype, device=x.device)
        return dphi, cnn, att

    def forward_chunk_variable(
        self,
        *,
        x,
        mu,
        t,
        spks,
        cond,
        cnn_cache=None,
        att_cache=None,
        attention_cache_lengths=None,
        attention_valid_mask=None,
    ):
        self.calls.append(
            {
                "kind": "variable",
                "x": x.detach().clone(),
                "mu": mu.detach().clone(),
                "t": t.detach().clone(),
                "cnn_cache": None if cnn_cache is None else cnn_cache.detach().clone(),
                "att_cache": None if att_cache is None else att_cache.detach().clone(),
                "attention_cache_lengths": tuple(attention_cache_lengths or ()),
                "attention_valid_mask": (
                    None
                    if attention_valid_mask is None
                    else attention_valid_mask.detach().clone()
                ),
            }
        )
        batch = x.shape[0]
        # Encode the physical row in dphi and in every returned cache row.
        dphi = torch.arange(batch, dtype=x.dtype, device=x.device).view(
            batch, 1, 1
        ).expand_as(x)
        cnn = torch.arange(batch, dtype=x.dtype, device=x.device).view(
            1, batch, 1, 1
        )
        cache_len = 0 if att_cache is None else att_cache.shape[3]
        chunk_len = x.shape[-1]
        att = torch.arange(batch, dtype=x.dtype, device=x.device).view(
            1, batch, 1, 1, 1
        ).expand(1, batch, 1, cache_len + chunk_len, 2)
        return dphi, cnn, att


def _state(
    request_id: str,
    *,
    cache_length: int,
    current_length: int = 3,
    value: float = 1.0,
    step_index: int = 0,
    last_chunk: bool = False,
) -> CausalCFMStepState:
    steps = 2
    t_span = torch.tensor([0.0, 0.5, 1.0])
    att = torch.full(
        (1, 2, 1, cache_length, 2), value, dtype=torch.float32
    )
    return CausalCFMStepState(
        x=torch.full((1, 2, current_length), value),
        t=t_span[step_index].reshape(1),
        dt=(t_span[step_index + 1] - t_span[step_index]).reshape(()),
        step_index=step_index,
        t_span=t_span,
        mu=torch.full((1, 2, current_length), value + 10),
        speaker=torch.full((1, 4), value + 20),
        condition=torch.full((1, 2, current_length), value + 30),
        input_cnn_cache=(torch.full((1, 2, 1, 1), value + 40),) * steps,
        input_att_cache=(att,) * steps,
        request_id=request_id,
        generation_id=int(value),
        sequence_no=int(value),
        version=int(value),
        _cnn_history=(torch.full((1, 2, 1, 1), value + 40),) * steps,
        _att_history=(att,) * steps,
    )


def test_variable_api_accepts_unequal_attention_cache_and_routes_cfg_rows():
    estimator = RowEncodingEstimator()
    cfm = CausalConditionalCFM(estimator, inference_cfg_rate=0.7)
    first = _state("a", cache_length=2, value=1.0)
    second = _state("b", cache_length=5, value=2.0)

    updated = cfm.advance_chunk_step_variable_batch((first, second))

    assert [state.request_id for state in updated] == ["a", "b"]
    call = estimator.calls[-1]
    assert call["kind"] == "variable"
    assert call["x"].shape[0] == 4
    assert call["attention_cache_lengths"] == (2, 5, 2, 5)
    assert call["att_cache"].shape == (1, 4, 1, 5, 2)
    # Current tokens precede cache tokens in the real attention implementation.
    mask = call["attention_valid_mask"]
    assert mask.shape == (4, 3, 8)
    assert mask[0, :, :3].all() and mask[0, :, 3:5].all()
    assert not mask[0, :, 5:].any()
    assert mask[1, :, :8].all()
    # The fake cache encodes physical rows 0,1,2,3.  Logical rows must receive
    # conditional/unconditional pairs (0,2) and (1,3), not adjacent rows.
    assert updated[0].completed_cnn_cache[:, :, 0, 0].tolist() == [[0.0, 2.0]]
    assert updated[1].completed_cnn_cache[:, :, 0, 0].tolist() == [[1.0, 3.0]]
    assert updated[0].completed_att_cache[:, :, 0, :, 0].tolist() == [
        [[0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0]]
    ]
    assert updated[1].completed_att_cache[:, :, 0, :, 0].tolist() == [
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]
    ]
    assert updated[0].completed_att_cache.shape[3] == 5
    assert updated[1].completed_att_cache.shape[3] == 8


def test_variable_api_accepts_different_generation_ids_without_identity_change():
    estimator = RowEncodingEstimator()
    cfm = CausalConditionalCFM(estimator)
    first = _state("a", cache_length=2, value=1.0)
    second = _state("b", cache_length=2, value=2.0)
    second.generation_id = 99
    second.version = 17

    updated = cfm.advance_chunk_step_variable_batch((first, second))

    assert (updated[0].generation_id, updated[0].version) == (1, 1)
    assert (updated[1].generation_id, updated[1].version) == (99, 17)
    assert [state.request_id for state in updated] == ["a", "b"]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda state: setattr(state, "step_index", 1), "step_index"),
        (lambda state: setattr(state, "last_chunk", True), "last_chunk"),
    ],
)
def test_variable_api_rejects_incompatible_states_before_estimator_call(mutator, message):
    estimator = RowEncodingEstimator()
    cfm = CausalConditionalCFM(estimator)
    first = _state("a", cache_length=2, value=1.0)
    second = _state("b", cache_length=2, value=2.0)
    mutator(second)

    with pytest.raises(ValueError, match=message):
        cfm.advance_chunk_step_variable_batch((first, second))
    assert estimator.calls == []


def test_variable_api_rejects_current_chunk_length_mismatch_before_cuda_call():
    estimator = RowEncodingEstimator()
    cfm = CausalConditionalCFM(estimator)
    first = _state("a", cache_length=2, current_length=3)
    second = _state("b", cache_length=2, current_length=4)

    with pytest.raises(ValueError, match="current chunk length"):
        cfm.advance_chunk_step_variable_batch((first, second))
    assert estimator.calls == []


def test_b1_variable_api_delegates_to_frozen_step_path():
    estimator = RowEncodingEstimator()
    cfm = CausalConditionalCFM(estimator)
    state = _state("single", cache_length=2)

    result = cfm.advance_chunk_step_variable_batch((state,))

    assert result == (state,)
    assert estimator.calls[-1]["kind"] == "single"


def test_real_dit_variable_path_passes_row_mask_to_sdpa(monkeypatch):
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def wrapped(q, k, v, *, attn_mask=None, **kwargs):
        calls.append(attn_mask.detach().clone() if attn_mask is not None else None)
        return original(q, k, v, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", wrapped)
    model = DiT(
        in_channels=7,
        out_channels=2,
        depth=1,
        num_heads=1,
        head_dim=2,
        hidden_size=4,
    )
    x = torch.randn(4, 2, 3)
    mu = torch.randn_like(x)
    t = torch.zeros(4)
    spks = torch.randn(4, 1)
    cond = torch.randn_like(x)
    att_cache = torch.zeros(1, 4, 1, 5, 4)

    model.forward_chunk_variable(
        x=x,
        mu=mu,
        t=t,
        spks=spks,
        cond=cond,
        cnn_cache=None,
        att_cache=att_cache,
        attention_cache_lengths=(2, 5, 2, 5),
    )

    assert calls and calls[0] is not None
    assert calls[0].shape == (4, 1, 3, 8)
    assert calls[0][0, 0, :, 3:5].all()
    assert not calls[0][0, 0, :, 5:].any()
    assert calls[0][1, 0, :, 3:].all()


def test_flow_wrapper_forwards_variable_public_api_without_private_access():
    class Decoder:
        def __init__(self):
            self.seen = None

        def advance_chunk_step_variable_batch(self, states):
            self.seen = tuple(states)
            return tuple(states)

    wrapper = object.__new__(CausalMaskedDiffWithXvec)
    wrapper.decoder = Decoder()
    states = (_state("a", cache_length=2), _state("b", cache_length=2, value=2.0))

    assert wrapper.advance_chunk_step_variable_batch(states) == states
    assert wrapper.decoder.seen == states
