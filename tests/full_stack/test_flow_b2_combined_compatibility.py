from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "Step-Audio2"))

from cosyvoice2.flow.flow_matching import CausalCFMStepState, CausalConditionalCFM
from lychee_fd.runtime.apr.deadline_bounded_flow_scheduler import (
    DeadlineBoundedFlowScheduler,
    FlowBatchCompatibilityPolicy,
    FlowStepItem,
)
from lychee_fd.runtime.apr.flow_batch_runtime import FlowBatchRuntime


def _item(request_id: str, *, step: int, shape: tuple[int, ...]) -> FlowStepItem:
    return FlowStepItem(
        request_id=request_id,
        generation_id=10 if request_id == "a" else 20,
        version=0,
        state=SimpleNamespace(
            request_id=request_id,
            generation_id=10 if request_id == "a" else 20,
            version=0,
            step_index=step,
        ),
        ready_at_ns=0,
        model_identity="flow",
        device="cuda:1",
        dtype="torch.float32",
        step_index=step,
        shape_signature=shape,
        last_chunk=False,
        n_timesteps=10,
    )


def test_combined_policy_relaxes_only_step_and_current_shape():
    policy = FlowBatchCompatibilityPolicy("mixed_step_chunk_padding")
    assert policy.allow_mixed_step is True
    assert policy.allow_chunk_padding is True
    assert policy.key(_item("a", step=1, shape=(1, 80, -1))) == policy.key(
        _item("b", step=4, shape=(1, 80, -1))
    )
    assert policy.key(_item("a", step=1, shape=(1, 80, 20))) != policy.key(
        _item("b", step=4, shape=(1, 80, 24))
    )

    scheduler = DeadlineBoundedFlowScheduler(
        max_batch_size=2,
        max_batch_wait_ms=0.0,
        compatibility_policy="mixed_step_chunk_padding",
    )
    scheduler.submit(_item("a", step=1, shape=(1, 80, -1)))
    scheduler.submit(_item("b", step=4, shape=(1, 80, -1)))
    selected = scheduler.next_batch(now_ns=0)
    assert selected is not None
    assert [item.request_id for item in selected] == ["a", "b"]


class _CombinedBackend:
    variable_length = True
    mixed_step = True
    chunk_padding = True
    mixed_step_max_batch_size = 2

    def __init__(self):
        self.combined_calls: list[tuple[str, ...]] = []

    def advance_step(self, state):
        raise AssertionError("combined batch must not use singleton execution")

    def advance_step_batch(self, states):
        raise AssertionError("combined batch must use its public method")

    def advance_step_mixed_batch(self, states):
        raise AssertionError("combined batch must not use mixed-only method")

    def advance_step_mixed_chunk_padding_batch(self, states):
        self.combined_calls.append(tuple(state.request_id for state in states))
        return tuple(
            SimpleNamespace(**{**vars(state), "step_index": state.step_index + 1})
            for state in states
        )

    def finish(self, state):
        return None


def test_runtime_dispatches_combined_public_backend_method():
    backend = _CombinedBackend()
    runtime = FlowBatchRuntime(
        backend=backend,
        max_batch_size=2,
        max_batch_wait_ms=0.0,
        compatibility_policy="mixed_step_chunk_padding",
    )
    runtime.submit(_item("a", step=1, shape=(1, 80, -1)))
    runtime.submit(_item("b", step=4, shape=(1, 80, -1)))
    result = runtime.run_next(now_ns=0)
    assert result is not None and result.committed
    assert backend.combined_calls == [("a", "b")]


class _CombinedEstimator:
    out_channels = 2

    def __init__(self):
        self.calls: list[dict[str, object]] = []

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
        current_lengths=None,
    ):
        self.calls.append(
            {
                "x": x.detach().clone(),
                "current_lengths": tuple(current_lengths or ()),
                "attention_cache_lengths": tuple(attention_cache_lengths or ()),
                "mask": attention_valid_mask.detach().clone(),
            }
        )
        batch = int(x.shape[0])
        dphi = torch.zeros_like(x)
        cnn = torch.zeros(1, batch, 1, 2, dtype=x.dtype)
        cache_len = 0 if att_cache is None else int(att_cache.shape[3])
        att = torch.zeros(1, batch, 1, cache_len + int(x.shape[-1]), 2, dtype=x.dtype)
        return dphi, cnn, att


def _state(
    request_id: str,
    *,
    step_index: int,
    current_length: int,
    attention_length: int,
    value: float,
) -> CausalCFMStepState:
    t_span = torch.tensor([0.0, 0.2, 0.6, 1.0])
    cnn_slots = tuple(
        torch.full((1, 2, 1, 2), value + slot)
        for slot in range(3)
    )
    att_slots = tuple(
        torch.full((1, 2, 1, attention_length, 2), value + slot)
        for slot in range(3)
    )
    return CausalCFMStepState(
        x=torch.full((1, 2, current_length), value),
        t=t_span[step_index].reshape(1),
        dt=(t_span[step_index + 1] - t_span[step_index]).reshape(()),
        step_index=step_index,
        t_span=t_span,
        mu=torch.full((1, 2, current_length), value + 1),
        speaker=torch.full((1, 4), value + 2),
        condition=torch.full((1, 2, current_length), value + 3),
        input_cnn_cache=cnn_slots,
        input_att_cache=att_slots,
        request_id=request_id,
        generation_id=int(value),
        sequence_no=int(value),
        version=0,
        _cnn_history=cnn_slots,
        _att_history=att_slots,
    )


def test_combined_flow_api_pads_current_lengths_and_preserves_rows():
    estimator = _CombinedEstimator()
    cfm = CausalConditionalCFM(estimator)
    first = _state(
        "first", step_index=0, current_length=3, attention_length=2, value=1.0
    )
    second = _state(
        "second", step_index=1, current_length=5, attention_length=4, value=2.0
    )

    updated = cfm.advance_chunk_step_variable_mixed_chunk_padding_batch(
        (first, second)
    )
    assert [state.step_index for state in updated] == [1, 2]
    assert [state.x.shape[-1] for state in updated] == [3, 5]
    assert [state.completed_att_cache.shape[3] for state in updated] == [5, 9]
    call = estimator.calls[-1]
    assert call["x"].shape[-1] == 5
    assert call["current_lengths"] == (3, 5, 3, 5)
    mask = call["mask"]
    assert not mask[0, :, 3].any()
    assert mask[1, :, 4].all()


def test_combined_flow_rejects_invalid_zero_length_state_before_estimator():
    estimator = _CombinedEstimator()
    cfm = CausalConditionalCFM(estimator)
    first = _state(
        "first", step_index=0, current_length=3, attention_length=2, value=1.0
    )
    second = _state(
        "second", step_index=1, current_length=5, attention_length=4, value=2.0
    )
    second.x = torch.zeros(1, 2, 0)
    second.mu = torch.zeros(1, 2, 0)
    second.condition = torch.zeros(1, 2, 0)
    with pytest.raises(ValueError, match="current chunk length"):
        cfm.advance_chunk_step_variable_mixed_chunk_padding_batch((first, second))
    assert estimator.calls == []
