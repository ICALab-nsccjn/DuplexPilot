from __future__ import annotations

from types import SimpleNamespace

from lychee_fd.runtime.apr.deadline_bounded_flow_scheduler import (
    FlowBatchCompatibilityPolicy,
    FlowStepItem,
)


def _item(
    request_id: str,
    *,
    shape: tuple[int, ...],
    last_chunk: bool = False,
    n_timesteps: int = 10,
) -> FlowStepItem:
    state = SimpleNamespace(
        request_id=request_id,
        generation_id=1,
        version=0,
        step_index=0,
    )
    return FlowStepItem(
        request_id=request_id,
        generation_id=1,
        version=0,
        state=state,
        ready_at_ns=0,
        model_identity="flow",
        device="cuda:1",
        dtype="torch.float32",
        step_index=0,
        shape_signature=shape,
        last_chunk=last_chunk,
        n_timesteps=n_timesteps,
    )


def test_combined_policy_reports_shape_and_tail_fields_without_shift():
    policy = FlowBatchCompatibilityPolicy("mixed_step_chunk_padding")
    left = _item("a", shape=(1, 80, -1, 128))
    right = _item("b", shape=(1, 96, -1, 128), last_chunk=True, n_timesteps=12)

    comparisons = policy.comparison_fields(policy.key(left), policy.key(right))
    by_name = {name: (before, after) for name, before, after in comparisons}

    assert by_name["shape_signature"] == (left.shape_signature, right.shape_signature)
    assert by_name["last_chunk"] == (left.last_chunk, right.last_chunk)
    assert by_name["n_timesteps"] == (left.n_timesteps, right.n_timesteps)
