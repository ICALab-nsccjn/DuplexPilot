from __future__ import annotations

from types import SimpleNamespace

from lychee_fd.runtime.apr.deadline_bounded_flow_scheduler import (
    DeadlineBoundedFlowScheduler,
    FlowStepItem,
)


def _item(request_id: str, shape: tuple[int, ...]) -> FlowStepItem:
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
        last_chunk=False,
        n_timesteps=10,
    )


def test_chunk_padding_keeps_non_padable_shape_dimensions_compatible():
    scheduler = DeadlineBoundedFlowScheduler(
        max_batch_size=2,
        max_batch_wait_ms=0.0,
        compatibility_policy="chunk_padding",
    )
    scheduler.submit(_item("a", (1, 80, -1, 128)))
    scheduler.submit(_item("b", (1, 96, -1, 128)))

    selected = scheduler.next_batch(now_ns=0)

    assert selected is not None
    assert len(selected) == 1
    assert selected[0].request_id == "a"


def test_chunk_padding_requires_wildcard_marker_on_each_item():
    scheduler = DeadlineBoundedFlowScheduler(
        max_batch_size=2,
        max_batch_wait_ms=0.0,
        compatibility_policy="chunk_padding",
    )
    scheduler.submit(_item("a", (1, 80, -1, 128)))
    # A fixed value in a pad-able position is intentionally not treated as a
    # wildcard.  Producers must mark that position on every item in the
    # opt-in policy; otherwise the scheduler fails closed.
    scheduler.submit(_item("b", (1, 80, 24, 128)))

    selected = scheduler.next_batch(now_ns=0)

    assert selected is not None
    assert selected is not None
    assert len(selected) == 1


def test_chunk_padding_groups_explicitly_wildcarded_shapes():
    scheduler = DeadlineBoundedFlowScheduler(
        max_batch_size=2,
        max_batch_wait_ms=0.0,
        compatibility_policy="chunk_padding",
    )
    scheduler.submit(_item("a", (1, 80, -1, 128)))
    scheduler.submit(_item("b", (1, 80, -1, 128)))

    selected = scheduler.next_batch(now_ns=0)

    assert selected is not None
    assert [item.request_id for item in selected] == ["a", "b"]
