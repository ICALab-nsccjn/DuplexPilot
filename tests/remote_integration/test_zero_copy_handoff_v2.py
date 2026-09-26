from __future__ import annotations

import pytest
import torch

from lychee_fd.runtime.apr.elastic_worker_pool import AcousticWorkerPoolV2
from lychee_fd.runtime.apr.worker_handoff import (
    HandoffError,
    LogicalAcousticSessionState,
    WorkerHandoffController,
)


def make_state() -> LogicalAcousticSessionState:
    tensor = torch.arange(16, dtype=torch.float32)
    return LogicalAcousticSessionState(
        request_id="req-zero-copy",
        stream_id="stream-zero-copy",
        generation_id=3,
        sequence_no=4,
        state_version=5,
        flow_state={"x": tensor},
        token2wav_state={"cache": tensor},
        hift_state={"ready": True},
        flush_state={"event_active": True},
    )


def test_zero_copy_transfer_preserves_storage_and_context_identity():
    events: list[dict] = []
    pool = AcousticWorkerPoolV2(worker_count=2, device="cpu", event_sink=events.append)
    source = pool.acquire(target_worker_id=0, request_id="req-zero-copy")
    state = make_state()
    state.lease = source
    pointer = state.flow_state["x"].data_ptr()
    controller = WorkerHandoffController(pool, event_sink=events.append)

    ticket = controller.begin_zero_copy_handoff(
        state=state,
        source_lease=source,
        target_worker_id=1,
        quiesce=lambda _context, _state: {"boundary": "flow_step", "in_flight": 0},
    )
    result = controller.commit_zero_copy_handoff(ticket)

    assert result.zero_copy is True
    assert result.checkpoint is None
    assert state.flow_state["x"].data_ptr() == pointer
    assert state.lease is not None
    assert state.lease.worker_id == 1
    assert result.source_execution_context_id != result.target_execution_context_id
    assert result.copied_state_bytes == 0
    assert any(e.get("event_type") == "APR_ZERO_COPY_HANDOFF_COMMIT" for e in events)
    pool.release(state.lease)
    pool.close()


def test_zero_copy_target_validation_failure_rolls_back_without_copy():
    pool = AcousticWorkerPoolV2(worker_count=2, device="cpu")
    source = pool.acquire(target_worker_id=0, request_id="req-zero-copy")
    state = make_state()
    state.lease = source
    pointer = state.flow_state["x"].data_ptr()
    controller = WorkerHandoffController(pool)
    ticket = controller.begin_zero_copy_handoff(
        state=state,
        source_lease=source,
        target_worker_id=1,
    )

    with pytest.raises(HandoffError, match="rollback|validation"):
        controller.commit_zero_copy_handoff(
            ticket,
            validate=lambda *_args: (_ for _ in ()).throw(RuntimeError("target validation failed")),
        )

    assert state.lease is not None and state.lease.worker_id == 0
    assert pool.lease_for_request("req-zero-copy").worker_id == 0
    assert state.flow_state["x"].data_ptr() == pointer
    pool.release(state.lease)
    pool.close()


def test_zero_copy_cancel_and_stale_fences_leave_source_lease_intact():
    pool = AcousticWorkerPoolV2(worker_count=2, device="cpu")
    source = pool.acquire(target_worker_id=0, request_id="req-zero-copy")
    state = make_state()
    state.lease = source
    controller = WorkerHandoffController(pool)
    ticket = controller.begin_zero_copy_handoff(
        state=state,
        source_lease=source,
        target_worker_id=1,
    )
    state.cancel()
    with pytest.raises(HandoffError, match="cancel|fence"):
        controller.commit_zero_copy_handoff(ticket)
    assert pool.lease_for_request("req-zero-copy").worker_id == 0
    assert state.lease == source
    pool.release(source)
    pool.close()


def test_zero_copy_rejects_busy_target_without_mutating_pool():
    pool = AcousticWorkerPoolV2(worker_count=2, device="cpu")
    source = pool.acquire(target_worker_id=0, request_id="req-source")
    target_owner = pool.acquire(target_worker_id=1, request_id="req-target")
    state = make_state()
    state.request_id = "req-source"
    state.lease = source
    controller = WorkerHandoffController(pool)

    ticket = controller.begin_zero_copy_handoff(
        state=state, source_lease=source, target_worker_id=1
    )
    with pytest.raises(HandoffError, match="unavailable|failed"):
        controller.commit_zero_copy_handoff(ticket)

    assert pool.lease_for_request("req-source") == source
    assert pool.lease_for_request("req-target") == target_owner
    pool.release(source)
    pool.release(target_owner)
    pool.close()


def test_zero_copy_commit_is_one_shot_and_explicit_rollback_transfers_back():
    pool = AcousticWorkerPoolV2(worker_count=2, device="cpu")
    source = pool.acquire(target_worker_id=0, request_id="req-zero-copy")
    state = make_state()
    state.lease = source
    controller = WorkerHandoffController(pool)
    ticket = controller.begin_zero_copy_handoff(
        state=state, source_lease=source, target_worker_id=1
    )
    result = controller.commit_zero_copy_handoff(ticket)
    with pytest.raises(HandoffError, match="closed|already"):
        controller.commit_zero_copy_handoff(ticket)

    assert result.zero_copy
    assert state.lease is not None and state.lease.worker_id == 1
    controller.rollback_zero_copy_handoff(ticket)
    assert state.lease is not None and state.lease.worker_id == 0
    pool.release(state.lease)
    pool.close()


def test_zero_copy_generation_fence_fails_before_pool_transfer():
    pool = AcousticWorkerPoolV2(worker_count=2, device="cpu")
    source = pool.acquire(target_worker_id=0, request_id="req-zero-copy")
    state = make_state()
    state.lease = source
    controller = WorkerHandoffController(pool)
    ticket = controller.begin_zero_copy_handoff(
        state=state, source_lease=source, target_worker_id=1
    )
    state.state_version += 1
    with pytest.raises(HandoffError, match="fence"):
        controller.commit_zero_copy_handoff(ticket)
    assert pool.lease_for_request("req-zero-copy") == source
    pool.release(source)
    pool.close()
