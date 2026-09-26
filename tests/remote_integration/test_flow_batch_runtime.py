from __future__ import annotations

from dataclasses import dataclass, replace

from lychee_fd.runtime.apr.flow_batch_runtime import (
    FlowBatchRuntime,
    FlowBatchRuntimeError,
    FlowStepItem,
)


@dataclass(frozen=True)
class State:
    request_id: str
    generation_id: int = 1
    version: int = 0
    step_index: int = 0


class FakeFlowBackend:
    def __init__(self):
        self.single_calls = 0
        self.batch_calls: list[int] = []
        self.finished: list[str] = []

    def advance_step(self, state):
        self.single_calls += 1
        return replace(state, step_index=state.step_index + 1)

    def advance_step_batch(self, states):
        self.batch_calls.append(len(states))
        return tuple(replace(state, step_index=state.step_index + 1) for state in states)

    def finish(self, state):
        self.finished.append(state.request_id)
        return f"mel:{state.request_id}:{state.step_index}"


def make_item(request_id: str, *, ready_at_ns: int, version: int = 0) -> FlowStepItem:
    return FlowStepItem(
        request_id=request_id,
        generation_id=1,
        version=version,
        state=State(request_id, version=version),
        ready_at_ns=ready_at_ns,
        model_identity="flow-v1",
        device="cuda:0",
        dtype="float32",
        step_index=0,
        shape_signature=(1, 80, 26),
        last_chunk=False,
        n_timesteps=10,
    )


def test_runtime_uses_public_batch_step_and_commits_identity_preserving_states():
    events = []
    backend = FakeFlowBackend()
    runtime = FlowBatchRuntime(
        backend=backend,
        max_batch_size=2,
        max_batch_wait_ms=2.0,
        event_sink=events.append,
    )
    runtime.submit(make_item("a", ready_at_ns=100))
    runtime.submit(make_item("b", ready_at_ns=110))

    result = runtime.run_next(now_ns=120)

    assert result is not None
    assert result.committed
    assert backend.batch_calls == [2]
    assert backend.single_calls == 0
    assert runtime.committed_state("a").step_index == 1
    assert runtime.committed_state("b").step_index == 1
    assert [event["event"] for event in events if event["event"].startswith("FLOW_BATCH")] == [
        "FLOW_BATCH_FORMED",
        "FLOW_BATCH_SUBMIT",
        "FLOW_BATCH_COMPLETE",
    ]


def test_runtime_emits_wall_and_cuda_timing_when_enabled():
    events = []
    runtime = FlowBatchRuntime(
        backend=FakeFlowBackend(),
        max_batch_wait_ms=0.0,
        event_sink=events.append,
        timing_enabled=True,
    )
    runtime.submit(make_item("a", ready_at_ns=100))

    result = runtime.run_next(now_ns=100)

    assert result is not None and result.committed
    timing = [event for event in events if event["event"] == "FLOW_BATCH_TIMING"]
    assert len(timing) == 1
    assert timing[0]["batch_size"] == 1
    assert timing[0]["wall_time_ms"] >= 0.0
    assert timing[0]["cuda_time_ms"] >= 0.0


def test_runtime_falls_back_to_b1_after_bounded_wait():
    backend = FakeFlowBackend()
    runtime = FlowBatchRuntime(
        backend=backend,
        max_batch_size=2,
        max_batch_wait_ms=2.0,
    )
    runtime.submit(make_item("a", ready_at_ns=100))

    assert runtime.run_next(now_ns=100 + 1_999_999) is None
    result = runtime.run_next(now_ns=100 + 2_000_000)

    assert result is not None and result.committed
    assert backend.single_calls == 1
    assert backend.batch_calls == []


def test_cancel_during_execution_suppresses_stale_commit_and_emits_cancelled():
    events = []
    backend = FakeFlowBackend()
    runtime = FlowBatchRuntime(
        backend=backend,
        max_batch_size=2,
        max_batch_wait_ms=0.0,
        event_sink=events.append,
    )
    runtime.submit(make_item("a", ready_at_ns=100))
    runtime.cancel("a", generation_id=1)

    assert runtime.run_next(now_ns=100) is None
    assert runtime.committed_state("a") is None
    assert any(event["event"] == "FLOW_BATCH_CANCELLED" for event in events)


def test_version_change_after_submit_rejects_stale_commit():
    events = []
    runtime = FlowBatchRuntime(
        backend=FakeFlowBackend(),
        max_batch_size=2,
        max_batch_wait_ms=0.0,
        event_sink=events.append,
    )
    runtime.submit(make_item("a", ready_at_ns=100, version=0))
    runtime.update_version("a", generation_id=1, version=1)

    result = runtime.run_next(now_ns=100)

    assert result is not None
    assert not result.committed
    assert runtime.committed_state("a") is None
    assert any(event["event"] == "FLOW_BATCH_CANCELLED" for event in events)


def test_runtime_finish_uses_backend_finalize_without_private_state_access():
    backend = FakeFlowBackend()
    runtime = FlowBatchRuntime(backend=backend, max_batch_wait_ms=0.0)
    runtime.submit(make_item("a", ready_at_ns=100))
    runtime.run_next(now_ns=100)

    assert runtime.finish("a") == "mel:a:1"
    assert backend.finished == ["a"]


def test_runtime_rejects_returned_identity_mismatch():
    class BadBackend(FakeFlowBackend):
        def advance_step(self, state):
            return replace(state, request_id="other", step_index=1)

    runtime = FlowBatchRuntime(backend=BadBackend(), max_batch_wait_ms=0.0)
    runtime.submit(make_item("a", ready_at_ns=100))

    try:
        runtime.run_next(now_ns=100)
    except FlowBatchRuntimeError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("identity mismatch must fail closed")
