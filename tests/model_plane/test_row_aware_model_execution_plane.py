"""Contract tests for the benchmark-only row-aware model execution plane.

These tests deliberately use a fake engine.  They verify request ownership,
batch formation, routing, and generation fencing without loading a model.
"""

from __future__ import annotations

from concurrent.futures import Future
import threading
import time

import pytest

from lychee_fd.runtime.row_aware_model_execution_plane import (
    ModelExecutionPlaneBackpressure,
    ModelExecutionPlaneClosed,
    ModelExecutionPlaneError,
    ModelExecutionPlaneStale,
    RowAwareModelExecutionPlane,
)


def _await(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    assert predicate(), "condition did not become true before timeout"


def _plane(step_fn, *, start_gate: threading.Event | None = None, **kwargs):
    # ``start_gate`` is a deterministic unit-test hook.  Production callers
    # leave it unset; it lets a test queue two real demands before the single
    # driver takes its first scheduling turn.
    return RowAwareModelExecutionPlane(step_fn, start_gate=start_gate, **kwargs)


def test_two_requests_form_one_natural_driver_batch_and_route_outputs():
    started = threading.Event()
    calls: list[tuple[str, ...]] = []

    def step(rounds):
        calls.append(tuple(item.request_id for item in rounds))
        started.set()
        # Return in reverse order to prove request-owned routing.
        return [
            {"request_id": item.request_id, "generation_id": item.generation_id}
            for item in reversed(rounds)
        ]

    gate = threading.Event()
    plane = _plane(step, start_gate=gate, max_batch_size=2)
    try:
        plane.register({"request_id": "r0", "generation_id": 4})
        plane.register({"request_id": "r1", "generation_id": 9})
        f0 = plane.submit_round("r0", 4, {"token": 0})
        f1 = plane.submit_round("r1", 9, {"token": 1})
        gate.set()
        assert f0.result(timeout=2)["request_id"] == "r0"
        assert f1.result(timeout=2)["request_id"] == "r1"
        assert calls == [("r0", "r1")]
        assert started.is_set()
    finally:
        plane.close()


def test_single_request_is_supported_and_driver_is_single_threaded():
    driver_threads: list[int] = []

    def step(rounds):
        driver_threads.append(threading.get_ident())
        return [{"request_id": rounds[0].request_id}]

    plane = _plane(step)
    try:
        plane.register({"request_id": "r0", "generation_id": 0})
        result = plane.submit_round("r0", 0, None).result(timeout=2)
        assert result["request_id"] == "r0"
        assert len(set(driver_threads)) == 1
        assert driver_threads[0] != threading.get_ident()
    finally:
        plane.close()


def test_one_request_cannot_have_two_pending_or_inflight_rounds():
    entered = threading.Event()
    release = threading.Event()

    def step(rounds):
        entered.set()
        release.wait(timeout=2)
        return [{"request_id": rounds[0].request_id}]

    plane = _plane(step)
    try:
        plane.register({"request_id": "r0"})
        first = plane.submit_round("r0", 0, None)
        assert entered.wait(timeout=2)
        second = plane.submit_round("r0", 0, None)
        with pytest.raises(ModelExecutionPlaneError):
            second.result(timeout=1)
        release.set()
        assert first.result(timeout=2)["request_id"] == "r0"
    finally:
        release.set()
        plane.close()


def test_generation_fence_drops_cancelled_inflight_output():
    entered = threading.Event()
    release = threading.Event()
    cancel_calls: list[tuple[str, int]] = []

    def step(rounds):
        entered.set()
        release.wait(timeout=2)
        return [{"request_id": rounds[0].request_id, "generation_id": rounds[0].generation_id}]

    plane = _plane(
        step,
        cancel_fn=lambda request_id, generation_id: cancel_calls.append(
            (request_id, generation_id)
        ),
        output_generation_id=lambda output: output["generation_id"],
    )
    try:
        plane.register({"request_id": "r0", "generation_id": 3})
        future = plane.submit_round("r0", 3, None)
        assert entered.wait(timeout=2)
        plane.cancel("r0", 3)
        release.set()
        with pytest.raises(ModelExecutionPlaneStale):
            future.result(timeout=2)
        _await(lambda: ("r0", 3) in cancel_calls)
    finally:
        release.set()
        plane.close()


def test_reset_invalidates_old_generation_and_allows_new_round():
    calls: list[tuple[str, int]] = []

    def step(rounds):
        calls.append((rounds[0].request_id, rounds[0].generation_id))
        return [{"request_id": rounds[0].request_id, "generation_id": rounds[0].generation_id}]

    plane = _plane(step, output_generation_id=lambda output: output["generation_id"])
    try:
        plane.register({"request_id": "r0", "generation_id": 1})
        assert plane.submit_round("r0", 1, None).result(timeout=2)["generation_id"] == 1
        handle = plane.reset("r0", 2)
        assert handle.generation_id == 2
        with pytest.raises(ModelExecutionPlaneStale):
            plane.submit_round("r0", 1, None).result(timeout=1)
        assert plane.submit_round("r0", 2, None).result(timeout=2)["generation_id"] == 2
        assert calls == [("r0", 1), ("r0", 2)]
    finally:
        plane.close()


def test_cancel_before_dispatch_removes_pending_demand_without_calling_engine():
    gate = threading.Event()
    calls: list[tuple[str, ...]] = []

    def step(rounds):
        calls.append(tuple(item.request_id for item in rounds))
        return [{"request_id": item.request_id} for item in rounds]

    plane = _plane(step, start_gate=gate)
    try:
        plane.register({"request_id": "r0", "generation_id": 0})
        future = plane.submit_round("r0", 0, None)
        plane.cancel("r0", 0)
        with pytest.raises(ModelExecutionPlaneStale):
            future.result(timeout=1)
        gate.set()
        time.sleep(0.05)
        assert calls == []
    finally:
        plane.close()


def test_bounded_queue_backpressure_and_close_are_fail_closed():
    gate = threading.Event()

    def step(rounds):
        return [{"request_id": item.request_id} for item in rounds]

    plane = _plane(step, start_gate=gate, max_pending_rounds=1)
    try:
        plane.register({"request_id": "r0"})
        plane.register({"request_id": "r1"})
        first = plane.submit_round("r0", 0, None)
        second = plane.submit_round("r1", 0, None)
        with pytest.raises(ModelExecutionPlaneBackpressure):
            second.result(timeout=1)
        gate.set()
        assert first.result(timeout=2)["request_id"] == "r0"
        plane.close()
        with pytest.raises(ModelExecutionPlaneClosed):
            plane.submit_round("r0", 0, None).result(timeout=1)
    finally:
        gate.set()
        plane.close()


def test_backend_exception_is_reported_and_does_not_leak_pending_state():
    gate = threading.Event()

    def step(rounds):
        raise RuntimeError("fake engine failure")

    plane = _plane(step, start_gate=gate)
    try:
        plane.register({"request_id": "r0"})
        future = plane.submit_round("r0", 0, None)
        gate.set()
        with pytest.raises(RuntimeError, match="fake engine failure"):
            future.result(timeout=2)
        assert isinstance(plane.driver_error, RuntimeError)
        assert plane.pending_ids() == ()
    finally:
        plane.close()


def test_control_commands_run_on_same_driver_and_preserve_queue_order():
    calls: list[tuple[str, int]] = []
    driver_id: list[int] = []

    def step(rounds):
        driver_id.append(threading.get_ident())
        calls.append(("step", len(rounds)))
        return [{"request_id": item.request_id} for item in rounds]

    plane = _plane(step)
    try:
        plane.register({"request_id": "r0"})
        command = plane.call_on_driver(
            lambda: calls.append(("command", threading.get_ident())) or "ok",
            description="add_request",
        )
        assert command.result(timeout=2) == "ok"
        result = plane.submit_round("r0", 0, None).result(timeout=2)
        assert result["request_id"] == "r0"
        assert calls[0][0] == "command"
        assert calls[1] == ("step", 1)
        assert calls[0][1] == driver_id[0]
    finally:
        plane.close()


def test_control_command_exception_closes_plane_and_rejects_future_work():
    plane = _plane(lambda rounds: [{"request_id": item.request_id} for item in rounds])
    try:
        plane.register({"request_id": "r0", "generation_id": 0})
        command = plane.call_on_driver(
            lambda: (_ for _ in ()).throw(RuntimeError("control failure")),
            description="abort_request",
        )
        with pytest.raises(RuntimeError, match="control failure"):
            command.result(timeout=2)

        pending = plane.submit_round("r0", 0, None)
        with pytest.raises(ModelExecutionPlaneClosed):
            pending.result(timeout=2)
        assert isinstance(plane.driver_error, RuntimeError)
    finally:
        plane.close()
