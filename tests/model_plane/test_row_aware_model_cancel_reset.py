"""Lifecycle tests for cancellation, finish, and output fencing."""

from __future__ import annotations

from concurrent.futures import Future
import threading

import pytest

from lychee_fd.runtime.row_aware_model_execution_plane import (
    ModelExecutionPlaneStale,
    RowAwareModelExecutionPlane,
)


def test_finish_rejects_pending_round_and_removes_inactive_request():
    gate = threading.Event()

    def step(rounds):
        return [{"request_id": item.request_id} for item in rounds]

    plane = RowAwareModelExecutionPlane(step, start_gate=gate)
    try:
        plane.register({"request_id": "r0", "generation_id": 0})
        future = plane.submit_round("r0", 0, None)
        plane.finish("r0", 0)
        with pytest.raises(ModelExecutionPlaneStale):
            future.result(timeout=1)
        gate.set()
        assert plane.pending_ids() == ()
        with pytest.raises(ModelExecutionPlaneStale):
            plane.submit_round("r0", 0, None).result(timeout=1)
    finally:
        gate.set()
        plane.close()


def test_stale_output_generation_is_dropped_even_if_engine_returns_it():
    gate = threading.Event()

    def step(rounds):
        return [{"request_id": rounds[0].request_id, "generation_id": 99}]

    plane = RowAwareModelExecutionPlane(
        step,
        start_gate=gate,
        output_generation_id=lambda output: output["generation_id"],
    )
    try:
        plane.register({"request_id": "r0", "generation_id": 3})
        future = plane.submit_round("r0", 3, None)
        gate.set()
        with pytest.raises(ModelExecutionPlaneStale):
            future.result(timeout=2)
    finally:
        plane.close()
