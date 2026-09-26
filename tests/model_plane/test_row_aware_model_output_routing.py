"""Output routing and global-state isolation contract tests."""

from __future__ import annotations

import inspect
import threading

from lychee_fd.runtime.row_aware_model_execution_plane import RowAwareModelExecutionPlane


def test_outputs_are_routed_by_request_id_not_position():
    gate = threading.Event()

    def step(rounds):
        return [
            {"request_id": rounds[1].request_id, "value": "second"},
            {"request_id": rounds[0].request_id, "value": "first"},
        ]

    plane = RowAwareModelExecutionPlane(step, start_gate=gate, max_batch_size=2)
    try:
        plane.register({"request_id": "first"})
        plane.register({"request_id": "second"})
        first = plane.submit_round("first", 0, None)
        second = plane.submit_round("second", 0, None)
        gate.set()
        assert first.result(timeout=2)["value"] == "first"
        assert second.result(timeout=2)["value"] == "second"
    finally:
        plane.close()


def test_plane_has_no_dependency_on_legacy_global_duplex_state():
    source = inspect.getsource(RowAwareModelExecutionPlane)
    assert "LycheeDuplexState" not in source
    assert "lychee_side_state" not in source
