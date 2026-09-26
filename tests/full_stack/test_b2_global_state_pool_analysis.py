from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmarks.analyze_b2_global_state_pool import (
    analyze_records,
    detect_worker_provenance,
    inspect_runtime_topology,
)


def _opportunity(*, request_ids, worker_ids=None, compatible_count=1, potential_b2=False):
    value = {
        "event": "FLOW_BATCH_OPPORTUNITY",
        "request_ids": list(request_ids),
        "compatible_request_ids": list(request_ids[:compatible_count]),
        "compatible_count": compatible_count,
        "potential_b2": potential_b2,
        "potential_b4": False,
        "step_index_distribution": {"0": len(request_ids)},
    }
    if worker_ids is not None:
        value["worker_ids"] = list(worker_ids)
    return value


def test_worker_count_is_not_worker_provenance():
    records = [{"event": "RUN_METADATA", "worker_count": 2}]

    result = detect_worker_provenance(records)

    assert result.present is False
    assert result.keys == ()
    assert result.record_count == 0


def test_explicit_worker_ids_enable_cross_worker_pair_count():
    records = [
        _opportunity(
            request_ids=("a", "b", "c"),
            worker_ids=(0, 1, 0),
            compatible_count=3,
            potential_b2=True,
        ),
        {
            "event": "FLOW_BATCH_FORMED",
            "batch_size": 2,
            "request_ids": ["a", "b"],
            "worker_ids": [0, 1],
        },
    ]

    result = analyze_records(records, topology="partitioned")

    assert result.actual_b2 == 1
    assert result.global_pair_count == 3
    assert result.same_worker_pair_count == 1
    assert result.cross_worker_pair_count == 2


def test_current_global_scheduler_is_noop_for_global_pool_candidate():
    records = [
        _opportunity(request_ids=("a",), compatible_count=1),
        {"event": "FLOW_BATCH_SUBMIT", "batch_size": 1},
        {"event": "FLOW_BATCH_FORMED", "batch_size": 2, "request_ids": ["a", "b"]},
    ]

    result = analyze_records(records, topology="global")

    assert result.status == "CROSS_WORKER_POOLING_NOT_APPLICABLE"
    assert result.predicted_e2e_speedup == pytest.approx(1.0)
    assert result.incremental_b2_work_fraction == pytest.approx(0.0)
    assert result.cross_worker_pair_count is None


def test_b2_work_fraction_includes_singleton_fallbacks():
    records = [
        {"event": "FLOW_BATCH_FALLBACK", "batch_size": 1},
        {"event": "FLOW_BATCH_FORMED", "batch_size": 2, "request_ids": ["a", "b"]},
    ]

    result = analyze_records(records, topology="global")

    assert result.actual_b1 == 1
    assert result.actual_b2 == 1
    assert result.total_executed_logical_work == 3
    assert result.actual_b2_work_fraction == pytest.approx(2 / 3)


def test_topology_inspection_detects_single_global_online_runtime(tmp_path: Path):
    (tmp_path / "online_step_coordinator.py").write_text(
        "class OnlineFlowStepCoordinator:\n"
        "    self._ingress = deque()\n"
        "    self._runtime = FlowBatchRuntime()\n"
        "    self._runtime.scheduler = scheduler\n",
        encoding="utf-8",
    )
    (tmp_path / "flow_batch_runtime.py").write_text(
        "class FlowBatchRuntime:\n"
        "    def run_next(self): ...\n",
        encoding="utf-8",
    )
    (tmp_path / "deadline_bounded_flow_scheduler.py").write_text(
        "class DeadlineBoundedFlowScheduler:\n"
        "    self._pending = []\n",
        encoding="utf-8",
    )
    (tmp_path / "paper_backend_selector.py").write_text(
        "return APRFlowBatchAcousticLane(**common)\n",
        encoding="utf-8",
    )
    (tmp_path / "flow_batch_acoustic_lane.py").write_text(
        "self.worker_for = {}\nself._worker_cursor = 0\n",
        encoding="utf-8",
    )

    result = inspect_runtime_topology(tmp_path)

    assert result.online_global_queue is True
    assert result.physical_worker_partition is False
    assert result.worker_label_is_metadata is True
    assert result.online_router_shared_coordinator is False
    assert result.flow_step_item_has_worker_id is False
