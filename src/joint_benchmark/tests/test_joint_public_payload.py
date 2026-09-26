"""The public API payload must carry both independent batch dimensions."""

from __future__ import annotations

from lychee_fd.runtime.apr.paper_systems import get_system_spec
from tools.benchmarks.realtime_public_client import build_start_payload


def test_joint_payload_exposes_model_and_acoustic_dimensions():
    payload = build_start_payload(
        get_system_spec("rsv_dsv_apr_joint_j22"), run_id="joint-test"
    )
    assert payload["joint_id"] == "J22"
    assert payload["model_execution_mode"] == "row_aware_cap2"
    assert payload["max_model_batch_size"] == 2
    assert payload["max_acoustic_batch_size"] == 2

