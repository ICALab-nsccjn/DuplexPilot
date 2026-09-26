"""Server-side fail-closed validation for joint benchmark payloads."""

from __future__ import annotations

import pytest

from lychee_fd.runtime.apr.online_router import resolve_paper_system_payload
from lychee_fd.runtime.apr.paper_systems import get_system_spec


def test_server_accepts_an_unchanged_j22_contract():
    spec = get_system_spec("rsv_dsv_apr_joint_j22")
    payload = {
        "paper_system_id": spec.system_id,
        "runtime_mode": spec.model_runtime_mode,
        "acoustic_mode": spec.acoustic_mode,
        "max_flow_batch_size": spec.max_flow_batch_size,
        "joint_id": "J22",
        "model_execution_mode": "row_aware_cap2",
        "acoustic_execution_mode": "mixed_chunk_padding_b2",
        "max_model_batch_size": 2,
        "max_acoustic_batch_size": 2,
    }
    assert resolve_paper_system_payload(
        payload, runtime_mode=spec.model_runtime_mode
    ) == spec


@pytest.mark.parametrize(
    "field,value",
    [
        ("joint_id", "J12"),
        ("model_execution_mode", "legacy_serialized"),
        ("max_model_batch_size", 1),
        ("max_acoustic_batch_size", 1),
    ],
)
def test_server_rejects_joint_dimension_or_identity_tampering(field, value):
    spec = get_system_spec("rsv_dsv_apr_joint_j22")
    payload = {
        "paper_system_id": spec.system_id,
        "joint_id": "J22",
        "model_execution_mode": "row_aware_cap2",
        "acoustic_execution_mode": "mixed_chunk_padding_b2",
        "max_model_batch_size": 2,
        "max_acoustic_batch_size": 2,
    }
    payload[field] = value
    with pytest.raises(ValueError):
        resolve_paper_system_payload(payload, runtime_mode=spec.model_runtime_mode)


def test_server_rejects_joint_fields_on_non_joint_system():
    spec = get_system_spec("rsv_dsv_apr_step_b1")
    payload = {
        "paper_system_id": spec.system_id,
        "joint_id": "J21",
        "model_execution_mode": "row_aware_cap2",
        "acoustic_execution_mode": "apr_step_b1",
        "max_model_batch_size": 2,
        "max_acoustic_batch_size": 1,
    }
    with pytest.raises(ValueError):
        resolve_paper_system_payload(payload, runtime_mode=spec.model_runtime_mode)
