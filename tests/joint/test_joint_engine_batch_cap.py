"""Regression tests for the physical model-engine cap in the joint matrix."""

from __future__ import annotations

from pathlib import Path

from lychee_fd.runtime.apr.joint_execution import get_joint_execution_spec


def test_joint_model_cap_is_also_the_vllm_engine_cap():
    assert get_joint_execution_spec("J11").vllm_max_num_seqs == 1
    assert get_joint_execution_spec("J12").vllm_max_num_seqs == 1
    assert get_joint_execution_spec("J21").vllm_max_num_seqs == 2
    assert get_joint_execution_spec("J22").vllm_max_num_seqs == 2


def test_case_runner_applies_physical_model_cap_not_logical_concurrency():
    runner = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "benchmarks"
        / "run_joint_b22_case.sh"
    ).read_text(encoding="utf-8")
    assert '-e LYCHEEFD_VLLM_MAX_NUM_SEQS="$ROW_CAP"' in runner
    assert '-e LYCHEEFD_VLLM_MAX_NUM_SEQS="$N"' not in runner

