from __future__ import annotations

from lychee_fd.runtime.apr.rate_matching_policy import RateMatchingController, RateMatchingPolicyConfig


def test_memory_guard_is_reflected_in_joint_decision():
    controller = RateMatchingController(
        RateMatchingPolicyConfig(model_memory_budget_bytes=100, min_dwell_ms=0)
    )
    controller.observe_model_step({"model_batch_size": 1, "duration_ns": 10, "state_payload_bytes": 100})
    controller.observe_model_step({"model_batch_size": 2, "duration_ns": 10, "state_payload_bytes": 200})
    decision = controller.decide(
        now_ns=1, max_model_batch_size=2, max_acoustic_batch_size=1,
        model_ready_rows=2, acoustic_ready_rows=1, compatible_acoustic_rows=1,
    )
    assert decision.applied_model_cap == 1
    assert decision.memory_guard is True

