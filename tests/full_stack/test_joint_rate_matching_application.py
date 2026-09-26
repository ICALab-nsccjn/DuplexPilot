from __future__ import annotations

from lychee_fd.runtime.apr.rate_matching_policy import (
    JointRateMatchingDecision,
    RateMatchingController,
    RateMatchingPolicyConfig,
)


def _controller():
    return RateMatchingController(
        RateMatchingPolicyConfig(model_memory_budget_bytes=1_000_000, min_dwell_ms=0)
    )


def test_joint_decision_applies_runtime_caps_and_exposes_all_widths():
    controller = _controller()
    decision = controller.decide(
        now_ns=1,
        max_model_batch_size=4,
        max_acoustic_batch_size=1,
        model_ready_rows=4,
        acoustic_ready_rows=2,
        compatible_acoustic_rows=2,
    )
    assert isinstance(decision, JointRateMatchingDecision)
    assert decision.requested_model_cap <= 4
    assert decision.requested_acoustic_cap == 1
    assert decision.applied_model_cap <= 4
    assert decision.applied_acoustic_cap == 1
    assert decision.eligible_model_rows == 4
    assert decision.eligible_acoustic_rows == 2
    assert decision.compatible_acoustic_rows == 2


def test_b1_selector_never_reports_acoustic_two():
    controller = _controller()
    assert controller.select_acoustic_cap(
        ("a", "b"), max_batch_size=1, compatible_count=2
    ) == 1
    decision = controller.last_decision
    assert decision is not None
    assert decision.applied_acoustic_cap == 1
    assert decision.requested_acoustic_cap == 1

