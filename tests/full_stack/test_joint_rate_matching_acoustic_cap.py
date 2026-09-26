from __future__ import annotations

from lychee_fd.runtime.apr.rate_matching_policy import RateMatchingController, RateMatchingPolicyConfig


def test_acoustic_two_requires_compatibility_and_runtime_limit():
    controller = RateMatchingController(
        RateMatchingPolicyConfig(model_memory_budget_bytes=1_000_000, min_dwell_ms=0)
    )
    one = controller.decide(
        now_ns=1, max_model_batch_size=4, max_acoustic_batch_size=2,
        model_ready_rows=2, acoustic_ready_rows=2, compatible_acoustic_rows=1,
    )
    assert one.applied_acoustic_cap == 1
    two = controller.decide(
        now_ns=2, max_model_batch_size=4, max_acoustic_batch_size=2,
        model_ready_rows=2, acoustic_ready_rows=2, compatible_acoustic_rows=2,
    )
    assert two.applied_acoustic_cap in (1, 2)
    limited = controller.decide(
        now_ns=3, max_model_batch_size=4, max_acoustic_batch_size=1,
        model_ready_rows=2, acoustic_ready_rows=2, compatible_acoustic_rows=2,
    )
    assert limited.applied_acoustic_cap == 1

