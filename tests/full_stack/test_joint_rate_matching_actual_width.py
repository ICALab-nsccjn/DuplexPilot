from __future__ import annotations

from lychee_fd.runtime.apr.rate_matching_policy import RateMatchingController, RateMatchingPolicyConfig


def test_observed_width_is_recorded_separately_from_applied_cap():
    controller = RateMatchingController(
        RateMatchingPolicyConfig(model_memory_budget_bytes=1_000_000, min_dwell_ms=0)
    )
    controller.decide(
        now_ns=1, max_model_batch_size=8, max_acoustic_batch_size=1,
        model_ready_rows=2, acoustic_ready_rows=1, compatible_acoustic_rows=1,
    )
    controller.observe_model_step({"model_batch_size": 1, "duration_ns": 10})
    controller.observe_acoustic_step({"batch_size": 1, "duration_ns": 10})
    assert controller.last_decision.observed_model_width == 1
    assert controller.last_decision.applied_model_cap >= 1
    assert controller.last_decision.observed_model_width <= controller.last_decision.applied_model_cap

