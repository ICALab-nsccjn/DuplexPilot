from __future__ import annotations

from lychee_fd.runtime.apr.rate_matching_policy import RateMatchingController, RateMatchingPolicyConfig


def test_reset_clears_observed_width_and_decision():
    controller = RateMatchingController(
        RateMatchingPolicyConfig(model_memory_budget_bytes=1_000_000, min_dwell_ms=0)
    )
    controller.observe_model_step({"model_batch_size": 2, "duration_ns": 10})
    controller.decide(
        now_ns=1, max_model_batch_size=2, max_acoustic_batch_size=1,
        model_ready_rows=2, acoustic_ready_rows=1, compatible_acoustic_rows=1,
    )
    controller.reset()
    assert controller.last_decision is None
    assert controller.stats["observed_model_width_samples"] == 0

