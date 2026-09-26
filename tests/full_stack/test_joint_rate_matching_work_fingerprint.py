from __future__ import annotations

from lychee_fd.runtime.apr.rate_matching_policy import RateMatchingController, RateMatchingPolicyConfig


def test_work_fingerprint_observation_does_not_change_width_decision():
    controller = RateMatchingController(
        RateMatchingPolicyConfig(model_memory_budget_bytes=1_000_000, min_dwell_ms=0)
    )
    before = controller.decide(
        now_ns=1, max_model_batch_size=4, max_acoustic_batch_size=1,
        model_ready_rows=2, acoustic_ready_rows=1, compatible_acoustic_rows=1,
    )
    controller.observe_work_fingerprint({"token_hash": "abc", "chunk_count": 3})
    after = controller.decide(
        now_ns=2, max_model_batch_size=4, max_acoustic_batch_size=1,
        model_ready_rows=2, acoustic_ready_rows=1, compatible_acoustic_rows=1,
    )
    assert after.applied_acoustic_cap == 1
    assert after.applied_model_cap == before.applied_model_cap

