from __future__ import annotations

from lychee_fd.runtime.apr.rate_matching_policy import RateMatchingController, RateMatchingPolicyConfig


def test_joint_decision_preserves_identity_metadata_in_event():
    events = []
    controller = RateMatchingController(
        RateMatchingPolicyConfig(model_memory_budget_bytes=1_000_000, min_dwell_ms=0),
        event_sink=events.append,
    )
    controller.decide(
        now_ns=10, max_model_batch_size=2, max_acoustic_batch_size=2,
        model_ready_rows=2, acoustic_ready_rows=2, compatible_acoustic_rows=2,
    )
    event = next(item for item in events if item["event"] == "RATE_MATCHING_DECISION")
    for key in ("requested_model_cap", "requested_acoustic_cap", "applied_model_cap",
                "applied_acoustic_cap", "eligible_model_rows", "eligible_acoustic_rows",
                "compatible_acoustic_rows", "observed_model_width", "observed_acoustic_width"):
        assert key in event

