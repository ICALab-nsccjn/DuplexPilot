import pytest

from profiling.model_execution_plane.oracle import (
    ServiceCurve,
    analyze_model_trace,
    compare_work_fingerprint,
    derive_feedback_delays,
    normalize_model_trace,
    simulate_central_driver,
)


def _new_trace(*, two_rows=False):
    request_ids = ["r0", "r1"] if two_rows else ["r0"]
    return [
        {
            "schema": "lychee-model-execution-v1",
            "event_type": "MODEL_ENGINE_BATCH_FORMED",
            "timestamp_monotonic_ns": 100,
            "request_ids": request_ids,
            "model_batch_size": len(request_ids),
            "row_aware": True,
        },
        {
            "schema": "lychee-model-execution-v1",
            "event_type": "MODEL_ENGINE_STEP_END",
            "timestamp_monotonic_ns": 200,
            "engine_step": 0,
            "requested_request_id": request_ids[0],
            "output_request_ids": request_ids,
            "model_batch_size": len(request_ids),
            "duration_ns": 80,
        },
    ]


def test_normalize_new_schema_preserves_rows_and_duration():
    observations = normalize_model_trace(_new_trace(two_rows=True))

    assert len(observations) == 1
    assert observations[0].request_ids == ("r0", "r1")
    assert observations[0].batch_size == 2
    assert observations[0].duration_ns == 80
    assert observations[0].start_ns == 120


def test_normalize_legacy_model_execution_event():
    observations = normalize_model_trace(
        [
            {
                "event": "MODEL_EXECUTION",
                "timestamp_monotonic_ns": 1_000,
                "start_ns": 900,
                "end_ns": 1_140,
                "physical_batch_size": 2,
                "request_ids": ["r0", "r1"],
            }
        ]
    )

    assert len(observations) == 1
    assert observations[0].source_schema == "legacy-model-execution"
    assert observations[0].batch_size == 2
    assert observations[0].duration_ns == 240


def test_central_driver_pairs_only_distinct_ready_requests():
    observations = normalize_model_trace(
        [
            {
                "event": "MODEL_EXECUTION",
                "timestamp_monotonic_ns": 100,
                "start_ns": 0,
                "end_ns": 100,
                "physical_batch_size": 1,
                "request_ids": ["r0"],
            },
            {
                "event": "MODEL_EXECUTION",
                "timestamp_monotonic_ns": 200,
                "start_ns": 100,
                "end_ns": 200,
                "physical_batch_size": 1,
                "request_ids": ["r1"],
            },
            {
                "event": "MODEL_EXECUTION",
                "timestamp_monotonic_ns": 300,
                "start_ns": 200,
                "end_ns": 300,
                "physical_batch_size": 1,
                "request_ids": ["r0"],
            },
        ]
    )

    result = simulate_central_driver(
        observations,
        ServiceCurve(b1_ns=100, b2_ns=120),
        ready_times_ns=(0, 0, 120),
    )

    assert result.batch2_count == 1
    assert result.batches[0] == ("r0", "r1")
    assert all(len(set(batch)) == len(batch) for batch in result.batches)


def test_analyze_reports_non_causal_without_aligned_e2e_or_fingerprint():
    result = analyze_model_trace(
        _new_trace(),
        service_curve=ServiceCurve(b1_ns=100, b2_ns=120),
    )[0]

    assert result.causal_eligible is False
    assert result.predicted_e2e_speedup is None
    assert "aligned_e2e_intervals_missing" in result.assumptions


def test_analyze_uses_aligned_e2e_intervals_for_conservative_amdahl_bound():
    records = [
        {
            "event": "MODEL_EXECUTION",
            "timestamp_monotonic_ns": 100,
            "start_ns": 0,
            "end_ns": 100,
            "physical_batch_size": 1,
            "request_ids": ["r0"],
        },
        {
            "event": "MODEL_EXECUTION",
            "timestamp_monotonic_ns": 200,
            "start_ns": 100,
            "end_ns": 200,
            "physical_batch_size": 1,
            "request_ids": ["r1"],
        },
    ]
    result = analyze_model_trace(
        records,
        service_curve=ServiceCurve(b1_ns=100, b2_ns=120),
        e2e_intervals={"r0": (0, 1_000), "r1": (0, 1_000)},
        baseline_work_fingerprint={"flow_step_count": 10, "audio_chunk_count": 2},
        candidate_work_fingerprint={"flow_step_count": 10, "audio_chunk_count": 2},
    )[0]

    assert result.causal_eligible is True
    assert result.predicted_e2e_speedup is not None
    assert 1.0 <= result.predicted_e2e_speedup <= result.predicted_model_speedup


def test_work_fingerprint_mismatch_blocks_causal_comparison():
    assert compare_work_fingerprint(
        {"flow_step_count": 10, "audio_chunk_count": 3},
        {"flow_step_count": 10, "audio_chunk_count": 4},
    ) is False
    assert compare_work_fingerprint(
        {"flow_step_count": 10},
        {"flow_step_count": 10},
    ) is True


def test_missing_service_curve_rejects_speed_claim():
    result = analyze_model_trace(_new_trace(), service_curve=ServiceCurve(b1_ns=100))[0]

    assert result.predicted_model_speedup is None
    assert result.causal_eligible is False
    assert "b2_service_time_missing" in result.assumptions


def test_feedback_driver_does_not_carry_old_lock_queue_delay_forward():
    observations = normalize_model_trace(
        [
            {"event": "MODEL_EXECUTION", "timestamp_monotonic_ns": 100,
             "start_ns": 0, "end_ns": 100, "physical_batch_size": 1,
             "request_ids": ["r0"]},
            {"event": "MODEL_EXECUTION", "timestamp_monotonic_ns": 600,
             "start_ns": 500, "end_ns": 600, "physical_batch_size": 1,
             "request_ids": ["r1"]},
            {"event": "MODEL_EXECUTION", "timestamp_monotonic_ns": 1_100,
             "start_ns": 1_000, "end_ns": 1_100, "physical_batch_size": 1,
             "request_ids": ["r0"]},
            {"event": "MODEL_EXECUTION", "timestamp_monotonic_ns": 1_600,
             "start_ns": 1_500, "end_ns": 1_600, "physical_batch_size": 1,
             "request_ids": ["r1"]},
        ]
    )
    # A zero downstream think time is the controlled sensitivity case: the
    # alternative driver is allowed to make the next demand immediately after
    # the preceding output rather than inheriting the old absolute timestamp.
    delays = (0, 0, 0, 0)
    static = simulate_central_driver(
        observations, ServiceCurve(100, 120),
        ready_times_ns=(0, 500, 1_000, 1_500),
    )
    feedback = simulate_central_driver(
        observations, ServiceCurve(100, 120),
        ready_times_ns=(0, 500, 1_000, 1_500),
        feedback_delays_ns=delays,
    )

    assert feedback.makespan_ns < static.makespan_ns
    assert feedback.batch2_count >= 1
