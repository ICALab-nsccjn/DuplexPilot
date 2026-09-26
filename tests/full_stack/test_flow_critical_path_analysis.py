from tools.benchmarks.analyze_flow_critical_path import (
    analyze_event_trace,
    estimate_b1_cost_by_step,
)


def _event(name, timestamp, **fields):
    value = {
        "event": name,
        "event_type": name,
        "timestamp_monotonic_ns": timestamp,
    }
    value.update(fields)
    return value


def _step_fields(request_ids, steps, sequence_nos=None, batch_size=None):
    if sequence_nos is None:
        sequence_nos = [0] * len(request_ids)
    if batch_size is None:
        batch_size = len(request_ids)
    return {
        "request_ids": request_ids,
        "generation_ids": [1] * len(request_ids),
        "sequence_no": sequence_nos,
        "step_indices": steps,
        "last_chunk": [False] * len(request_ids),
        "batch_size": batch_size,
    }


def _singleton_step(request_id, step, start, end, wall_ms=10.0):
    fields = _step_fields([request_id], [step], batch_size=1)
    return [
        _event("FLOW_STEP_START", start, **fields),
        _event(
            "FLOW_BATCH_TIMING",
            end,
            **fields,
            wall_time_ms=wall_ms,
            cuda_time_ms=wall_ms,
        ),
        _event("FLOW_STEP_END", end, **fields),
    ]


def test_b2_critical_path_propagation_distinguishes_rows_with_different_slack():
    events = [
        _event(
            "ACOUSTIC_CHUNK_READY",
            0,
            **_step_fields(["a", "b"], [0, 0], batch_size=0),
        ),
        _event(
            "FLOW_STEP_START",
            100,
            **_step_fields(["a", "b"], [0, 0], batch_size=2),
        ),
        _event(
            "FLOW_BATCH_TIMING",
            110,
            **_step_fields(["a", "b"], [0, 0], batch_size=2),
            wall_time_ms=10.0,
            cuda_time_ms=10.0,
        ),
        _event(
            "FLOW_STEP_END",
            110,
            **_step_fields(["a", "b"], [0, 0], batch_size=2),
        ),
    ]
    events += _singleton_step("a", 1, 110, 120)
    events += _singleton_step("b", 1, 130, 140)
    events += [
        _event(
            "PCM_READY",
            125,
            **_step_fields(["a"], [10], batch_size=0),
        ),
        _event(
            "PCM_READY",
            150,
            **_step_fields(["b"], [10], batch_size=0),
        ),
    ]

    result = analyze_event_trace(events, b1_cost_by_step={0: 20.0, 1: 10.0})

    assert result["summary"]["b2_work_fraction"] == 0.5
    assert result["summary"]["b2_critical_work_fraction"] == 0.25
    b2_rows = [row for row in result["rows"] if row["batch_size"] == 2]
    assert {row["request_id"] for row in b2_rows} == {"a", "b"}
    assert next(row for row in b2_rows if row["request_id"] == "a")[
        "propagated_saving_ms"
    ] == 15.0
    assert next(row for row in b2_rows if row["request_id"] == "b")[
        "propagated_saving_ms"
    ] == 0.0


def test_b2_equivalent_cost_is_recorded_as_counterfactual_saving():
    events = _singleton_step("a", 0, 100, 110, wall_ms=10.0)
    events += _singleton_step("b", 0, 110, 120, wall_ms=10.0)
    events += [
        _event("ACOUSTIC_CHUNK_READY", 0, **_step_fields(["a"], [0], batch_size=0)),
        _event("ACOUSTIC_CHUNK_READY", 0, **_step_fields(["b"], [0], batch_size=0)),
        _event("PCM_READY", 130, **_step_fields(["a"], [10], batch_size=0)),
        _event("PCM_READY", 140, **_step_fields(["b"], [10], batch_size=0)),
    ]
    result = analyze_event_trace(events, b1_cost_by_step={0: 25.0})
    assert result["summary"]["b2_batch_count"] == 0
    assert result["summary"]["estimated_b1_equivalent_ms"] == 0.0


def test_b1_cost_estimator_ignores_batched_steps_and_uses_singleton_medians():
    events = _singleton_step("a", 0, 100, 120, wall_ms=20.0)
    events += _singleton_step("b", 0, 200, 230, wall_ms=30.0)
    events += [
        _event(
            "FLOW_STEP_START",
            300,
            **_step_fields(["a", "b"], [0, 0], batch_size=2),
        ),
        _event(
            "FLOW_BATCH_TIMING",
            310,
            **_step_fields(["a", "b"], [0, 0], batch_size=2),
            wall_time_ms=10.0,
        ),
        _event(
            "FLOW_STEP_END",
            310,
            **_step_fields(["a", "b"], [0, 0], batch_size=2),
        ),
    ]
    costs = estimate_b1_cost_by_step(events)
    assert costs == {0: 25.0}
