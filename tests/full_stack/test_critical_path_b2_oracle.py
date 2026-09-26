from tools.benchmarks.critical_path_b2_oracle import (
    CriticalPathObservation,
    pair_net_benefit,
    simulate_critical_path,
)


def _obs(
    job_id,
    ready,
    service=10.0,
    stream=None,
    slack=None,
    deadline=None,
    first_audio=False,
):
    return CriticalPathObservation(
        job_id=job_id,
        stream_id=stream or job_id,
        ready_ms=ready,
        service_ms=service,
        compatibility_key=("flow", "cuda:1", "float32", "shape"),
        slack_ms=slack,
        deadline_ms=deadline,
        first_audio=first_audio,
    )


def test_positive_net_benefit_is_admitted_by_p1():
    observations = [_obs("a", 0, stream="a"), _obs("b", 0, stream="b")]
    assert pair_net_benefit(observations[0], observations[1], batch_speedup=2.0) == 15.0
    result = simulate_critical_path(observations, policy="p1_net_benefit", batch_speedup=2.0)
    assert result["b2_batch_count"] == 1
    assert result["batches"][0]["job_ids"] == ["a", "b"]


def test_negative_net_benefit_forces_singletons_even_when_fifo_would_batch():
    observations = [_obs("a", 0), _obs("b", 0)]
    fifo = simulate_critical_path(observations, policy="p0_fifo", batch_speedup=0.5)
    guarded = simulate_critical_path(observations, policy="p1_net_benefit", batch_speedup=0.5)
    assert fifo["b2_batch_count"] == 1
    assert guarded["b2_batch_count"] == 0
    assert guarded["fallback_count"] == 2


def test_slack_aware_wait_does_not_wait_past_first_audio_deadline():
    observations = [
        _obs("a", 0, slack=0.5, deadline=0.5, first_audio=True),
        _obs("b", 1.0, slack=100.0),
    ]
    result = simulate_critical_path(
        observations,
        policy="p2_net_benefit_slack_wait",
        batch_speedup=2.0,
        wait_ms=2.0,
    )
    assert result["wait_ms"] == 0.0
    assert result["b2_batch_count"] == 0


def test_slack_aware_wait_uses_a_ready_compatible_partner_within_window():
    observations = [
        _obs("a", 0, slack=20.0),
        _obs("b", 1.0, slack=20.0),
    ]
    result = simulate_critical_path(
        observations,
        policy="p2_net_benefit_slack_wait",
        batch_speedup=2.0,
        wait_ms=2.0,
    )
    assert result["wait_ms"] == 1.0
    assert result["b2_batch_count"] == 1


def test_pair_hint_is_a_tiebreaker_and_never_creates_a_barrier():
    observations = [
        _obs("a0", 0, stream="a"),
        _obs("b0", 0, stream="b"),
        _obs("a1", 6, stream="a"),
        _obs("b1", 20, stream="b"),
    ]
    result = simulate_critical_path(
        observations,
        policy="p3_net_benefit_pair_hint",
        batch_speedup=2.0,
        wait_ms=2.0,
    )
    assert result["b2_batch_count"] >= 1
    assert result["max_wait_ms"] <= 2.0


def test_named_compatibility_fields_can_be_relaxed_without_dropping_other_fields():
    def named(step):
        return (
            ("model_identity", "flow"),
            ("device", "cuda:1"),
            ("dtype", "float32"),
            ("current_chunk_shape", "shape"),
            ("step_index", step),
        )

    a = CriticalPathObservation("a", "a", 0, 10, named(0))
    b = CriticalPathObservation("b", "b", 0, 10, named(1))
    result = simulate_critical_path(
        [a, b],
        policy="p1_net_benefit",
        batch_speedup=2.0,
        ignored_fields=frozenset({"step_index"}),
    )
    assert result["b2_batch_count"] == 1


def test_singleton_baseline_respects_ready_time_idle_gaps():
    observations = [_obs("a", 0, service=10), _obs("b", 20, service=10)]
    result = simulate_critical_path(
        observations,
        policy="p1_net_benefit",
        batch_speedup=2.0,
        max_batch_size=1,
    )
    assert result["b2_batch_count"] == 0
    assert result["baseline_makespan_ms"] == 30.0
    assert result["makespan_ms"] == 30.0


def test_oracle_returns_per_job_and_stream_completion_times():
    observations = [
        _obs("a0", 0, service=10, stream="a"),
        _obs("b0", 0, service=10, stream="b"),
        _obs("a1", 20, service=10, stream="a"),
        _obs("b1", 20, service=10, stream="b"),
    ]
    result = simulate_critical_path(
        observations,
        policy="p1_net_benefit",
        batch_speedup=2.0,
    )
    assert set(result["job_completion_ms"]) == {"a0", "b0", "a1", "b1"}
    assert set(result["stream_completion_ms"]) == {"a", "b"}
    assert result["stream_completion_ms"]["a"] == result["job_completion_ms"]["a1"]
