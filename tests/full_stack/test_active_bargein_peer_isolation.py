from tools.lychee_fd_bargein.active_bargein_peer import audit_peer_isolation


def test_peer_progress_is_allowed_when_each_event_remains_b_owned():
    a_events = [
        {"type": "PRE_INTERRUPT_SNAPSHOT", "timestamp_monotonic_ns": 100},
        {"type": "CLIENT_INTERRUPT_SENT", "timestamp_monotonic_ns": 200},
        {"type": "SERVER_CONFIRMED_INTERRUPT", "timestamp_monotonic_ns": 300},
    ]
    b_events = [
        {"type": "stage_timing", "timestamp_monotonic_ns": 110, "phase": "s"},
        {"type": "audio_chunk_pcm", "timestamp_monotonic_ns": 150, "pcm_owner": "B"},
        {"type": "state_change", "timestamp_monotonic_ns": 250, "phase": "l"},
        {"type": "playback_enqueue", "timestamp_monotonic_ns": 350, "playback_owner": "B"},
    ]

    result = audit_peer_isolation(a_events, b_events, "A", "B", post_window_ns=100)

    assert result["pass"] is True
    assert result["peer_live_observed"] is True
    assert result["cross_request_errors"] == []
    assert result["suspicious_events"] == []


def test_peer_owner_mismatch_fails_closed():
    a_events = [
        {"type": "PRE_INTERRUPT_SNAPSHOT", "timestamp_monotonic_ns": 100},
        {"type": "CLIENT_INTERRUPT_SENT", "timestamp_monotonic_ns": 200},
        {"type": "SERVER_CONFIRMED_INTERRUPT", "timestamp_monotonic_ns": 300},
    ]
    b_events = [
        {"type": "audio_chunk_pcm", "timestamp_monotonic_ns": 250, "pcm_owner": "A"},
    ]

    result = audit_peer_isolation(a_events, b_events, "A", "B", post_window_ns=100)

    assert result["pass"] is False
    assert result["cross_request_errors"]


def test_peer_reset_caused_by_interrupted_request_fails():
    a_events = [
        {"type": "PRE_INTERRUPT_SNAPSHOT", "timestamp_monotonic_ns": 100},
        {"type": "CLIENT_INTERRUPT_SENT", "timestamp_monotonic_ns": 200},
        {"type": "SERVER_CONFIRMED_INTERRUPT", "timestamp_monotonic_ns": 300},
    ]
    b_events = [
        {
            "type": "state_reset",
            "timestamp_monotonic_ns": 320,
            "caused_by_request_id": "A",
        },
    ]

    result = audit_peer_isolation(a_events, b_events, "A", "B", post_window_ns=100)

    assert result["pass"] is False
    assert result["suspicious_events"]


def test_peer_activity_that_finished_before_t0_is_not_active_peer_evidence():
    a_events = [
        {"type": "PRE_INTERRUPT_SNAPSHOT", "timestamp_monotonic_ns": 300},
        {"type": "CLIENT_INTERRUPT_SENT", "timestamp_monotonic_ns": 400},
        {"type": "SERVER_CONFIRMED_INTERRUPT", "timestamp_monotonic_ns": 500},
    ]
    b_events = [
        {"type": "audio_chunk_pcm", "timestamp_monotonic_ns": 100, "pcm_owner": "B"},
        {"type": "done", "timestamp_monotonic_ns": 200},
    ]

    result = audit_peer_isolation(a_events, b_events, "A", "B", post_window_ns=100)

    assert result["pass"] is False
    assert {item["kind"] for item in result["cross_request_errors"]} == {"peer_activity_missing"}
