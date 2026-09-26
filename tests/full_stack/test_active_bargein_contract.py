import pytest

from tools.lychee_fd_bargein.active_bargein_contract import (
    active_speaking_precondition,
    server_confirmed_interrupt,
)


def _valid_snapshot():
    return {
        "logical_request_id": "A",
        "speaking": True,
        "token2wav_active": True,
        "pcm_chunks_emitted": 2,
        "playback_enqueues": 1,
    }


def _valid_interrupt_events():
    return [
        {
            "type": "CLIENT_INTERRUPT_SENT",
            "logical_request_id": "A",
            "timestamp_monotonic_ns": 200,
        },
        {
            "type": "state_change",
            "source": "model_early_exit",
            "from": "S",
            "to": "L",
            "interrupt": True,
            "interrupt_reason": "control_sl_without_tts_end",
            "logical_request_id": "A",
            "timestamp_monotonic_ns": 210,
        },
        {
            "type": "audio_interrupt",
            "source": "model_early_exit",
            "reason": "control_sl_without_tts_end",
            "server_interrupt_epoch_ms": 1234,
            "tts_old_stream_id": "old",
            "tts_new_stream_id": "new",
            "tts_abort_generation_id": 3,
            "logical_request_id": "A",
            "timestamp_monotonic_ns": 220,
        },
    ]


def test_accepts_confirmed_speaking_with_active_output():
    ok, reason = active_speaking_precondition(_valid_snapshot())

    assert ok is True
    assert reason == "ok"


def test_rejects_speaking_without_downstream_output():
    snapshot = _valid_snapshot()
    snapshot["pcm_chunks_emitted"] = 0
    snapshot["playback_enqueues"] = 0

    ok, reason = active_speaking_precondition(snapshot)

    assert ok is False
    assert reason == "active_output_missing"


def test_rejects_speaking_evidence_for_wrong_request():
    snapshot = _valid_snapshot()
    snapshot["logical_request_id"] = "B"
    snapshot["observed_owner"] = "A"

    ok, reason = active_speaking_precondition(snapshot)

    assert ok is False
    assert reason == "ownership_mismatch"


def test_accepts_server_confirmed_interrupt_only_after_client_send():
    ok, reason = server_confirmed_interrupt(_valid_interrupt_events(), "A")

    assert ok is True
    assert reason == "ok"


def test_rejects_client_marker_without_server_confirmation():
    events = [_valid_interrupt_events()[0]]

    ok, reason = server_confirmed_interrupt(events, "A")

    assert ok is False
    assert reason == "server_confirmation_missing"


def test_rejects_confirmation_for_another_request():
    events = _valid_interrupt_events()
    for event in events[1:]:
        event["logical_request_id"] = "B"

    ok, reason = server_confirmed_interrupt(events, "A")

    assert ok is False
    assert reason == "ownership_mismatch"

