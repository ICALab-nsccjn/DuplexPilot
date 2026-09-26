import pytest
import threading
from pathlib import Path

from tools.lychee_fd_bargein.official_active_bargein_runner import (
    ActiveBargeInGate,
    ConditionDrivenSessionRunner,
)


def test_gate_waits_for_speaking_and_owned_output():
    gate = ActiveBargeInGate("A")
    gate.observe({"type": "state_change", "from": "L", "to": "S", "logical_request_id": "A"})

    ok, reason = gate.can_trigger()

    assert ok is False
    assert reason == "active_output_missing"

    gate.observe({
        "type": "audio_chunk_pcm",
        "logical_request_id": "A",
        "num_samples": 960,
    })

    ok, reason = gate.can_trigger()

    assert ok is True
    assert reason == "ok"


def test_gate_rejects_client_marker_before_precondition():
    gate = ActiveBargeInGate("A")

    with pytest.raises(RuntimeError, match="precondition"):
        gate.mark_client_interrupt()


def test_gate_accepts_server_confirmation_after_client_marker():
    gate = ActiveBargeInGate("A")
    gate.observe({"type": "state_change", "from": "L", "to": "S", "logical_request_id": "A"})
    gate.observe({"type": "audio_chunk_pcm", "logical_request_id": "A", "num_samples": 960})
    marker = gate.mark_client_interrupt()

    assert marker["type"] == "CLIENT_INTERRUPT_SENT"
    gate.observe({
        "type": "state_change",
        "source": "model_early_exit",
        "from": "S",
        "to": "L",
        "interrupt": True,
        "interrupt_reason": "control_sl_without_tts_end",
        "logical_request_id": "A",
    })
    gate.observe({
        "type": "audio_interrupt",
        "source": "model_early_exit",
        "reason": "control_sl_without_tts_end",
        "server_interrupt_epoch_ms": 1234,
        "tts_old_stream_id": "old",
        "tts_new_stream_id": "new",
        "tts_abort_generation_id": 3,
        "logical_request_id": "A",
    })

    ok, reason = gate.confirmation()

    assert ok is True
    assert reason == "ok"


def test_gate_does_not_count_client_marker_as_confirmation():
    gate = ActiveBargeInGate("A")
    gate.observe({"type": "state_change", "from": "L", "to": "S", "logical_request_id": "A"})
    gate.observe({"type": "audio_chunk_pcm", "logical_request_id": "A", "num_samples": 960})
    gate.mark_client_interrupt()

    ok, reason = gate.confirmation()

    assert ok is False
    assert reason == "server_confirmation_missing"


def test_peer_request_id_ref_preserves_shared_empty_mapping():
    peer_ref = {}
    runner = ConditionDrivenSessionRunner(
        "http://127.0.0.1:7860",
        "active_barge_in",
        Path("normal.wav"),
        Path("out"),
        1,
        threading.Barrier(1),
        set(),
        threading.Lock(),
        1,
        10.0,
        "dynamic_virtualized",
        interrupt_input_path=Path("interrupt.wav"),
        peer_ready_event=threading.Event(),
        peer_request_id_ref=peer_ref,
    )

    assert runner.peer_request_id_ref is peer_ref


def test_runner_waits_for_terminal_event_after_confirmation():
    runner = ConditionDrivenSessionRunner(
        "http://127.0.0.1:7860",
        "active_barge_in",
        Path("normal.wav"),
        Path("out"),
        1,
        threading.Barrier(1),
        set(),
        threading.Lock(),
        1,
        10.0,
        "dynamic_virtualized",
        interrupt_input_path=Path("interrupt.wav"),
    )
    timer = threading.Timer(0.01, runner.done.set)
    timer.start()
    try:
        assert runner._wait_for_post_confirmation_terminal(0.2) is True
    finally:
        timer.cancel()
