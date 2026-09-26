"""Tests for the metadata-only online timing summary used by J11--J22."""

from __future__ import annotations

from tools.benchmarks.apr_public_online_e2e import summarize_session_events


def test_summary_extracts_ttfa_gap_done_and_pcm_work():
    events = [
        {
            "type": "status",
            "server_sse_send_epoch_ms": 1_000,
        },
        {
            "type": "audio_chunk_pcm",
            "server_sse_send_epoch_ms": 1_250,
            "server_audio_emit_epoch_ms": 1_240,
            "server_chunk_recv_to_emit_ms": 40,
            "sample_rate": 24_000,
            "frame_audio": {"num_samples": 2_400, "pcm_b64": "AAAA"},
        },
        {
            "type": "audio_chunk_pcm",
            "server_sse_send_epoch_ms": 1_400,
            "server_audio_emit_epoch_ms": 1_390,
            "sample_rate": 24_000,
            "frame_audio": {"num_samples": 1_200, "pcm_b64": "BBBB"},
        },
        {"type": "done", "server_sse_send_epoch_ms": 1_500},
    ]

    summary = summarize_session_events(events, client_start_epoch_ms=1_000)

    assert summary["first_pcm_sse_epoch_ms"] == 1_250
    assert summary["first_pcm_emit_epoch_ms"] == 1_240
    assert summary["ttfa_ms"] == 250
    assert summary["pcm_chunk_count"] == 2
    assert summary["pcm_sample_count"] == 3_600
    assert summary["audio_gap_ms"] == 150
    assert summary["done_epoch_ms"] == 1_500


def test_summary_is_missing_safe_and_does_not_decode_pcm():
    summary = summarize_session_events(
        [{"type": "status", "server_sse_send_epoch_ms": 10}],
        client_start_epoch_ms=0,
    )

    assert summary["first_pcm_sse_epoch_ms"] is None
    assert summary["ttfa_ms"] is None
    assert summary["pcm_chunk_count"] == 0
    assert summary["pcm_sample_count"] == 0
    assert summary["audio_gap_ms"] is None

