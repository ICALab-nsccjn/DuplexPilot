"""Contract tests for the metadata-only J11--J22 case analyzer."""

from __future__ import annotations

from tools.analysis.analyze_joint_b22 import (
    analyze_result_root,
    compare_work_fingerprints,
    summarize_online_attempt,
    summarize_trace_events,
)


def test_analyzer_keeps_failed_attempt_without_run_metadata(tmp_path):
    case = tmp_path / "counted_HD-Burst_N8_J11_r1"
    case.mkdir()
    (case / "attempt_status.json").write_text(
        '{"status":"FAILED", "runner_rc":2, "N":8, '
        '"workload":"HD-Burst", "joint_mode":"J11", '
        '"system":"rsv_dsv_apr_joint_j11", "repeat":1}',
        encoding="utf-8",
    )
    rows, schedule = analyze_result_root(tmp_path)
    assert schedule == []
    assert len(rows) == 1
    assert rows[0]["case_id"] == case.name
    assert rows[0]["valid"] is False
    assert rows[0]["attempt_status"] == "FAILED"
    assert rows[0]["runner_rc"] == 2


def test_online_attempt_summary_reports_case_metrics_and_percentiles():
    attempt = {
        "valid": True,
        "session_count": 2,
        "completed_sessions": 2,
        "ownership_errors": 0,
        "runtime_errors": 0,
        "sessions": [
            {
                "done": True,
                "elapsed_s": 2.0,
                "pcm_chunks": 2,
                "pcm_sample_count": 24_000,
                "ttfa_ms": 100,
                "pcm_sse_epochs_ms": [1_100, 1_250],
                "event_errors": [],
                "runtime_errors": [],
            },
            {
                "done": True,
                "elapsed_s": 3.0,
                "pcm_chunks": 1,
                "pcm_sample_count": 12_000,
                "ttfa_ms": 200,
                "pcm_sse_epochs_ms": [1_200],
                "event_errors": [],
                "runtime_errors": [],
            },
        ],
    }

    summary = summarize_online_attempt(attempt)

    assert summary["valid"] is True
    assert summary["completed_sessions"] == 2
    assert summary["session_span_s"] == 3.0
    assert summary["completed_sessions_per_s"] == 2 / 3
    assert summary["useful_audio_seconds"] == 1.5
    assert summary["ttfa_p50_ms"] == 150
    assert summary["audio_gap_p95_ms"] == 150


def test_trace_summary_separates_model_and_acoustic_batch_dimensions():
    events = [
        {
            "event_type": "MODEL_ENGINE_BATCH_DISPATCH",
            "timestamp_monotonic_ns": 100,
            "request_ids": ["r0", "r1"],
            "model_batch_size": 2,
        },
        {
            "event": "FLOW_BATCH_COMPLETE",
            "timestamp_monotonic_ns": 110,
            "request_ids": ["r0", "r1"],
            "batch_size": 2,
            "next_step_indices": [1, 1],
        },
        {
            "event": "FLOW_BATCH_COMPLETE",
            "timestamp_monotonic_ns": 300,
            "request_ids": ["r0"],
            "batch_size": 1,
        },
    ]

    summary = summarize_trace_events(events)

    assert summary["model_batch2_count"] == 1
    assert summary["model_b2_row_fraction"] == 1.0
    assert summary["acoustic_batch2_count"] == 1
    assert summary["acoustic_b2_work_fraction"] == 2 / 3
    assert summary["joint_b22_event_count"] == 1


def test_trace_summary_separates_model_prefill_from_decode_rows():
    events = [
        {
            "event_type": "MODEL_ENGINE_BATCH_FORMED",
            "timestamp_monotonic_ns": 10,
            "request_ids": ["r0", "r1"],
            "model_batch_size": 2,
            "row_aware": False,
        },
        {
            "event_type": "MODEL_ENGINE_BATCH_DISPATCH",
            "timestamp_monotonic_ns": 20,
            "request_ids": ["r0", "r1"],
            "model_batch_size": 2,
        },
        {
            "event_type": "MODEL_ENGINE_BATCH_FORMED",
            "timestamp_monotonic_ns": 21,
            "request_ids": ["r0", "r1"],
            "model_batch_size": 2,
            "row_aware": True,
        },
    ]

    summary = summarize_trace_events(events)

    assert summary["model_prefill_event_count"] == 1
    assert summary["model_prefill_rows"] == 2
    assert summary["model_decode_event_count"] == 1
    assert summary["model_decode_b2_count"] == 1
    assert summary["model_decode_b2_row_fraction"] == 1.0


def test_work_fingerprint_comparison_ignores_ephemeral_request_ids():
    left = [
        {
            "event_type": "MODEL_WORK_FINGERPRINT",
            "request_id": "left-id",
            "generation_id": 0,
            "generated_token_count": 3,
            "token_sequence_sha256": "aaa",
            "termination_reason": "quota_reached",
        }
    ]
    right = [
        {
            "event_type": "MODEL_WORK_FINGERPRINT",
            "request_id": "right-id",
            "generation_id": 0,
            "generated_token_count": 3,
            "token_sequence_sha256": "aaa",
            "termination_reason": "quota_reached",
        }
    ]
    assert compare_work_fingerprints(left, right)["status"] == "MATCHED"


def test_work_fingerprint_comparison_detects_token_work_divergence():
    left = [{"event_type": "MODEL_WORK_FINGERPRINT", "generated_token_count": 3, "token_sequence_sha256": "aaa"}]
    right = [{"event_type": "MODEL_WORK_FINGERPRINT", "generated_token_count": 4, "token_sequence_sha256": "bbb"}]
    result = compare_work_fingerprints(left, right)
    assert result["status"] == "DIVERGENT"
