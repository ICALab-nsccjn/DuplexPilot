"""Analyze the joint ``(B_model, B_acoustic)`` online experiment.

The analyzer is deliberately metadata-only.  It consumes the public client
attempt JSON and the scheduler/model traces, but never decodes ``pcm_b64`` or
retains tensors.  It is usable both as a small library (the unit tests import
the two summary functions) and as a command-line report generator.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from lychee_fd.runtime.apr.joint_execution_trace import JointEventCorrelator


_MISSING = object()


def _fingerprint_signature(events: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], str]:
    """Return terminal per-request work records without ephemeral IDs.

    ``MODEL_WORK_FINGERPRINT`` is emitted cumulatively while a request is
    decoded.  Comparing every cumulative record would incorrectly classify
    two identical runs as different merely because their request UUIDs differ.
    We therefore retain the last record for each request/generation and sort
    the resulting multiset.  Request IDs are intentionally excluded.
    """
    terminal: dict[tuple[str, str], Mapping[str, Any]] = {}
    ordinal = 0
    for event in events:
        if _event_type(event) != "MODEL_WORK_FINGERPRINT":
            continue
        request_id = str(event.get("request_id") or "")
        generation_id = str(event.get("generation_id") or "")
        terminal[(request_id, generation_id)] = event
        ordinal += 1
    records: list[tuple[Any, ...]] = []
    for event in terminal.values():
        records.append(
            (
                _as_int(event.get("generation_id")),
                _as_int(event.get("generated_token_count")),
                str(event.get("token_sequence_sha256") or event.get("work_digest") or ""),
                str(event.get("stoken_sequence_sha256") or ""),
                str(event.get("control_sequence_sha256") or ""),
                str(event.get("termination_reason") or ""),
                str(event.get("request_finished_reason") or ""),
            )
        )
    normalized = tuple(sorted(records, key=repr))
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest() if normalized else ""
    return normalized, digest


def compare_work_fingerprints(
    left_events: Sequence[Mapping[str, Any]],
    right_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two case-level work fingerprints, ignoring request UUIDs."""
    left_records, left_digest = _fingerprint_signature(left_events)
    right_records, right_digest = _fingerprint_signature(right_events)
    if not left_records or not right_records:
        status = "UNKNOWN"
    elif left_records == right_records:
        status = "MATCHED"
    else:
        status = "DIVERGENT"
    return {
        "status": status,
        "left_digest": left_digest,
        "right_digest": right_digest,
        "left_count": len(left_records),
        "right_count": len(right_records),
    }


def _event_type(event: Mapping[str, Any]) -> str:
    return str(
        event.get("event_type")
        or event.get("event")
        or event.get("kind")
        or ""
    ).upper()


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_numeric_token(value: Any) -> float | None:
    """Parse a numeric field from ``nvidia-smi`` output.

    The runtime monitor is invoked with ``nounits`` but keeping this parser
    tolerant of ``MiB``/``%`` suffixes makes old and manually inspected
    captures explicit rather than silently dropping a sample.
    """
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    allowed = "0123456789.+-eE"
    token = "".join(char for char in text if char in allowed)
    if token in {"", "+", "-", "."}:
        return None
    try:
        number = float(token)
    except ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def summarize_gpu_monitor(path: Path | str) -> dict[str, Any]:
    """Summarize in-run GPU samples without confusing an end snapshot for a peak.

    New attempts contain headerless rows emitted by ``nvidia-smi`` as
    ``timestamp,index,memory.used,memory.total,utilization.gpu``.  Older
    attempts have no monitor file; those are deliberately reported as
    ``MISSING`` rather than using their one-time ``gpu_after.csv`` value as a
    peak estimate.
    """
    monitor_path = Path(path)
    result: dict[str, Any] = {
        "gpu_monitor_status": "MISSING",
        "gpu_monitor_samples": 0,
        "gpu0_peak_memory_mib": None,
        "gpu1_peak_memory_mib": None,
        "gpu0_peak_memory_bytes": None,
        "gpu1_peak_memory_bytes": None,
    }
    if not monitor_path.exists():
        return result
    peaks: dict[int, float] = {}
    samples = 0
    try:
        with monitor_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                lowered = " ".join(str(value).lower() for value in row)
                if "memory.used" in lowered or "timestamp" in lowered and "index" in lowered:
                    continue
                if len(row) >= 5:
                    index = _parse_numeric_token(row[1])
                    memory = _parse_numeric_token(row[2])
                elif len(row) >= 2:
                    # Tolerate the older four-column end-snapshot shape, but
                    # only when it is explicitly passed as a monitor file.
                    index = _parse_numeric_token(row[0])
                    memory = _parse_numeric_token(row[1])
                else:
                    continue
                if index is None or memory is None or index < 0 or memory < 0:
                    continue
                gpu_index = int(index)
                peaks[gpu_index] = max(peaks.get(gpu_index, 0.0), memory)
                samples += 1
    except OSError:
        return result
    if not samples:
        return result
    result["gpu_monitor_status"] = "OBSERVED"
    result["gpu_monitor_samples"] = samples
    for gpu_index in (0, 1):
        value = peaks.get(gpu_index)
        if value is None:
            continue
        result[f"gpu{gpu_index}_peak_memory_mib"] = value
        result[f"gpu{gpu_index}_peak_memory_bytes"] = int(round(value * 1024 * 1024))
    return result


def _number_list(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        number = _as_float(value)
        if number is not None:
            result.append(number)
    return result


def percentile(values: Sequence[float], q: float) -> float | None:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _event_epoch_ms(event: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = _as_int(event.get(name))
        if value is not None:
            return value
    return None


def _pcm_event_data(events: Sequence[Mapping[str, Any]]) -> tuple[list[int], int, int | None]:
    """Return server emission times, sample count and observed sample rate."""
    epochs: list[int] = []
    sample_count = 0
    sample_rate: int | None = None
    for event in events:
        if str(event.get("type") or "") != "audio_chunk_pcm":
            continue
        timestamp = _event_epoch_ms(
            event, "server_sse_send_epoch_ms", "server_audio_emit_epoch_ms"
        )
        if timestamp is not None:
            epochs.append(timestamp)
        frame_audio = event.get("frame_audio")
        if isinstance(frame_audio, Mapping):
            raw_count = frame_audio.get("num_samples", frame_audio.get("samples"))
            raw_rate = frame_audio.get("sample_rate")
        else:
            raw_count = event.get("num_samples", event.get("sample_count"))
            raw_rate = event.get("sample_rate")
        count = _as_int(raw_count, 0) or 0
        sample_count += max(0, count)
        rate = _as_int(raw_rate)
        if rate and sample_rate is None:
            sample_rate = rate
    return epochs, sample_count, sample_rate


def _session_metadata(session: Mapping[str, Any]) -> dict[str, Any]:
    """Fill timing fields for old attempts that predate the client patch."""
    raw_events = session.get("events")
    events = [event for event in raw_events if isinstance(event, Mapping)] if isinstance(raw_events, list) else []
    epochs, sample_count, sample_rate = _pcm_event_data(events)
    start_epoch = _as_int(session.get("client_start_epoch_ms"))
    if not epochs:
        epochs = [
            timestamp
            for event in events
            for timestamp in (
                _event_epoch_ms(event, "server_sse_send_epoch_ms"),
            )
            if timestamp is not None
            and str(event.get("type") or "") == "audio_chunk_pcm"
        ]
    first = min(epochs) if epochs else None
    last = max(epochs) if epochs else None
    done_epochs = [
        timestamp
        for event in events
        for timestamp in (_event_epoch_ms(event, "server_sse_send_epoch_ms"),)
        if timestamp is not None and str(event.get("type") or "") == "done"
    ]
    done_epoch = min(done_epochs) if done_epochs else None
    observed = dict(session)
    if observed.get("pcm_sample_count") is None and sample_count:
        observed["pcm_sample_count"] = sample_count
    if observed.get("pcm_chunk_count") is None and epochs:
        observed["pcm_chunk_count"] = len(epochs)
    if observed.get("pcm_chunks") is None and epochs:
        observed["pcm_chunks"] = len(epochs)
    if observed.get("pcm_sse_epochs_ms") is None and epochs:
        observed["pcm_sse_epochs_ms"] = epochs
    if observed.get("ttfa_ms") is None and first is not None and start_epoch is not None:
        observed["ttfa_ms"] = first - start_epoch
    if observed.get("audio_gap_ms") is None and first is not None and last is not None:
        observed["audio_gap_ms"] = last - first
    if observed.get("done_epoch_ms") is None and done_epoch is not None:
        observed["done_epoch_ms"] = done_epoch
    if observed.get("sample_rate") is None and sample_rate is not None:
        observed["sample_rate"] = sample_rate
    return observed


def summarize_online_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize one public online attempt without decoding audio payloads."""
    raw_sessions = attempt.get("sessions")
    sessions = [
        _session_metadata(session)
        for session in raw_sessions
        if isinstance(session, Mapping)
    ] if isinstance(raw_sessions, list) else []
    completed = [session for session in sessions if bool(session.get("done"))]
    elapsed = _number_list(
        session.get("elapsed_s")
        for session in completed
        if session.get("elapsed_s") is not None
    )
    start_epochs = _number_list(
        session.get("client_start_epoch_ms")
        for session in sessions
        if session.get("client_start_epoch_ms") is not None
    )
    end_epochs = _number_list(
        session.get("client_end_epoch_ms")
        for session in sessions
        if session.get("client_end_epoch_ms") is not None
    )
    if start_epochs and end_epochs:
        session_span_s = max(0.0, (max(end_epochs) - min(start_epochs)) / 1000.0)
    else:
        session_span_s = max(elapsed, default=0.0)

    ttfa = _number_list(
        session.get("ttfa_ms") for session in completed if session.get("ttfa_ms") is not None
    )
    gaps: list[float] = []
    for session in completed:
        raw_epochs = session.get("pcm_sse_epochs_ms")
        epochs = _number_list(raw_epochs) if isinstance(raw_epochs, (list, tuple)) else []
        gaps.extend(right - left for left, right in zip(epochs, epochs[1:]) if right >= left)
    if not gaps:
        gaps = _number_list(
            session.get("audio_gap_ms")
            for session in completed
            if session.get("audio_gap_ms") is not None
        )

    total_samples = 0
    for session in completed:
        total_samples += max(0, _as_int(session.get("pcm_sample_count"), 0) or 0)
    useful_audio_seconds = total_samples / 24_000.0
    session_count = _as_int(attempt.get("session_count"), len(sessions)) or len(sessions)
    valid = bool(attempt.get("valid"))
    return {
        "valid": valid,
        "session_count": session_count,
        "completed_sessions": len(completed),
        "completion_rate": len(completed) / session_count if session_count else 0.0,
        "ownership_errors": _as_int(attempt.get("ownership_errors"), 0) or 0,
        "runtime_errors": _as_int(attempt.get("runtime_errors"), 0) or 0,
        "pcm_chunks": sum(max(0, _as_int(session.get("pcm_chunk_count", session.get("pcm_chunks")), 0) or 0) for session in completed),
        "pcm_sample_count": total_samples,
        "useful_audio_seconds": useful_audio_seconds,
        "session_span_s": session_span_s,
        "completed_sessions_per_s": len(completed) / session_span_s if session_span_s > 0 else None,
        "useful_audio_throughput": useful_audio_seconds / session_span_s if session_span_s > 0 else None,
        "ttfa_p50_ms": percentile(ttfa, 0.50),
        "ttfa_p95_ms": percentile(ttfa, 0.95),
        "ttfa_p99_ms": percentile(ttfa, 0.99),
        "completion_latency_p50_s": percentile(elapsed, 0.50),
        "completion_latency_p95_s": percentile(elapsed, 0.95),
        "completion_latency_p99_s": percentile(elapsed, 0.99),
        "audio_gap_p50_ms": percentile(gaps, 0.50),
        "audio_gap_p95_ms": percentile(gaps, 0.95),
        "audio_gap_p99_ms": percentile(gaps, 0.99),
        "audio_gap_observations": len(gaps),
        "timing_observations": len(ttfa),
    }


def _trace_batch_size(event: Mapping[str, Any], *, model: bool) -> int:
    keys = ("model_batch_size", "batch_size") if model else ("acoustic_batch_size", "batch_size")
    for key in keys:
        value = _as_int(event.get(key))
        if value is not None and value >= 0:
            if value > 0:
                return value
    for key in ("request_ids", "scheduled_request_ids", "output_request_ids"):
        value = event.get(key)
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


def _trace_events_for_correlator(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for event in events:
        event_type = _event_type(event)
        timestamp = _as_int(event.get("timestamp_monotonic_ns"))
        if timestamp is None:
            continue
        if event_type == "MODEL_ENGINE_BATCH_DISPATCH":
            normalized.append({
                "event_type": event_type,
                "timestamp_monotonic_ns": timestamp,
                "model_batch_size": _trace_batch_size(event, model=True),
                "scheduled_request_ids": event.get("request_ids") or event.get("scheduled_request_ids") or [],
                "critical_path_flag": event.get("critical_path_flag", event.get("critical_path", False)),
            })
        elif event_type == "FLOW_BATCH_COMPLETE":
            normalized.append({
                "event_type": event_type,
                "timestamp_monotonic_ns": timestamp,
                "acoustic_batch_size": _trace_batch_size(event, model=False),
                "scheduled_request_ids": event.get("request_ids") or event.get("scheduled_request_ids") or [],
                "critical_path_flag": event.get("critical_path_flag", event.get("critical_path", False)),
            })
    return normalized


def summarize_trace_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate model and acoustic batch dimensions and correlate them."""
    materialized = [event for event in events if isinstance(event, Mapping)]
    formed_model_events = [
        event for event in materialized if _event_type(event) == "MODEL_ENGINE_BATCH_DISPATCH"
    ]
    all_formed_events = [
        event for event in materialized
        if _event_type(event) == "MODEL_ENGINE_BATCH_FORMED"
    ]

    # The row-aware driver emits an explicit DISPATCH event for decode steps;
    # legacy execution does not.  Initial prefill is represented by a
    # ``row_aware=false`` FORM event and is reported separately.  This avoids
    # attributing an initialization batch to the steady-state B_model result.
    prefill_events = [
        event for event in all_formed_events
        if event.get("row_aware") is False
    ]
    if formed_model_events:
        model_events = formed_model_events
    else:
        model_events = [
            event for event in all_formed_events
            if event.get("row_aware") is not False
        ]
        if not model_events:
            model_events = all_formed_events
    acoustic_events = [
        event for event in materialized if _event_type(event) == "FLOW_BATCH_COMPLETE"
    ]
    model_sizes = [_trace_batch_size(event, model=True) for event in model_events]
    prefill_sizes = [_trace_batch_size(event, model=True) for event in prefill_events]
    acoustic_sizes = [_trace_batch_size(event, model=False) for event in acoustic_events]
    model_rows = sum(model_sizes)
    acoustic_rows = sum(acoustic_sizes)
    model_b2 = [size for size in model_sizes if size >= 2]
    prefill_b2 = [size for size in prefill_sizes if size >= 2]
    acoustic_b2 = [size for size in acoustic_sizes if size >= 2]

    combined_attempts = sum(
        _event_type(event) == "FLOW_MIXED_CHUNK_PADDING_BATCH_ATTEMPT"
        for event in materialized
    )
    combined_completes = sum(
        _event_type(event) == "FLOW_MIXED_CHUNK_PADDING_BATCH_COMPLETE"
        for event in materialized
    )
    timing_events = [
        event for event in materialized if _event_type(event) == "FLOW_BATCH_TIMING"
    ]
    flow_wall_ms = sum(
        _as_float(event.get("wall_time_ms"), 0.0) or 0.0 for event in timing_events
    )
    flow_cuda_ms = sum(
        _as_float(event.get("cuda_time_ms", event.get("CUDA_time")), 0.0) or 0.0
        for event in timing_events
    )
    lock_wait_ns = sum(
        _as_int(event.get("wait_ns", event.get("lock_wait_ns", event.get("duration_ns"))), 0) or 0
        for event in materialized
        if _event_type(event) in {"MODEL_LOCK_WAIT", "MODEL_DRIVER_DEMAND_WAIT"}
    )
    fallback_reasons = Counter(
        str(event.get("reason") or event.get("fallback_reason") or "unknown")
        for event in materialized
        if _event_type(event) in {"FLOW_BATCH_FALLBACK", "FLOW_BATCH_REJECTION_SUMMARY"}
    )

    correlator = JointEventCorrelator(join_window_ns=50_000_000, max_events=max(1, len(materialized) + 1))
    for event in _trace_events_for_correlator(materialized):
        event_type = str(event.pop("event_type"))
        timestamp = int(event.pop("timestamp_monotonic_ns"))
        correlator.record(event_type, timestamp_monotonic_ns=timestamp, **event)
    correlated = correlator.snapshot()
    joint_rows = [row for row in correlated if row.get("joint_2x2")]
    joint_critical = [row for row in correlated if row.get("joint_critical_path")]

    return {
        "trace_event_count": len(materialized),
        "model_batch_event_count": len(model_events),
        "model_batch2_count": len(model_b2),
        "model_rows": model_rows,
        "model_b2_row_fraction": sum(model_b2) / model_rows if model_rows else 0.0,
        "model_batch2_row_count": sum(model_b2),
        "model_decode_event_count": len(model_events),
        "model_decode_b2_count": len(model_b2),
        "model_decode_rows": model_rows,
        "model_decode_b2_row_fraction": sum(model_b2) / model_rows if model_rows else 0.0,
        "model_decode_b2_row_count": sum(model_b2),
        "model_prefill_event_count": len(prefill_events),
        "model_prefill_b2_count": len(prefill_b2),
        "model_prefill_rows": sum(prefill_sizes),
        "model_prefill_b2_row_fraction": sum(prefill_b2) / sum(prefill_sizes) if prefill_sizes else 0.0,
        "acoustic_batch_event_count": len(acoustic_events),
        "acoustic_batch2_count": len(acoustic_b2),
        "acoustic_rows": acoustic_rows,
        "acoustic_b2_work_fraction": sum(acoustic_b2) / acoustic_rows if acoustic_rows else 0.0,
        "acoustic_batch2_row_count": sum(acoustic_b2),
        "mixed_padding_attempt_count": combined_attempts,
        "mixed_padding_complete_count": combined_completes,
        "flow_wall_ms": flow_wall_ms,
        "flow_cuda_ms": flow_cuda_ms,
        "model_lock_wait_ms": lock_wait_ns / 1_000_000.0,
        "fallback_event_count": sum(fallback_reasons.values()),
        "fallback_reasons": dict(fallback_reasons),
        "joint_b22_event_count": len(joint_rows),
        "joint_b22_critical_path_count": len(joint_critical),
        "joint_same_request_pair_count": sum(bool(row.get("same_request_pair")) for row in joint_rows),
        "joint_critical_path_fraction": len(joint_critical) / len(correlated) if correlated else 0.0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
    return events


def _failed_attempt_row(result_root: Path, status_path: Path) -> dict[str, Any] | None:
    """Create an explicit invalid row for a runner failure before metadata exists."""
    status = _read_json(status_path)
    if not isinstance(status, Mapping):
        return None
    case_root = status_path.parent
    try:
        case_id = str(case_root.relative_to(result_root))
    except ValueError:
        return None
    return {
        "case_id": case_id,
        "system": status.get("system", ""),
        "joint_mode": status.get("joint_mode", ""),
        "workload": status.get("workload", ""),
        "N": status.get("N", ""),
        "repeat": status.get("repeat", ""),
        "trace": status.get("trace", ""),
        "max_model_batch_size": status.get("max_model_batch_size", ""),
        "max_acoustic_batch_size": status.get("max_acoustic_batch_size", ""),
        "attempt_status": status.get("status", "FAILED"),
        "runner_rc": status.get("runner_rc", ""),
        "failure_reason": status.get("failure_reason", ""),
        "valid": False,
        "session_count": _as_int(status.get("session_count"), 0) or 0,
        "completed_sessions": 0,
        "completion_rate": 0.0,
        "ownership_errors": _as_int(status.get("ownership_errors"), 0) or 0,
        "runtime_errors": _as_int(status.get("runtime_errors"), 0) or 0,
        "pcm_chunks": 0,
        "pcm_sample_count": 0,
        "useful_audio_seconds": 0.0,
        "session_span_s": None,
        "completed_sessions_per_s": None,
        "useful_audio_throughput": None,
        "work_fingerprint_status": "NOT_OBSERVED",
        "work_fingerprint_digest": "",
        "trace_event_count": 0,
        "model_batch_event_count": 0,
        "model_batch2_count": 0,
        "model_b2_row_fraction": 0.0,
        "model_decode_b2_row_fraction": 0.0,
        "acoustic_batch_event_count": 0,
        "acoustic_batch2_count": 0,
        "acoustic_b2_work_fraction": 0.0,
        "mixed_padding_complete_count": 0,
        "gpu_monitor_status": "MISSING",
        "gpu_monitor_samples": 0,
        "gpu0_peak_memory_mib": None,
        "gpu1_peak_memory_mib": None,
        "gpu0_peak_memory_bytes": None,
        "gpu1_peak_memory_bytes": None,
    }


def _resolve_trace(root: Path, metadata: Mapping[str, Any], key: str, filename: str) -> Path:
    value = metadata.get(key)
    if value:
        candidate = Path(str(value))
        if candidate.exists():
            return candidate
    return root / filename


def _safe_metadata(value: Any, *, depth: int = 0) -> Any:
    """Keep schedule traces bounded and free of audio/tensor payloads."""
    if depth > 2:
        return "<truncated>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key).lower()
            if any(token in name for token in ("pcm_b64", "frame_audio", "tensor", "audio_bytes", "payload")):
                continue
            result[str(key)] = _safe_metadata(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {"length": len(value), "sha256": __import__("hashlib").sha256(repr(value).encode()).hexdigest()}
        return [_safe_metadata(item, depth=depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 512:
            return value[:512] + "…"
        return value
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def analyze_result_root(result_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for metadata_path in sorted(result_root.rglob("run_metadata.json")):
        case_root = metadata_path.parent
        attempt_path = case_root / "client" / "online_attempt.json"
        if not attempt_path.exists():
            continue
        metadata = _read_json(metadata_path)
        attempt = _read_json(attempt_path)
        online = summarize_online_attempt(attempt)
        trace_events: list[dict[str, Any]] = []
        for key, filename in (
            ("online_trace", "online_trace.jsonl"),
            ("model_execution_trace", "model_execution_trace.jsonl"),
            ("opportunity_trace", "opportunity_trace.jsonl"),
        ):
            source = _resolve_trace(case_root, metadata, key, filename)
            source_events = _read_jsonl(source)
            for event in source_events:
                normalized = dict(event)
                normalized["_source"] = source.name
                trace_events.append(normalized)
                schedule.append({
                    "case_id": str(case_root.relative_to(result_root)),
                    "system": metadata.get("system", attempt.get("system", "")),
                    "workload": metadata.get("workload", ""),
                    "N": metadata.get("N", ""),
                    "repeat": metadata.get("repeat", attempt.get("repeat", "")),
                    "source": source.name,
                    **_safe_metadata(event),
                })
        trace = summarize_trace_events(trace_events)
        case_id = str(case_root.relative_to(result_root))
        row: dict[str, Any] = {
            "case_id": case_id,
            "system": metadata.get("system", attempt.get("system", "")),
            "joint_mode": metadata.get("joint_mode", ""),
            "workload": metadata.get("workload", ""),
            "N": metadata.get("N", ""),
            "repeat": metadata.get("repeat", attempt.get("repeat", "")),
            "trace": metadata.get("trace", ""),
            "max_model_batch_size": metadata.get("max_model_batch_size", ""),
            "max_acoustic_batch_size": metadata.get("max_acoustic_batch_size", ""),
            "work_fingerprint_status": "NOT_OBSERVED",
            "attempt_status": "COMPLETED",
        }
        row.update(online)
        row.update(trace)
        row.update(summarize_gpu_monitor(case_root / "gpu_monitor.csv"))
        fingerprints = [
            event for event in trace_events if _event_type(event) == "MODEL_WORK_FINGERPRINT"
        ]
        if fingerprints:
            _, digest = _fingerprint_signature(fingerprints)
            # A case is observable when it contains at least one terminal
            # record per logical request.  The cumulative records themselves
            # are expected to have different token hashes over time.
            row["work_fingerprint_status"] = "OBSERVED"
            row["work_fingerprint_digest"] = digest
            row["model_work_fingerprint_count"] = len(fingerprints)
        else:
            row["work_fingerprint_digest"] = ""
        rows.append(row)
        seen_case_ids.add(case_id)

    # A runner can fail before it has a chance to write run_metadata.json
    # (for example, a launch/validation syntax failure).  The matrix launcher
    # writes attempt_status.json for exactly this case.  Keep that evidence in
    # the metrics table instead of silently shrinking the denominator.
    for status_path in sorted(result_root.rglob("attempt_status.json")):
        case_id = str(status_path.parent.relative_to(result_root))
        if case_id in seen_case_ids:
            continue
        row = _failed_attempt_row(result_root, status_path)
        if row is not None:
            rows.append(row)
    return rows, schedule


def _ratio_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("workload", "")), str(row.get("N", "")), str(row.get("repeat", "")))
        grouped[key][str(row.get("joint_mode") or row.get("system") or "")] = row
    result: list[dict[str, Any]] = []
    for (workload, n, repeat), systems in sorted(grouped.items()):
        for left, right in (
            ("J11", "J21"),
            ("J11", "J12"),
            ("J11", "J22"),
            ("J21", "J22"),
            ("J12", "J22"),
            ("J21", "J12"),
        ):
            a = systems.get(left)
            b = systems.get(right)
            if not a or not b:
                continue
            av = _as_float(a.get("useful_audio_throughput"))
            bv = _as_float(b.get("useful_audio_throughput"))
            result.append({
                "workload": workload,
                "N": n,
                "repeat": repeat,
                "comparison": f"{right}/{left}",
                "left_case": a.get("case_id"),
                "right_case": b.get("case_id"),
                "useful_audio_throughput_ratio": bv / av if av and bv is not None else None,
                "session_span_ratio": (
                    _as_float(a.get("session_span_s")) / _as_float(b.get("session_span_s"))
                    if _as_float(a.get("session_span_s")) and _as_float(b.get("session_span_s")) else None
                ),
                "work_comparability": (
                    "MATCHED" if a.get("work_fingerprint_digest")
                    and a.get("work_fingerprint_digest") == b.get("work_fingerprint_digest")
                    else ("DIVERGENT" if a.get("work_fingerprint_digest") and b.get("work_fingerprint_digest") else "UNKNOWN")
                ),
                "left_work_fingerprint_digest": a.get("work_fingerprint_digest", ""),
                "right_work_fingerprint_digest": b.get("work_fingerprint_digest", ""),
            })
    return result


def write_reports(result_root: Path) -> dict[str, Path]:
    rows, schedule = analyze_result_root(result_root)
    metrics_path = result_root / "APR_JOINT_B22_ONLINE_METRICS.csv"
    schedule_path = result_root / "APR_JOINT_B22_ONLINE_SCHEDULE_TRACE.jsonl"
    report_path = result_root / "APR_JOINT_B22_ONLINE_REPORT.md"
    _write_csv(metrics_path, rows)
    with schedule_path.open("w", encoding="utf-8") as handle:
        for event in schedule:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    ratios = _ratio_rows(rows)
    lines = [
        "# APR Joint (B_model, B_acoustic) Online Report",
        "",
        "This report is generated from metadata-only public API attempts and scheduler traces.",
        "It keeps model batch and acoustic batch dimensions separate; local speedups are not multiplied.",
        "",
        f"- Cases discovered: **{len(rows)}**",
        f"- Validity is inherited from each attempt; malformed/failed attempts remain in the CSV.",
        f"- Paired ratio rows available: **{len(ratios)}**",
        "",
        "## Case summary",
        "",
        "| workload | N | repeat | joint | valid | completion | model decode B2 rows | model prefill B2 rows | acoustic B2 work | mixed-padding completes | useful-audio throughput | span (s) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {N} | {repeat} | {joint_mode} | {valid} | {completion_rate:.3f} | {decode_b2:.3f} | {prefill_b2:.3f} | {acoustic_b2_work_fraction:.3f} | {mixed_padding_complete_count} | {throughput} | {span} |".format(
                workload=row.get("workload", ""), N=row.get("N", ""), repeat=row.get("repeat", ""),
                joint_mode=row.get("joint_mode", ""), valid=row.get("valid", ""),
                completion_rate=float(row.get("completion_rate") or 0.0),
                decode_b2=float(row.get("model_decode_b2_row_fraction", row.get("model_b2_row_fraction")) or 0.0),
                prefill_b2=float(row.get("model_prefill_b2_row_fraction") or 0.0),
                acoustic_b2_work_fraction=float(row.get("acoustic_b2_work_fraction") or 0.0),
                mixed_padding_complete_count=row.get("mixed_padding_complete_count", 0),
                throughput=(f"{float(row['useful_audio_throughput']):.6f}" if row.get("useful_audio_throughput") is not None else "NA"),
                span=(f"{float(row['session_span_s']):.3f}" if row.get("session_span_s") is not None else "NA"),
            )
        )
    lines.extend([
        "",
        "## Paired descriptive ratios",
        "",
        "Ratios are descriptive unless the work fingerprints and live timeline are equalizable. A ratio above one means the right-hand system has higher useful-audio throughput (or, equivalently for the span ratio, the left-hand span is longer).",
        "",
        "| workload | N | repeat | comparison | throughput ratio | span ratio | work comparability |",
        "|---|---:|---:|---|---:|---:|---|",
    ])
    for row in ratios:
        lines.append(
            "| {workload} | {N} | {repeat} | {comparison} | {throughput} | {span} | {work_comparability} |".format(
                workload=row["workload"], N=row["N"], repeat=row["repeat"], comparison=row["comparison"],
                throughput=(f"{row['useful_audio_throughput_ratio']:.4f}" if row.get("useful_audio_throughput_ratio") is not None else "NA"),
                span=(f"{row['session_span_ratio']:.4f}" if row.get("session_span_ratio") is not None else "NA"),
                work_comparability=row["work_comparability"],
            )
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- `model decode B2` is measured from steady-state model-engine dispatch events; initial prefill is reported separately and is not counted as decode batching. `acoustic B2` is measured from completed Flow batches.",
        "- A `joint B22` observation requires a model B2 and acoustic B2 event within the bounded host-time correlation window; it does not imply the same request pair or simultaneous GPU execution.",
        "- Work comparability is a case-level multiset comparison of terminal token fingerprints with ephemeral request UUIDs removed. `DIVERGENT` ratios are descriptive, not causal.",
        "- Attempts predating client timing fields may have timing derived from retained SSE metadata. Missing sample-rate metadata uses the fixed 24-kHz acoustic contract for throughput accounting.",
        "- This artifact does not establish a paper claim by itself; it is the evidence table for the fixed-work and held-out analyses.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"metrics": metrics_path, "schedule": schedule_path, "report": report_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    paths = write_reports(args.result_root)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
