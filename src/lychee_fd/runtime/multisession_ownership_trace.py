"""Observation-only ownership records and validation for multi-session runs.

The validator intentionally treats physical rows as metadata.  The expected
owner for every downstream boundary is the logical request identity.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


SCHEMA = "lychee-e2e-request-ownership-v1"
OWNER_FIELDS = (
    "row_state_owner",
    "row_side_output_owner",
    "sampler_owner",
    "stoken_owner",
    "streaming_decoder_owner",
    "token2wav_owner",
    "pcm_owner",
    "playback_owner",
)
_WRITE_LOCK = threading.Lock()
_CONTEXT_LOCK = threading.Lock()
_REQUEST_CONTEXT: dict[str, dict[str, str]] = {}


def write_ownership_event(
    event_type: str,
    *,
    path: Optional[str] = None,
    source_file: Optional[str] = None,
    source_function: Optional[str] = None,
    **fields: Any,
) -> bool:
    """Append a JSONL ownership event without affecting serving behavior."""
    target = str(path or os.getenv("LYCHEEFD_MULTISESSION_OWNERSHIP_TRACE_PATH", "")).strip()
    if not target:
        return False
    record = {
        "schema": SCHEMA,
        "event_type": str(event_type),
        "timestamp_monotonic_ns": int(time.monotonic_ns()),
        "source_file": source_file,
        "source_function": source_function,
    }
    record.update(fields)
    try:
        encoded = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":"))
        with _WRITE_LOCK:
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
        return True
    except Exception:
        return False


def register_request_context(
    logical_request_id: str,
    *,
    run_id: str,
    session_id: str,
) -> None:
    """Register the stable run/session mapping used by model-side probes."""
    with _CONTEXT_LOCK:
        _REQUEST_CONTEXT[str(logical_request_id)] = {
            "run_id": str(run_id),
            "session_id": str(session_id),
        }


def unregister_request_context(logical_request_id: str) -> None:
    with _CONTEXT_LOCK:
        _REQUEST_CONTEXT.pop(str(logical_request_id), None)


def request_context(logical_request_id: Any) -> dict[str, Optional[str]]:
    with _CONTEXT_LOCK:
        context = dict(_REQUEST_CONTEXT.get(str(logical_request_id), {}))
    return {
        "run_id": context.get("run_id"),
        "session_id": context.get("session_id"),
    }


def _divergence(record: Mapping[str, Any], boundary: str, expected: Any, actual: Any, detail: str = "") -> dict:
    return {
        "run_id": record.get("run_id"),
        "session_id": record.get("session_id"),
        "logical_request_id": record.get("logical_request_id"),
        "execution_step": record.get("execution_step"),
        "physical_row": record.get("physical_row"),
        "previous_owner": record.get("previous_owner"),
        "expected_owner": expected,
        "actual_owner": actual,
        "boundary": boundary,
        "source_file": record.get("source_file"),
        "source_function": record.get("source_function"),
        "state_key": record.get("state_key"),
        "pcm_chunk_id": record.get("pcm_chunk_id"),
        "detail": detail,
    }


def validate_ownership_trace(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate ownership identity, timestamps, and PCM sequence invariants."""
    last_timestamp: dict[str, int] = {}
    seen_chunks: dict[str, Mapping[str, Any]] = {}
    last_sequence: dict[str, int] = {}
    count = 0

    for raw_record in records:
        record = dict(raw_record)
        count += 1
        session_id = str(record.get("session_id") or "")
        logical_id = str(record.get("logical_request_id") or "")
        if not session_id or not logical_id:
            return {
                "status": "FAIL",
                "record_count": count,
                "first_divergence": _divergence(record, "request_identity", logical_id or "non-empty", logical_id, "missing request identity"),
            }

        try:
            timestamp = int(record.get("timestamp_monotonic_ns"))
        except (TypeError, ValueError):
            return {
                "status": "FAIL",
                "record_count": count,
                "first_divergence": _divergence(record, "timestamp_monotonic_ns", "integer", record.get("timestamp_monotonic_ns"), "invalid timestamp"),
            }
        prior_timestamp = last_timestamp.get(session_id)
        if prior_timestamp is not None and timestamp < prior_timestamp:
            return {
                "status": "FAIL",
                "record_count": count,
                "first_divergence": _divergence(record, "timestamp_monotonic_ns", prior_timestamp, timestamp, "session timestamp decreased"),
            }
        last_timestamp[session_id] = timestamp

        try:
            batch_size = int(record.get("physical_batch_size"))
        except (TypeError, ValueError):
            batch_size = 0
        if batch_size <= 0:
            return {
                "status": "FAIL",
                "record_count": count,
                "first_divergence": _divergence(record, "physical_batch_size", ">=1", record.get("physical_batch_size"), "missing physical batch evidence"),
            }

        for field in OWNER_FIELDS:
            actual = record.get(field)
            if actual is None or actual == "":
                continue
            if str(actual) != logical_id:
                return {
                    "status": "FAIL",
                    "record_count": count,
                    "first_divergence": _divergence(record, field, logical_id, actual),
                }

        chunk_id = record.get("pcm_chunk_id")
        if chunk_id not in (None, ""):
            chunk_key = str(chunk_id)
            if chunk_key in seen_chunks:
                return {
                    "status": "FAIL",
                    "record_count": count,
                    "first_divergence": _divergence(record, "pcm_chunk_id", "unique", chunk_key, "duplicate PCM chunk ID"),
                }
            seen_chunks[chunk_key] = record
            try:
                sequence = int(record.get("pcm_sequence_index"))
            except (TypeError, ValueError):
                return {
                    "status": "FAIL",
                    "record_count": count,
                    "first_divergence": _divergence(record, "pcm_sequence_index", "integer", record.get("pcm_sequence_index"), "invalid PCM sequence index"),
                }
            if sequence < 0:
                return {
                    "status": "FAIL",
                    "record_count": count,
                    "first_divergence": _divergence(record, "pcm_sequence_index", ">=0", sequence, "negative PCM sequence index"),
                }
            previous_sequence = last_sequence.get(logical_id)
            if previous_sequence is not None and sequence <= previous_sequence:
                return {
                    "status": "FAIL",
                    "record_count": count,
                    "first_divergence": _divergence(record, "pcm_sequence_index", f">{previous_sequence}", sequence, "PCM sequence did not increase"),
                }
            last_sequence[logical_id] = sequence

    return {"status": "PASS", "record_count": count, "first_divergence": None}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows
