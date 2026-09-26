"""Fail-safe, versioned JSONL observability for APR progression."""

from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping

from .profiling import StageProfiler


class APRTraceError(ValueError):
    """Raised when an APR trace event violates the trace contract."""


APR_TRACE_EVENT_TYPES = (
    "APR_TOKEN_ENQUEUED",
    "APR_TOKEN_DEQUEUED",
    "APR_QUEUE_DEPTH",
    "APR_STATE_ACQUIRE",
    "APR_STATE_RESTORE",
    "APR_ACOUSTIC_SCHEDULE",
    "APR_ACOUSTIC_START",
    "APR_ACOUSTIC_END",
    "APR_STATE_COMMIT",
    "APR_PCM_COMMIT",
    "APR_BACKPRESSURE",
    "APR_CANCEL",
    "APR_STALE_OUTPUT",
    "APR_ERROR",
    "APR_WORKER_ACQUIRE",
    "APR_WORKER_RELEASE",
    "APR_WORKER_ERROR",
)


class APRTrace:
    """Write structured APR events without changing the serving result."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        sink: Callable[[dict[str, Any]], Any] | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if path is not None and sink is not None:
            raise APRTraceError("APRTrace accepts either path or sink, not both")
        self._path = Path(path) if path is not None else None
        self._sink = sink
        if profiler is not None and not isinstance(profiler, StageProfiler):
            raise APRTraceError("profiler must be a StageProfiler")
        self._profiler = profiler
        self._lock = threading.RLock()

    def profile_span(self, stage: str, **fields: Any):
        """Return a no-op span when profiling is not configured."""
        if self._profiler is None:
            return nullcontext()
        return self._profiler.span(stage, **fields)

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any] | None:
        if event_type not in APR_TRACE_EVENT_TYPES:
            raise APRTraceError(f"unknown APR event type: {event_type}")
        record = dict(fields)
        record["event_type"] = event_type
        record["timestamp_monotonic_ns"] = time.monotonic_ns()
        try:
            with self._lock:
                if self._path is not None:
                    with self._path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                        output.write("\n")
                if self._sink is not None:
                    self._sink(dict(record))
        except Exception:
            # Trace loss must never turn into a serving failure.
            return None
        return record

    def record(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        if not isinstance(event, Mapping):
            raise APRTraceError("APR trace record requires a mapping")
        event_fields = dict(event)
        event_type = event_fields.pop("event_type", None)
        if not isinstance(event_type, str):
            raise APRTraceError("APR trace record requires event_type")
        event_fields.pop("timestamp_monotonic_ns", None)
        return self.emit(event_type, **event_fields)
