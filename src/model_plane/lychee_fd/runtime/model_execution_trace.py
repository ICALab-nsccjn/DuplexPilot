"""Fail-open, metadata-only tracing for the model execution audit.

The recorder is intentionally independent from the serving scheduler.  It is
enabled only when a path is supplied (normally through
``LYCHEEFD_MODEL_EXECUTION_TRACE_PATH``), and it never serializes tensor
contents or changes lock ownership.  The resulting records are suitable for
the Phase-0 critical-path profile, not for a performance-counted run.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any, Iterator, Mapping, Sequence


TRACE_SCHEMA = "lychee-model-execution-v1"
TRACE_ENV = "LYCHEEFD_MODEL_EXECUTION_TRACE_PATH"
_MAX_ITEMS = 64
_MAX_DEPTH = 4


class ModelWorkFingerprint:
    """Bounded, order-sensitive fingerprint of generated model work.

    The accumulator deliberately stores only a SHA-256 state and a token
    count.  It never retains token arrays or model outputs, so it is safe to
    use on an online diagnostic path.  Sequence numbers are included in the
    digest to detect reordering as well as differing token values.
    """

    def __init__(self) -> None:
        self._hasher = hashlib.sha256()
        self._last_sequence_no = -1
        self.generated_token_count = 0

    @staticmethod
    def _nonnegative_int(value: Any, field: str) -> int:
        if value is None or isinstance(value, bool):
            raise ValueError(f"{field} must be a non-negative integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a non-negative integer") from exc
        if result < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return result

    def update(
        self,
        *,
        sequence_no: int,
        text_token: int,
        stoken_token: int,
        control_token: int,
    ) -> None:
        sequence = self._nonnegative_int(sequence_no, "sequence_no")
        if sequence <= self._last_sequence_no:
            raise ValueError("sequence_no must increase strictly")
        text = self._nonnegative_int(text_token, "text_token")
        stoken = self._nonnegative_int(stoken_token, "stoken_token")
        control = self._nonnegative_int(control_token, "control_token")
        # Fixed-width encoding makes the digest independent of Python's
        # textual formatting and preserves exact tuple boundaries.
        self._hasher.update(struct.pack(">QQQQ", sequence, text, stoken, control))
        self._last_sequence_no = sequence
        self.generated_token_count += 1

    def digest(self) -> str:
        return self._hasher.hexdigest()

    def summary(
        self,
        *,
        request_id: str,
        generation_id: int,
        termination_reason: str | None,
        request_finished_reason: str | None,
    ) -> dict[str, Any]:
        generation = self._nonnegative_int(generation_id, "generation_id")
        return {
            "request_id": str(request_id),
            "generation_id": generation,
            "generated_token_count": int(self.generated_token_count),
            "token_sequence_sha256": self.digest(),
            "termination_reason": (
                str(termination_reason) if termination_reason is not None else None
            ),
            "request_finished_reason": (
                str(request_finished_reason)
                if request_finished_reason is not None
                else None
            ),
        }


def _tensor_metadata(value: Any) -> dict[str, Any] | None:
    """Describe tensor-like values without reading their data."""
    shape = getattr(value, "shape", None)
    if shape is None or not hasattr(value, "dtype"):
        return None
    try:
        normalized_shape = [int(item) for item in shape]
    except (TypeError, ValueError):
        normalized_shape = []
    try:
        numel = int(value.numel()) if callable(getattr(value, "numel", None)) else None
    except (TypeError, ValueError, RuntimeError):
        numel = None
    return {
        "container": "tensor",
        "shape": normalized_shape,
        "dtype": str(getattr(value, "dtype", "")),
        "device": str(getattr(value, "device", "")),
        "numel": numel,
    }


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    """Convert metadata to bounded JSON without invoking tensor ``repr``."""
    tensor = _tensor_metadata(value)
    if tensor is not None:
        return tensor
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= _MAX_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS:
                break
            result[str(key)] = _safe_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _safe_value(item, depth=depth + 1)
            for item in list(value)[:_MAX_ITEMS]
        ]
    # Do not use repr: user/model objects can expose huge or data-bearing
    # representations.  A short type marker is enough for an audit record.
    return f"<{type(value).__name__}>"


class ModelExecutionTraceRecorder:
    """Append bounded model execution events, failing open on I/O errors."""

    def __init__(self, path: str | os.PathLike[str] | None = None,
                 *, clock_ns=None) -> None:
        raw_path = str(path or "").strip()
        self.path = Path(raw_path) if raw_path else None
        self._clock_ns = clock_ns or time.monotonic_ns
        self._write_lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "ModelExecutionTraceRecorder":
        return cls(os.getenv(TRACE_ENV, ""))

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(self, event_type: str, **payload: Any) -> bool:
        if not self.enabled:
            return False
        record: dict[str, Any] = {
            "schema": TRACE_SCHEMA,
            "event_type": str(event_type),
            "timestamp_monotonic_ns": int(self._clock_ns()),
            "pid": int(os.getpid()),
        }
        record.update({str(key): _safe_value(value) for key, value in payload.items()})
        try:
            encoded = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            assert self.path is not None
            with self._write_lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
            return True
        except Exception:
            # Diagnostics must never turn a model request into an inference
            # failure (for example, when a worker exits before its trace path
            # is available).
            return False

    @contextmanager
    def lock_scope(
        self,
        lock: Any,
        *,
        request_ids: Sequence[str] = (),
        lock_name: str = "_stream_lock",
        protected_operation: str = "multi_head_generate_stream",
    ) -> Iterator[None]:
        """Record wait/hold time while preserving the supplied lock exactly."""
        wait_start = int(self._clock_ns())
        lock.acquire()
        acquired = int(self._clock_ns())
        try:
            yield
        finally:
            released = int(self._clock_ns())
            lock.release()
            self.record(
                "MODEL_LOCK",
                lock=lock_name,
                protected_operation=protected_operation,
                request_ids=tuple(str(item) for item in request_ids),
                wait_ns=max(0, acquired - wait_start),
                hold_ns=max(0, released - acquired),
            )

    @contextmanager
    def span(self, event_type: str, **payload: Any) -> Iterator[None]:
        """Record a start/end pair with a host-clock duration."""
        start = int(self._clock_ns())
        self.record(f"{event_type}_START", **payload)
        try:
            yield
        finally:
            end = int(self._clock_ns())
            self.record(
                f"{event_type}_END",
                **payload,
                duration_ns=max(0, end - start),
            )


__all__ = [
    "ModelWorkFingerprint",
    "ModelExecutionTraceRecorder",
    "TRACE_ENV",
    "TRACE_SCHEMA",
]
