"""Bounded metadata helpers for the joint execution experiment.

No helper in this module stores tensors, PCM, CUDA events, or model objects.
It is intentionally usable in unit tests without importing vLLM or CUDA.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import struct
import threading
from typing import Any, Mapping, Sequence


def _ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


class JointEventCorrelator:
    """Correlate model and acoustic batch events by host monotonic time.

    Events are never merged merely because they occurred in the same run.
    A pair is a joint observation only when the two batch events are within the
    configured window.  Unmatched events remain in the snapshot so that a
    missing acoustic or model batch is visible rather than silently dropped.
    """

    def __init__(self, *, join_window_ns: int = 50_000_000, max_events: int = 8192) -> None:
        if isinstance(join_window_ns, bool) or int(join_window_ns) < 0:
            raise ValueError("join_window_ns must be non-negative")
        if isinstance(max_events, bool) or int(max_events) <= 0:
            raise ValueError("max_events must be positive")
        self.join_window_ns = int(join_window_ns)
        self.max_events = int(max_events)
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._counter = 0

    @staticmethod
    def _kind(event_type: str) -> str | None:
        upper = str(event_type).upper()
        if upper.startswith("MODEL"):
            return "model"
        if upper.startswith("FLOW") or upper.startswith("ACOUSTIC"):
            return "acoustic"
        return None

    def record(self, event_type: str, *, timestamp_monotonic_ns: int, **payload: Any) -> dict[str, Any]:
        kind = self._kind(event_type)
        if kind is None:
            raise ValueError(f"unsupported joint event type: {event_type!r}")
        try:
            timestamp = int(timestamp_monotonic_ns)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp_monotonic_ns must be an integer") from exc
        event: dict[str, Any] = {
            "event_id": f"joint-event-{self._counter}",
            "event_type": str(event_type),
            "kind": kind,
            "timestamp_monotonic_ns": timestamp,
        }
        self._counter += 1
        event.update({str(key): value for key, value in payload.items()})
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_events:
                del self._events[: len(self._events) - self.max_events]
        return dict(event)

    @staticmethod
    def _batch_size(event: Mapping[str, Any], kind: str) -> int:
        key = "model_batch_size" if kind == "model" else "acoustic_batch_size"
        value = event.get(key, event.get("batch_size", 0))
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _critical(event: Mapping[str, Any]) -> bool:
        return bool(event.get("critical_path_flag", event.get("critical_path", False)))

    def _rows_locked(self) -> list[dict[str, Any]]:
        models = [event for event in self._events if event["kind"] == "model"]
        acoustics = [event for event in self._events if event["kind"] == "acoustic"]
        used_acoustic: set[str] = set()
        pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for model in models:
            candidates = [
                acoustic
                for acoustic in acoustics
                if acoustic["event_id"] not in used_acoustic
                and abs(
                    int(model["timestamp_monotonic_ns"])
                    - int(acoustic["timestamp_monotonic_ns"])
                ) <= self.join_window_ns
            ]
            if candidates:
                acoustic = min(
                    candidates,
                    key=lambda item: (
                        abs(
                            int(model["timestamp_monotonic_ns"])
                            - int(item["timestamp_monotonic_ns"])
                        ),
                        int(item["timestamp_monotonic_ns"]),
                    ),
                )
                used_acoustic.add(acoustic["event_id"])
                pairs.append((model, acoustic))
            else:
                pairs.append((model, None))
        for acoustic in acoustics:
            if acoustic["event_id"] not in used_acoustic:
                pairs.append((acoustic, None))

        rows: list[dict[str, Any]] = []
        for left, right in pairs:
            if left["kind"] == "model":
                model, acoustic = left, right
            else:
                model, acoustic = right, left
            model_ids = _ids(model.get("scheduled_request_ids")) if model else ()
            acoustic_ids = _ids(acoustic.get("scheduled_request_ids")) if acoustic else ()
            model_batch = self._batch_size(model, "model") if model else 0
            acoustic_batch = self._batch_size(acoustic, "acoustic") if acoustic else 0
            timestamps = [
                int(item["timestamp_monotonic_ns"])
                for item in (model, acoustic)
                if item is not None
            ]
            within_window = (
                model is not None
                and acoustic is not None
                and abs(int(model["timestamp_monotonic_ns"]) - int(acoustic["timestamp_monotonic_ns"]))
                <= self.join_window_ns
            )
            rows.append(
                {
                    "model_event_id": model.get("event_id") if model else None,
                    "acoustic_event_id": acoustic.get("event_id") if acoustic else None,
                    "timestamp_monotonic_ns": min(timestamps) if timestamps else 0,
                    "model_batch_size": model_batch,
                    "acoustic_batch_size": acoustic_batch,
                    "model_request_ids": model_ids,
                    "acoustic_request_ids": acoustic_ids,
                    "same_request_pair": bool(model_ids and acoustic_ids and model_ids == acoustic_ids),
                    "joint_2x2": bool(within_window and model_batch >= 2 and acoustic_batch >= 2),
                    "within_join_window": within_window,
                    "critical_path": bool(
                        self._critical(model) if model else False
                    ) or bool(self._critical(acoustic) if acoustic else False),
                    "joint_critical_path": bool(
                        within_window
                        and model is not None
                        and acoustic is not None
                        and self._critical(model)
                        and self._critical(acoustic)
                    ),
                }
            )
        rows.sort(key=lambda row: (int(row["timestamp_monotonic_ns"]), str(row["model_event_id"] or ""), str(row["acoustic_event_id"] or "")))
        return rows

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._rows_locked()]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class JointWorkFingerprint:
    """Order-sensitive, metadata-only work fingerprint for one request."""

    def __init__(self, request_id: str, generation_id: int) -> None:
        self.request_id = str(request_id)
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if isinstance(generation_id, bool) or int(generation_id) < 0:
            raise ValueError("generation_id must be non-negative")
        self.generation_id = int(generation_id)
        self._hasher = hashlib.sha256()
        self._last_sequence: dict[str, int] = defaultdict(lambda: -1)
        self.model_round_count = 0
        self.acoustic_chunk_count = 0
        self.flow_step_count = 0
        self._model_token_count = 0
        self._acoustic_token_count = 0

    @staticmethod
    def _int(value: Any, field: str, *, nonnegative: bool = True) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc
        if nonnegative and result < 0:
            raise ValueError(f"{field} must be non-negative")
        return result

    def _begin(self, category: str, sequence_no: Any) -> int:
        sequence = self._int(sequence_no, "sequence_no")
        if sequence <= self._last_sequence[category]:
            raise ValueError(f"{category} sequence_no must increase strictly")
        self._last_sequence[category] = sequence
        return sequence

    def _update(self, category: str, sequence: int, *values: int) -> None:
        encoded_category = category.encode("ascii")
        self._hasher.update(struct.pack(">I", len(encoded_category)))
        self._hasher.update(encoded_category)
        self._hasher.update(struct.pack(">Q", int(sequence)))
        for value in values:
            self._hasher.update(struct.pack(">Q", self._int(value, "work value")))

    def add_model_round(
        self, sequence_no: int, text_token: int, stoken_token: int, control_token: int
    ) -> None:
        sequence = self._begin("model", sequence_no)
        values = (
            self._int(text_token, "text_token"),
            self._int(stoken_token, "stoken_token"),
            self._int(control_token, "control_token"),
        )
        self._update("model", sequence, *values)
        self.model_round_count += 1
        self._model_token_count += 1

    def add_acoustic_chunk(
        self, sequence_no: int, token_count: int, sample_count: int, sample_rate: int
    ) -> None:
        sequence = self._begin("acoustic", sequence_no)
        values = (
            self._int(token_count, "token_count"),
            self._int(sample_count, "sample_count"),
            self._int(sample_rate, "sample_rate"),
        )
        self._update("acoustic", sequence, *values)
        self.acoustic_chunk_count += 1
        self._acoustic_token_count += values[0]

    def add_flow_steps(self, sequence_no: int, step_count: int) -> None:
        sequence = self._begin("flow", sequence_no)
        count = self._int(step_count, "step_count")
        self._update("flow", sequence, count)
        self.flow_step_count += count

    def finalize(
        self,
        *,
        termination_reason: str | None = None,
        flush_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "generation_id": self.generation_id,
            "model_round_count": int(self.model_round_count),
            "acoustic_chunk_count": int(self.acoustic_chunk_count),
            "flow_step_count": int(self.flow_step_count),
            "model_token_count": int(self._model_token_count),
            "acoustic_token_count": int(self._acoustic_token_count),
            "work_digest": self._hasher.hexdigest(),
            "work_fingerprint_equalizable": True,
            "termination_reason": termination_reason,
            "flush_reason": flush_reason,
        }


class JointIdentityFence:
    """Request/generation/sequence fence used by joint telemetry adapters."""

    def __init__(self) -> None:
        self._generation: dict[str, int] = {}
        self._last_committed: dict[str, int] = defaultdict(lambda: -1)
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _generation_value(value: Any) -> int:
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError("generation_id must be non-negative")
        return int(value)

    def register(self, request_id: str, generation_id: int = 0) -> None:
        key = str(request_id)
        if not key:
            raise ValueError("request_id must be non-empty")
        generation = self._generation_value(generation_id)
        with self._lock:
            if key in self._generation:
                raise ValueError(f"request already registered: {key}")
            self._generation[key] = generation
            self._last_committed[key] = -1
            self._cancelled.discard(key)

    def reset(self, request_id: str, generation_id: int) -> None:
        key = str(request_id)
        generation = self._generation_value(generation_id)
        with self._lock:
            old = self._generation.get(key)
            if old is None:
                raise ValueError(f"request is not registered: {key}")
            if generation <= old:
                raise ValueError("reset generation must increase")
            self._generation[key] = generation
            self._last_committed[key] = -1
            self._cancelled.discard(key)

    def cancel(self, request_id: str, generation_id: int) -> None:
        key = str(request_id)
        generation = self._generation_value(generation_id)
        with self._lock:
            if self._generation.get(key) != generation:
                return
            self._cancelled.add(key)

    def accepts(self, request_id: str, generation_id: int, sequence_no: int) -> bool:
        key = str(request_id)
        generation = self._generation_value(generation_id)
        sequence = self._int_sequence(sequence_no)
        with self._lock:
            return (
                self._generation.get(key) == generation
                and key not in self._cancelled
                and sequence > self._last_committed[key]
            )

    @staticmethod
    def _int_sequence(value: Any) -> int:
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError("sequence_no must be non-negative")
        return int(value)

    def commit(self, request_id: str, generation_id: int, sequence_no: int) -> bool:
        key = str(request_id)
        generation = self._generation_value(generation_id)
        sequence = self._int_sequence(sequence_no)
        with self._lock:
            if (
                self._generation.get(key) != generation
                or key in self._cancelled
                or sequence <= self._last_committed[key]
            ):
                raise ValueError("joint identity fence rejected commit")
            self._last_committed[key] = sequence
            return True

    def unregister(self, request_id: str) -> None:
        key = str(request_id)
        with self._lock:
            self._generation.pop(key, None)
            self._last_committed.pop(key, None)
            self._cancelled.discard(key)


__all__ = [
    "JointEventCorrelator",
    "JointIdentityFence",
    "JointWorkFingerprint",
]
