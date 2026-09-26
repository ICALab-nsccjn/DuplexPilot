"""Opt-in, fail-safe stage profiling for APR runtime diagnostics."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from profiling.apr_timeline import TimelineSink


PROFILE_SCHEMA_VERSION = "apr-profile-v1"

PROFILE_STAGES = frozenset(
    {
        "REQUEST_ADMISSION",
        "MODEL_FORWARD",
        "TOKEN_TRANSFER",
        "APR_SCHEDULE_WAIT",
        "STATE_ACQUIRE",
        "STATE_RESTORE",
        "BACKEND_PROCESS",
        "STATE_CAPTURE",
        "STATE_COMMIT",
        "PCM_COMMIT",
        "PCM_EGRESS",
        "CLEANUP",
        "STREAMING_DECODER",
        "TOKEN2WAV_TOKEN_QUEUE",
        "TOKEN2WAV_FLOW",
        "TOKEN2WAV_HIFT",
        "TOKEN2WAV_VOCODER",
        "PCM_GENERATION",
    }
)

SESSION_SCOPED_STAGES = PROFILE_STAGES - {
    "REQUEST_ADMISSION",
    "MODEL_FORWARD",
    "TOKEN_TRANSFER",
    "CLEANUP",
}


class ProfileContractError(ValueError):
    """Raised when a profiling event violates the versioned contract."""


def estimate_state_size_bytes(value: Any, _seen: set[int] | None = None) -> int:
    """Estimate materialized logical-state bytes without serializing it."""
    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if value is None or isinstance(value, (bool, int, float, str)):
        return sys.getsizeof(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            return int(value.numel()) * int(value.element_size())
        except Exception:
            return sys.getsizeof(value)
    if isinstance(value, Mapping):
        return sys.getsizeof(value) + sum(
            estimate_state_size_bytes(key, seen) + estimate_state_size_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sys.getsizeof(value) + sum(estimate_state_size_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        try:
            return sys.getsizeof(value) + estimate_state_size_bytes(vars(value), seen)
        except Exception:
            return sys.getsizeof(value)
    return sys.getsizeof(value)


@dataclass(frozen=True)
class StageSpan:
    run_id: str
    system: str
    workload: str
    concurrency: int
    repeat: int
    session_id: str | None
    generation_id: int | None
    sequence_no: int | None
    state_version: int | None
    worker_id: int | None
    stage: str
    start_monotonic_ns: int
    end_monotonic_ns: int
    queue_depth: int | None = None
    state_size_bytes: int | None = None
    error_type: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["profile_schema_version"] = PROFILE_SCHEMA_VERSION
        record["event_type"] = record.pop("stage")
        record["duration_ns"] = self.end_monotonic_ns - self.start_monotonic_ns
        return record


class StageProfiler:
    """Collect spans in memory and never turn trace loss into serving failure."""

    def __init__(
        self,
        *,
        enabled: bool,
        run_context: Mapping[str, Any],
        sink: Callable[[dict[str, Any]], Any] | None = None,
        timeline: TimelineSink | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ProfileContractError("enabled must be a boolean")
        required = ("run_id", "system", "workload", "concurrency", "repeat")
        missing = [key for key in required if key not in run_context]
        if missing:
            raise ProfileContractError(
                f"run context is missing fields: {', '.join(missing)}"
            )
        self.enabled = enabled
        self.run_context = dict(run_context)
        self.sink = sink
        if timeline is not None and not isinstance(timeline, TimelineSink):
            raise ProfileContractError("timeline must be a TimelineSink")
        self.timeline = timeline
        self.spans: list[StageSpan] = []

    def _validate_stage(self, stage: str, session_id: str | None) -> None:
        if stage not in PROFILE_STAGES:
            raise ProfileContractError(f"unknown profiling stage: {stage}")
        if stage in SESSION_SCOPED_STAGES and not session_id:
            raise ProfileContractError(f"{stage} requires session_id")

    @contextmanager
    def span(
        self,
        stage: str,
        *,
        session_id: str | None = None,
        generation_id: int | None = None,
        sequence_no: int | None = None,
        state_version: int | None = None,
        worker_id: int | None = None,
        queue_depth: int | None = None,
        state_size_bytes: int | None = None,
    ) -> Iterator[StageSpan | None]:
        if not self.enabled:
            yield None
            return
        self._validate_stage(stage, session_id)
        start = time.monotonic_ns()
        error_type = None
        try:
            yield None
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            span = StageSpan(
                run_id=str(self.run_context["run_id"]),
                system=str(self.run_context["system"]),
                workload=str(self.run_context["workload"]),
                concurrency=int(self.run_context["concurrency"]),
                repeat=int(self.run_context["repeat"]),
                session_id=session_id,
                generation_id=generation_id,
                sequence_no=sequence_no,
                state_version=state_version,
                worker_id=worker_id,
                stage=stage,
                start_monotonic_ns=start,
                end_monotonic_ns=time.monotonic_ns(),
                queue_depth=queue_depth,
                state_size_bytes=state_size_bytes,
                error_type=error_type,
            )
            self._append(span)

    def _append(self, span: StageSpan) -> None:
        self.spans.append(span)
        self._append_timeline(span)
        if self.sink is None:
            return
        try:
            self.sink(span.to_record())
        except Exception:
            return

    def _append_timeline(self, span: StageSpan) -> None:
        timeline = self.timeline
        if timeline is None or not timeline.enabled:
            return
        common = {
            "session_id": span.session_id,
            "generation_id": span.generation_id,
            "sequence_no": span.sequence_no,
            "state_version": span.state_version,
            "worker_id": span.worker_id,
            "state_size_bytes": span.state_size_bytes,
        }
        try:
            if span.stage == "REQUEST_ADMISSION":
                timeline.emit("request_arrival", timestamp_monotonic_ns=span.start_monotonic_ns, **common)
            elif span.stage == "MODEL_FORWARD":
                timeline.emit("model_start", timestamp_monotonic_ns=span.start_monotonic_ns, **common)
                timeline.emit("model_step", timestamp_monotonic_ns=span.end_monotonic_ns, duration_ns=span.end_monotonic_ns - span.start_monotonic_ns, **common)
                timeline.emit("model_end", timestamp_monotonic_ns=span.end_monotonic_ns, duration_ns=span.end_monotonic_ns - span.start_monotonic_ns, **common)
            elif span.stage == "TOKEN_TRANSFER":
                timeline.emit("token_ready", timestamp_monotonic_ns=span.end_monotonic_ns, duration_ns=span.end_monotonic_ns - span.start_monotonic_ns, **common)
            elif span.stage == "STATE_RESTORE":
                timeline.emit("restore_start", timestamp_monotonic_ns=span.start_monotonic_ns, **common)
                timeline.emit("restore_end", timestamp_monotonic_ns=span.end_monotonic_ns, duration_ns=span.end_monotonic_ns - span.start_monotonic_ns, **common)
            elif span.stage == "STATE_CAPTURE":
                timeline.emit("APR_checkpoint_start", timestamp_monotonic_ns=span.start_monotonic_ns, **common)
                timeline.emit("APR_checkpoint_end", timestamp_monotonic_ns=span.end_monotonic_ns, duration_ns=span.end_monotonic_ns - span.start_monotonic_ns, **common)
            elif span.stage in {"TOKEN2WAV_TOKEN_QUEUE", "TOKEN2WAV_FLOW", "TOKEN2WAV_HIFT", "TOKEN2WAV_VOCODER"}:
                timeline.emit("token2wav_start", timestamp_monotonic_ns=span.start_monotonic_ns, **common)
                timeline.emit("token2wav_end", timestamp_monotonic_ns=span.end_monotonic_ns, duration_ns=span.end_monotonic_ns - span.start_monotonic_ns, **common)
            elif span.stage in {"PCM_GENERATION", "PCM_EGRESS"}:
                timeline.emit("pcm_ready", timestamp_monotonic_ns=span.end_monotonic_ns, duration_ns=span.end_monotonic_ns - span.start_monotonic_ns, **common)
            elif span.stage == "CLEANUP":
                timeline.emit("request_finish", timestamp_monotonic_ns=span.end_monotonic_ns, duration_ns=span.end_monotonic_ns - span.start_monotonic_ns, **common)
        except Exception:
            return

    def record(self, span: StageSpan) -> None:
        if not isinstance(span, StageSpan):
            raise ProfileContractError("record requires a StageSpan")
        self._append(span)

    def record_external_span(
        self,
        stage: str,
        *,
        start_monotonic_ns: int,
        end_monotonic_ns: int,
        session_id: str | None = None,
        generation_id: int | None = None,
        sequence_no: int | None = None,
        state_version: int | None = None,
        worker_id: int | None = None,
        state_size_bytes: int | None = None,
    ) -> None:
        """Accept timing from an instrumented backend without changing it."""
        if not self.enabled:
            return
        self._validate_stage(stage, session_id)
        self._append(
            StageSpan(
                run_id=str(self.run_context["run_id"]),
                system=str(self.run_context["system"]),
                workload=str(self.run_context["workload"]),
                concurrency=int(self.run_context["concurrency"]),
                repeat=int(self.run_context["repeat"]),
                session_id=session_id,
                generation_id=generation_id,
                sequence_no=sequence_no,
                state_version=state_version,
                worker_id=worker_id,
                stage=stage,
                start_monotonic_ns=int(start_monotonic_ns),
                end_monotonic_ns=int(end_monotonic_ns),
                state_size_bytes=state_size_bytes,
            )
        )

    def validate(self) -> None:
        for span in self.spans:
            if span.stage not in PROFILE_STAGES:
                raise ProfileContractError(f"unknown span stage: {span.stage}")
            if span.end_monotonic_ns < span.start_monotonic_ns:
                raise ProfileContractError("span timestamps are not monotonic")
            self._validate_stage(span.stage, span.session_id)

    def flush_jsonl(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for span in self.spans:
                handle.write(json.dumps(span.to_record(), sort_keys=True))
                handle.write("\n")
