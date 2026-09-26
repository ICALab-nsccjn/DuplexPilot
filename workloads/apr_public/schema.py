"""Canonical JSONL schema for public full-duplex workload traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

CANONICAL_EVENT_TYPES = (
    "arrival",
    "audio_chunk",
    "barge_in",
    "resume",
    "cancel",
    "finish",
)
_EVENT_PRIORITY = {event: index for index, event in enumerate(CANONICAL_EVENT_TYPES)}


@dataclass(frozen=True)
class PublicTraceManifest:
    schema_version: str
    trace_id: str
    dataset_revision: str
    source_hash: str
    converter_commit: str
    arrival_window: str
    arrival_scale: float
    scenario_distribution: Mapping[str, int]
    session_count: int

    def __post_init__(self) -> None:
        for name in (
            "schema_version", "trace_id", "dataset_revision", "source_hash",
            "converter_commit", "arrival_window",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != "public-trace-v1":
            raise ValueError("unsupported public trace schema version")
        if isinstance(self.arrival_scale, bool) or self.arrival_scale <= 0:
            raise ValueError("arrival_scale must be positive")
        if isinstance(self.session_count, bool) or self.session_count <= 0:
            raise ValueError("session_count must be positive")
        if not self.scenario_distribution or any(
            not isinstance(name, str) or not name or count <= 0
            for name, count in self.scenario_distribution.items()
        ):
            raise ValueError("scenario_distribution must contain positive counts")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicTraceEvent:
    session_id: str
    source_sample_id: str
    timestamp_s: float
    event_type: str
    scenario: str
    audio_path: str | None
    audio_offset_s: float
    audio_duration_s: float
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.source_sample_id or not self.scenario:
            raise ValueError("session_id, source_sample_id, and scenario are required")
        if self.event_type not in CANONICAL_EVENT_TYPES:
            raise ValueError(f"unsupported canonical event type: {self.event_type}")
        if self.timestamp_s < 0 or self.audio_offset_s < 0 or self.audio_duration_s < 0:
            raise ValueError("event times and durations must be non-negative")
        if self.event_type == "audio_chunk" and not self.audio_path:
            raise ValueError("audio_chunk requires audio_path")
        object.__setattr__(self, "payload", dict(self.payload or {}))

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["payload"] = dict(self.payload or {})
        return value


@dataclass(frozen=True)
class PublicTrace:
    manifest: PublicTraceManifest
    events: tuple[PublicTraceEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise ValueError("events must be an immutable tuple")
        session_ids = {event.session_id for event in self.events}
        if session_ids != {f"session-{i}" for i in range(self.manifest.session_count)}:
            raise ValueError("events must contain exactly the manifest sessions")
        if not self.events:
            raise ValueError("trace must contain events")
        ordered = tuple(
            sorted(
                self.events,
                key=lambda event: (
                    event.timestamp_s,
                    _EVENT_PRIORITY[event.event_type],
                    event.session_id,
                ),
            )
        )
        if ordered != self.events:
            raise ValueError("events must be globally sorted")
        for session_id in sorted(session_ids):
            session_events = [event for event in self.events if event.session_id == session_id]
            if session_events[0].event_type != "arrival":
                raise ValueError(f"session {session_id} is missing arrival")
            if session_events[-1].event_type not in {"finish", "cancel"}:
                raise ValueError(f"session {session_id} is missing finish/cancel")

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "events": [event.as_dict() for event in self.events],
        }


def write_trace_jsonl(path: str | Path, trace: PublicTrace) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "manifest", **trace.manifest.as_dict()}, sort_keys=True) + "\n")
        for event in trace.events:
            handle.write(json.dumps({"record_type": "event", **event.as_dict()}, sort_keys=True) + "\n")


def read_trace_jsonl(path: str | Path) -> PublicTrace:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or records[0].get("record_type") != "manifest":
        raise ValueError("trace must start with a manifest record")
    manifest_data = dict(records[0]); manifest_data.pop("record_type", None)
    events: list[PublicTraceEvent] = []
    for record in records[1:]:
        if record.get("record_type") != "event":
            raise ValueError("trace contains a non-event record after manifest")
        value = dict(record); value.pop("record_type", None)
        events.append(PublicTraceEvent(**value))
    return PublicTrace(PublicTraceManifest(**manifest_data), tuple(events))
