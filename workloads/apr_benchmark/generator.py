"""Pure, deterministic generators for public-inspired APR workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


EVENT_TYPES = (
    "arrival",
    "input_phase",
    "output_phase",
    "barge_in",
    "resume",
    "finish",
)
SUPPORTED_WORKLOADS = ("A", "B", "C")
BASELINES = (
    "original_affinity",
    "apr",
    "apr_no_migration",
    "no_rsv_dsv_apr",
)
_EVENT_PRIORITY = {event_type: index for index, event_type in enumerate(EVENT_TYPES)}
_DURATION_SCALE = {"short": 1.0, "medium": 3.0, "long": 8.0}


@dataclass(frozen=True)
class WorkloadEvent:
    session_id: str
    timestamp_s: float
    event_type: str
    input_phase: str | None = None
    output_phase: str | None = None
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {self.event_type}")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["payload"] = dict(self.payload or {})
        return value


@dataclass(frozen=True)
class WorkloadTrace:
    workload_id: str
    seed: int
    concurrency: int
    events: tuple[WorkloadEvent, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.workload_id not in SUPPORTED_WORKLOADS:
            raise ValueError(f"unsupported workload: {self.workload_id}")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        session_ids = {event.session_id for event in self.events}
        expected = {f"session-{index}" for index in range(self.concurrency)}
        if session_ids != expected:
            raise ValueError(
                f"trace session ids mismatch: expected={expected} got={session_ids}"
            )
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
            raise ValueError("events must be globally sorted by timestamp and priority")

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "seed": self.seed,
            "concurrency": self.concurrency,
            "metadata": dict(self.metadata),
            "events": [event.as_dict() for event in self.events],
        }


def _event(
    session_id: str,
    timestamp_s: float,
    event_type: str,
    *,
    input_phase: str | None = None,
    output_phase: str | None = None,
    **payload: Any,
) -> WorkloadEvent:
    return WorkloadEvent(
        session_id=session_id,
        timestamp_s=round(float(timestamp_s), 6),
        event_type=event_type,
        input_phase=input_phase,
        output_phase=output_phase,
        payload=payload,
    )


def _interaction_events(
    session_id: str,
    arrival_s: float,
    duration_s: float,
    *,
    barge_in: bool,
    turns: int,
) -> list[WorkloadEvent]:
    events = [_event(session_id, arrival_s, "arrival", turn_count=turns)]
    for turn in range(turns):
        offset = 0.08 + turn * min(0.55, duration_s / max(2, turns + 1))
        input_phase = "user_speaking" if turn % 2 == 0 else "user_listening"
        output_phase = "assistant_speaking" if turn % 2 == 0 else "assistant_listening"
        events.append(
            _event(
                session_id,
                arrival_s + offset,
                "input_phase",
                input_phase=input_phase,
                turn=turn,
            )
        )
        events.append(
            _event(
                session_id,
                arrival_s + offset + 0.16,
                "output_phase",
                output_phase=output_phase,
                turn=turn,
            )
        )
    if barge_in:
        interrupt_s = arrival_s + min(max(0.42, duration_s * 0.35), duration_s - 0.2)
        events.append(
            _event(
                session_id,
                interrupt_s,
                "barge_in",
                input_phase="user_speaking",
                output_phase="assistant_speaking",
            )
        )
        events.append(
            _event(
                session_id,
                interrupt_s + 0.14,
                "resume",
                input_phase="user_speaking",
                output_phase="assistant_speaking",
            )
        )
    events.append(_event(session_id, arrival_s + duration_s, "finish"))
    return events


def _arrivals(
    concurrency: int,
    rng: random.Random,
    arrival_process: str,
) -> list[float]:
    if arrival_process not in {"poisson", "burst"}:
        raise ValueError("arrival_process must be 'poisson' or 'burst'")
    if arrival_process == "burst":
        return [float((index // 4) * 1.5 + (index % 4) * 0.03) for index in range(concurrency)]
    arrivals = [0.0]
    for _ in range(1, concurrency):
        arrivals.append(arrivals[-1] + rng.expovariate(2.5))
    return arrivals


def _duration_class(index: int, concurrency: int) -> str:
    remainder = index % 4
    if remainder == 0:
        return "short"
    if remainder in (1, 2):
        return "medium"
    return "long"


def generate_workload(
    workload: str,
    *,
    concurrency: int,
    seed: int = 0,
    arrival_process: str = "poisson",
) -> WorkloadTrace:
    """Generate a deterministic trace for one workload family."""
    workload = str(workload).upper()
    if workload not in SUPPORTED_WORKLOADS:
        raise ValueError(f"workload must be one of {SUPPORTED_WORKLOADS}")
    if isinstance(concurrency, bool) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    rng = random.Random(seed)
    events: list[WorkloadEvent] = []
    durations: dict[str, float] = {}
    duration_classes: dict[str, str] = {}
    arrivals: list[float]
    metadata: dict[str, Any]

    if workload == "A":
        arrivals = [index * 0.12 for index in range(concurrency)]
        metadata = {
            "purpose": "public_full_duplex_interaction_trace",
            "source": "normalized_public_inspired_interaction_fixture",
            "conversion": "canonical_event_v1",
            "arrival_process": "trace_fixture",
        }
        for index, arrival_s in enumerate(arrivals):
            session_id = f"session-{index}"
            duration_classes[session_id] = "medium"
            durations[session_id] = 3.0
            events.extend(
                _interaction_events(
                    session_id,
                    arrival_s,
                    durations[session_id],
                    barge_in=index % 3 == 0,
                    turns=4,
                )
            )
    elif workload == "B":
        arrivals = _arrivals(concurrency, rng, arrival_process)
        metadata = {
            "purpose": "production_like_voice_serving",
            "source": "deterministic_heavy_tail_generator",
            "arrival_process": arrival_process,
        }
        for index, arrival_s in enumerate(arrivals):
            session_id = f"session-{index}"
            class_name = _duration_class(index, concurrency)
            duration_classes[session_id] = class_name
            durations[session_id] = _DURATION_SCALE[class_name]
            events.extend(
                _interaction_events(
                    session_id,
                    arrival_s,
                    durations[session_id],
                    barge_in=index % 5 == 0,
                    turns=4,
                )
            )
    else:
        long_count = max(2, concurrency // 4)
        arrivals = [0.0] * long_count + [1.0 + (index - long_count) * 0.08 for index in range(long_count, concurrency)]
        metadata = {
            "purpose": "resource_fragmentation_stress",
            "source": "deterministic_resident_long_session_fixture",
            "arrival_process": "resident_then_late_short",
            "long_session_ids": [f"session-{index}" for index in range(long_count)],
            "late_short_session_ids": [f"session-{index}" for index in range(long_count, concurrency)],
        }
        for index, arrival_s in enumerate(arrivals):
            session_id = f"session-{index}"
            class_name = "long" if index < long_count else "short"
            duration_classes[session_id] = class_name
            durations[session_id] = 8.0 if class_name == "long" else 1.0
            events.extend(
                _interaction_events(
                    session_id,
                    arrival_s,
                    durations[session_id],
                    barge_in=index % 2 == 0,
                    turns=4,
                )
            )

    metadata.update(
        {
            "duration_classes": duration_classes,
            "durations_s": durations,
            "arrival_times_s": {
                f"session-{index}": float(arrival_s)
                for index, arrival_s in enumerate(arrivals)
            },
            "concurrency": concurrency,
            "seed": seed,
        }
    )
    ordered = tuple(
        sorted(
            events,
            key=lambda event: (
                event.timestamp_s,
                _EVENT_PRIORITY[event.event_type],
                event.session_id,
            ),
        )
    )
    return WorkloadTrace(
        workload_id=workload,
        seed=seed,
        concurrency=concurrency,
        events=ordered,
        metadata=metadata,
    )


def summarize_trace(trace: WorkloadTrace) -> dict[str, Any]:
    durations = [float(value) for value in trace.metadata["durations_s"].values()]
    arrivals = sorted(
        float(value) for value in trace.metadata["arrival_times_s"].values()
    )
    interarrivals = [right - left for left, right in zip(arrivals, arrivals[1:])]
    interarrival_mean = statistics.mean(interarrivals) if interarrivals else 0.0
    interarrival_cv = (
        statistics.pstdev(interarrivals) / interarrival_mean
        if interarrivals and interarrival_mean > 0
        else 0.0
    )
    session_round_counts = {
        session_id: sum(
            event.session_id == session_id
            and event.event_type in {"input_phase", "output_phase", "barge_in", "resume"}
            for event in trace.events
        )
        for session_id in trace.metadata["durations_s"]
    }
    return {
        "workload": trace.workload_id,
        "seed": trace.seed,
        "session_count": trace.concurrency,
        "event_count": len(trace.events),
        "arrival_process": trace.metadata.get("arrival_process"),
        "duration_classes": sorted(set(trace.metadata["duration_classes"].values())),
        "max_session_duration_s": max(durations),
        "median_session_duration_s": statistics.median(durations),
        "duration_variance_s": statistics.pvariance(durations),
        "arrival_span_s": (max(arrivals) - min(arrivals)) if arrivals else 0.0,
        "arrival_interarrival_cv": interarrival_cv,
        "min_progression_opportunities": min(session_round_counts.values()),
        "barge_in_count": sum(event.event_type == "barge_in" for event in trace.events),
        "resume_count": sum(event.event_type == "resume" for event in trace.events),
    }


def trace_hash(trace: WorkloadTrace) -> str:
    """Return the stable hash recorded in run manifests."""
    payload = json.dumps(trace.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def progression_rounds(
    trace: WorkloadTrace, *, quantum_s: float = 0.25
) -> tuple[tuple[str, ...], ...]:
    """Map canonical interaction events to deterministic runtime opportunities.

    Arrival and finish markers delimit the lifetime but do not themselves create
    model work. Input/output/interruption/resume markers in one time quantum are
    coalesced into one logical opportunity, preserving which sessions are active
    without changing model or acoustic scheduling semantics.
    """
    if quantum_s <= 0:
        raise ValueError("quantum_s must be positive")
    buckets: dict[int, set[str]] = {}
    for event in trace.events:
        if event.event_type in {"input_phase", "output_phase", "barge_in", "resume"}:
            bucket = int(event.timestamp_s // quantum_s)
            buckets.setdefault(bucket, set()).add(event.session_id)
    return tuple(
        tuple(sorted(session_ids))
        for _, session_ids in sorted(buckets.items())
        if session_ids
    )


def write_trace_jsonl(trace: WorkloadTrace, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"type": "manifest", "trace_hash": trace_hash(trace), **trace.as_dict()},
                sort_keys=True,
            )
            + "\n"
        )
        for event in trace.events:
            handle.write(json.dumps({"type": "event", **event.as_dict()}, sort_keys=True) + "\n")


def build_experiment_matrix(
    *,
    workloads: Sequence[str] = SUPPORTED_WORKLOADS,
    concurrencies: Sequence[int] = (4, 8, 16),
    repeats: int = 5,
) -> tuple[dict[str, Any], ...]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rows: list[dict[str, Any]] = []
    for workload in workloads:
        if workload not in SUPPORTED_WORKLOADS:
            raise ValueError(f"unsupported workload: {workload}")
        for concurrency in concurrencies:
            if concurrency <= 0:
                raise ValueError("concurrency must be positive")
            for system in BASELINES:
                for repeat in range(1, repeats + 1):
                    rows.append(
                        {
                            "workload": workload,
                            "concurrency": int(concurrency),
                            "system": system,
                            "repeat": repeat,
                            "seed": 1000 + repeat,
                        }
                    )
    return tuple(rows)
