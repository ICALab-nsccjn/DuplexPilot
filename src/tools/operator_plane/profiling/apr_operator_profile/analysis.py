"""Conservative, auditable analysis for the APR Flow operator gate.

The module deliberately separates measured operator time from the portion of
the end-to-end timeline that is known to be critical.  It is an analysis
utility: it does not alter Flow execution or select a production operator by
itself.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_CANDIDATE_PRIORITY = {
    "state_cache_transition": 0,
    "dispatch_pack_reuse": 1,
    "single_dit_family": 2,
}


@dataclass(frozen=True)
class CategoryTiming:
    """Measured time for one pre-registered optimization boundary.

    ``critical_path_ms`` is an observed or explicitly documented upper-bound
    attribution.  ``avoidable_flow_ms`` is the portion used for the
    counterfactual score and must never exceed the measured Flow wall time.
    ``flow_cuda_ms`` is required for the non-QKV DiT-family eligibility rule.
    """

    name: str
    flow_wall_ms: float
    critical_path_ms: float
    avoidable_flow_ms: float
    flow_cuda_ms: float | None = None
    critical_cuda_ms: float | None = None
    evidence: str = ""

    def __post_init__(self) -> None:
        for field in (
            "flow_wall_ms",
            "critical_path_ms",
            "avoidable_flow_ms",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be a finite non-negative number")
        if self.avoidable_flow_ms > self.flow_wall_ms + 1e-9:
            raise ValueError("avoidable_flow_ms cannot exceed flow_wall_ms")
        for field in ("flow_cuda_ms", "critical_cuda_ms"):
            value = getattr(self, field)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{field} must be finite and non-negative")


@dataclass(frozen=True)
class CandidateAssessment:
    name: str
    flow_wall_ms: float
    critical_path_ms: float
    avoidable_flow_ms: float
    flow_wall_share: float
    critical_path_share: float
    avoidable_flow_share: float
    candidate_score: float
    ideal_bound: float
    eligible: bool
    decision_reason: str
    evidence: str


def _thresholds(name: str) -> tuple[float, float, str] | None:
    if name == "state_cache_transition":
        return 0.10, 0.10, "Flow wall"
    if name == "dispatch_pack_reuse":
        return 0.15, 0.05, "Flow wall"
    if name == "single_dit_family":
        return 0.20, 0.10, "Flow CUDA"
    return None


def analyze_categories(
    timings: Iterable[CategoryTiming],
    *,
    e2e_critical_ms: float,
) -> tuple[CandidateAssessment, ...]:
    """Compute the registered candidate score and Amdahl upper bound.

    ``e2e_critical_ms`` is the denominator for the Amdahl share.  Candidate
    categories may overlap in the raw profiler, but callers must provide
    mutually exclusive category timings; this function does not silently
    deduplicate them.
    """

    values = tuple(timings)
    if not values:
        raise ValueError("at least one category timing is required")
    e2e = float(e2e_critical_ms)
    if not math.isfinite(e2e) or e2e <= 0:
        raise ValueError("e2e_critical_ms must be positive and finite")
    total_flow = sum(item.flow_wall_ms for item in values)
    if total_flow <= 0:
        raise ValueError("total Flow wall time must be positive")
    total_cuda = sum(
        item.flow_cuda_ms if item.flow_cuda_ms is not None else 0.0
        for item in values
    )
    rows: list[CandidateAssessment] = []
    for item in values:
        flow_share = item.flow_wall_ms / total_flow
        critical_share = item.critical_path_ms / e2e
        avoidable_share = (
            item.avoidable_flow_ms / item.flow_wall_ms
            if item.flow_wall_ms > 0
            else 0.0
        )
        score = critical_share * avoidable_share
        ideal_bound = math.inf if critical_share >= 1.0 else 1.0 / (1.0 - critical_share)
        reasons: list[str] = []
        threshold = _thresholds(item.name)
        if threshold is None:
            reasons.append("unregistered candidate")
        else:
            minimum_share, minimum_critical, measured_label = threshold
            if item.name == "single_dit_family":
                measured = (item.flow_cuda_ms or 0.0) / total_cuda if total_cuda > 0 else 0.0
            else:
                measured = flow_share
            if measured < minimum_share:
                reasons.append(
                    f"{measured_label} share {measured:.4f} < {minimum_share:.4f}"
                )
            if critical_share < minimum_critical:
                reasons.append(
                    f"critical-path share {critical_share:.4f} < {minimum_critical:.4f}"
                )
            if ideal_bound < 1.15:
                reasons.append(
                    f"Amdahl ideal bound {ideal_bound:.4f} < 1.15"
                )
        rows.append(
            CandidateAssessment(
                name=item.name,
                flow_wall_ms=item.flow_wall_ms,
                critical_path_ms=item.critical_path_ms,
                avoidable_flow_ms=item.avoidable_flow_ms,
                flow_wall_share=flow_share,
                critical_path_share=critical_share,
                avoidable_flow_share=avoidable_share,
                candidate_score=score,
                ideal_bound=ideal_bound,
                eligible=not reasons,
                decision_reason="eligible" if not reasons else "; ".join(reasons),
                evidence=item.evidence,
            )
        )
    return tuple(rows)


def choose_candidate(rows: Iterable[CandidateAssessment]) -> CandidateAssessment | None:
    """Choose at most one eligible candidate using the registered priority."""

    eligible = [row for row in rows if row.eligible]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            row.candidate_score,
            -_CANDIDATE_PRIORITY.get(row.name, 99),
        ),
        reverse=True,
    )
    return eligible[0]


def load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load JSONL records while rejecting malformed non-empty lines."""

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return tuple(records)


def _duration_ms(event: Mapping[str, Any], *, prefer_cuda: bool = False) -> float:
    keys = (
        ("cuda_time_ms", "CUDA_time", "wall_time_ms", "duration_ms")
        if prefer_cuda
        else ("wall_time_ms", "duration_ms", "cuda_time_ms", "CUDA_time")
    )
    for key in keys:
        value = event.get(key)
        if value is not None:
            try:
                converted = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(converted) and converted >= 0:
                return converted
    return 0.0


def _is_critical(event: Mapping[str, Any]) -> bool:
    # Some preserved records contain both a false event-level flag and a true
    # row-level flag.  Treat the row as critical if any available annotation is
    # true rather than letting the first key mask the others.
    return any(
        bool(event[key])
        for key in ("critical_path", "critical_path_flag", "critical_path_row")
        if key in event
    )


def _critical_event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return a stable key for one physical Flow execution.

    The preserved critical-path traces contain one row per logical request.
    A B=2 execution therefore appears twice with the same ordinal and timing.
    Deduplicating here prevents a row-level trace from being mistaken for two
    GPU executions.
    """

    case_dir = event.get("case_dir")
    ordinal = event.get("ordinal")
    start = event.get("start_ms")
    end = event.get("end_ms")
    batch_size = event.get("batch_size")
    step_index = event.get("step_index")
    if ordinal is not None or start is not None or end is not None:
        return (case_dir, ordinal, start, end, batch_size, step_index)
    request_ids = event.get("request_ids")
    if isinstance(request_ids, list):
        request_ids = tuple(request_ids)
    return (case_dir, ordinal, request_ids, batch_size, step_index)


def summarize_critical_path_events(events: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summarize preserved row-level critical-path records.

    This function is intentionally separate from :func:`summarize_online_events`:
    online telemetry is event-level, while the historical critical-path trace
    is often duplicated once per logical row in a batch.
    """

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        key = _critical_event_key(event)
        prior = unique.get(key)
        if prior is None:
            unique[key] = dict(event)
        elif _is_critical(event):
            prior["critical_path_row"] = True

    summary: dict[str, float | int] = {
        "flow_wall_ms": 0.0,
        "flow_cuda_ms": 0.0,
        "critical_flow_wall_ms": 0.0,
        "critical_flow_cuda_ms": 0.0,
        "b2_flow_wall_ms": 0.0,
        "b2_critical_flow_wall_ms": 0.0,
        "flow_step_count": 0,
        "b2_step_count": 0,
        "unique_event_count": len(unique),
    }
    for event in unique.values():
        wall = _duration_ms(event)
        cuda = _duration_ms(event, prefer_cuda=True)
        try:
            batch_size = int(event.get("batch_size", 1))
        except (TypeError, ValueError):
            batch_size = 1
        critical = _is_critical(event)
        summary["flow_wall_ms"] += wall
        summary["flow_cuda_ms"] += cuda
        summary["flow_step_count"] += 1
        if batch_size > 1:
            summary["b2_step_count"] += 1
            summary["b2_flow_wall_ms"] += wall
        if critical:
            summary["critical_flow_wall_ms"] += wall
            summary["critical_flow_cuda_ms"] += cuda
            if batch_size > 1:
                summary["b2_critical_flow_wall_ms"] += wall
    return summary


def summarize_online_events(events: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summarize preserved online boundary events without changing them."""

    summary: dict[str, float | int] = {
        "flow_wall_ms": 0.0,
        "flow_cuda_ms": 0.0,
        "flow_critical_ms": 0.0,
        "finalize_critical_ms": 0.0,
        "pcm_critical_ms": 0.0,
        "flow_step_count": 0,
        "b2_step_count": 0,
    }
    for event in events:
        kind = str(event.get("event") or event.get("event_type") or "")
        if kind == "FLOW_STEP_END":
            wall = _duration_ms(event)
            cuda = _duration_ms(event, prefer_cuda=True)
            summary["flow_wall_ms"] += wall
            summary["flow_cuda_ms"] += cuda
            summary["flow_step_count"] += 1
            batch_size = event.get("batch_size", 1)
            try:
                if int(batch_size) > 1:
                    summary["b2_step_count"] += 1
            except (TypeError, ValueError):
                pass
            if _is_critical(event):
                summary["flow_critical_ms"] += wall if wall > 0 else cuda
        elif kind == "FLOW_FINALIZE_END" and _is_critical(event):
            summary["finalize_critical_ms"] += _duration_ms(event)
        elif kind == "PCM_READY" and _is_critical(event):
            summary["pcm_critical_ms"] += _duration_ms(event)
    return summary


def _profile_category(name: str) -> str:
    # Generic cat/copy/clone records are deliberately left unattributed:
    # they include model/cache internals and cannot prove host dispatch or
    # APR state-transition work without boundary instrumentation.
    if name == "aten::_efficient_attention_forward":
        return "single_dit_family"
    return "unattributed"


def aggregate_profile_categories(paths: Iterable[str | Path]) -> dict[str, dict[str, float | int]]:
    """Aggregate high-level ``aten::`` records, excluding child CUDA kernels.

    The mapping is intentionally conservative: generic ``addmm`` is kept
    unattributed because the profile cannot prove whether a call is QKV.
    """

    aggregate: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "cuda_ms": 0.0,
            "cpu_ms": 0.0,
            "kernel_count": 0,
            "record_count": 0,
            "explicit_boundary_ms": 0.0,
            "heuristic_ms": 0.0,
        }
    )
    for path in paths:
        for record in load_jsonl(path):
            if record.get("record_type") != "operator":
                continue
            name = str(record.get("operator_name", ""))
            if not name.startswith("aten::"):
                continue
            explicit_category = record.get("operator_category") or record.get("phase_category")
            if explicit_category in {
                "state_cache_transition",
                "dispatch_pack_reuse",
                "single_dit_family",
            }:
                category = str(explicit_category)
                attribution_key = "explicit_boundary_ms"
            else:
                category = _profile_category(name)
                attribution_key = "heuristic_ms"
            target = aggregate[category]
            cuda_ms = float(record.get("cuda_time_us") or 0.0) / 1000.0
            target["cuda_ms"] += cuda_ms
            target["cpu_ms"] += float(record.get("cpu_time_us") or 0.0) / 1000.0
            target["kernel_count"] += int(record.get("kernel_count") or 0)
            target["record_count"] += 1
            target[attribution_key] += cuda_ms
    for name in ("state_cache_transition", "dispatch_pack_reuse", "single_dit_family", "unattributed"):
        aggregate.setdefault(
            name,
            {
                "cuda_ms": 0.0,
                "cpu_ms": 0.0,
                "kernel_count": 0,
                "record_count": 0,
                "explicit_boundary_ms": 0.0,
                "heuristic_ms": 0.0,
            },
        )
    return dict(aggregate)


__all__ = [
    "CandidateAssessment",
    "CategoryTiming",
    "aggregate_profile_categories",
    "analyze_categories",
    "choose_candidate",
    "load_jsonl",
    "summarize_critical_path_events",
    "summarize_online_events",
]
