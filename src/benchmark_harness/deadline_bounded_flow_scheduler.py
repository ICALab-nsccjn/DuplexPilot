"""Deadline-bounded, exact-shape Flow-step scheduling primitives."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable


class DeadlineBoundedFlowSchedulerError(ValueError):
    """Raised when a Flow scheduling request is invalid."""


@dataclass(frozen=True)
class FlowStepItem:
    """Request-owned public Flow state plus immutable compatibility metadata."""

    request_id: str
    generation_id: int
    version: int
    state: Any
    ready_at_ns: int
    model_identity: str
    device: str
    dtype: str
    step_index: int
    shape_signature: tuple[int, ...]
    last_chunk: bool
    n_timesteps: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise DeadlineBoundedFlowSchedulerError("request_id must be non-empty")
        for name, value in (
            ("generation_id", self.generation_id),
            ("version", self.version),
            ("ready_at_ns", self.ready_at_ns),
            ("step_index", self.step_index),
            ("n_timesteps", self.n_timesteps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise DeadlineBoundedFlowSchedulerError(f"{name} must be an integer")
            if name != "ready_at_ns" and value < 0:
                raise DeadlineBoundedFlowSchedulerError(f"{name} must be non-negative")
        if not isinstance(self.model_identity, str) or not self.model_identity:
            raise DeadlineBoundedFlowSchedulerError("model_identity must be non-empty")
        if not isinstance(self.device, str) or not self.device:
            raise DeadlineBoundedFlowSchedulerError("device must be non-empty")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise DeadlineBoundedFlowSchedulerError("dtype must be non-empty")
        if not isinstance(self.shape_signature, tuple):
            raise DeadlineBoundedFlowSchedulerError("shape_signature must be a tuple")
        if not isinstance(self.last_chunk, bool):
            raise DeadlineBoundedFlowSchedulerError("last_chunk must be boolean")

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return (
            self.model_identity,
            self.device,
            self.dtype,
            self.step_index,
            self.shape_signature,
            self.last_chunk,
            self.n_timesteps,
        )


@dataclass(frozen=True)
class FlowBatchOpportunitySnapshot:
    """Read-only snapshot of the compatible states visible at one decision."""

    timestamp_monotonic_ns: int
    request_ids: tuple[str, ...]
    compatible_request_ids: tuple[str, ...]
    max_compatible_request_ids: tuple[str, ...]
    candidate_count: int
    compatible_count: int
    max_compatible_count: int
    compatibility_group_sizes: tuple[int, ...]
    potential_batch_size: int
    potential_b2: bool
    potential_b4: bool
    rejection_counts: dict[str, int]
    step_index_distribution: dict[str, int]
    shape_signature_distribution: dict[str, int]
    last_chunk_distribution: dict[str, int]
    ready_age_ms: float
    inter_ready_gap_ms: float
    max_batch_size: int
    wait_budget_ms: float
    logical_concurrency: int | None


class FlowBatchOpportunityObserver:
    """Observe batch opportunity without changing scheduler selection."""

    _COMPATIBILITY_FIELDS = (
        "model_identity",
        "device",
        "dtype",
        "step_index",
        "shape_signature",
        "last_chunk",
        "n_timesteps",
    )

    def __init__(
        self,
        *,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
        logical_concurrency: int | None = None,
        time_ns: Callable[[], int] | None = None,
    ) -> None:
        if logical_concurrency is not None and (
            isinstance(logical_concurrency, bool)
            or not isinstance(logical_concurrency, int)
            or logical_concurrency <= 0
        ):
            raise ValueError("logical_concurrency must be a positive integer or None")
        self._event_sink = event_sink
        self.logical_concurrency = logical_concurrency
        self._time_ns = time_ns or time.monotonic_ns

    @staticmethod
    def _shape_digest(shape_signature: tuple[int, ...]) -> str:
        encoded = json.dumps(
            [int(value) for value in shape_signature],
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _count(values: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return result

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(payload)
        except Exception:
            return

    def observe(
        self,
        items: tuple[FlowStepItem, ...] | list[FlowStepItem],
        *,
        max_batch_size: int,
        wait_budget_ns: int,
        now_ns: int | None = None,
    ) -> FlowBatchOpportunitySnapshot | None:
        """Record a snapshot for the current pending set.

        The method only reads immutable scheduling metadata and never removes,
        reorders, or mutates a pending item.
        """
        pending = tuple(items)
        if not pending:
            return None
        if isinstance(max_batch_size, bool) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if isinstance(wait_budget_ns, bool) or wait_budget_ns < 0:
            raise ValueError("wait_budget_ns must be non-negative")
        now = int(self._time_ns() if now_ns is None else now_ns)
        anchor = min(pending, key=lambda value: (value.ready_at_ns, value.request_id))
        groups: dict[tuple[Any, ...], list[FlowStepItem]] = {}
        for item in pending:
            groups.setdefault(item.compatibility_key, []).append(item)
        ordered_groups = sorted(
            (
                tuple(sorted(group, key=lambda value: (value.ready_at_ns, value.request_id)))
                for group in groups.values()
            ),
            key=lambda group: (group[0].ready_at_ns, group[0].request_id),
        )
        largest_group = max(
            ordered_groups,
            key=lambda group: (len(group), -group[0].ready_at_ns, group[0].request_id),
            default=(),
        )
        compatible = tuple(
            sorted(
                (item for item in pending if item.compatibility_key == anchor.compatibility_key),
                key=lambda value: (value.ready_at_ns, value.request_id),
            )
        )
        rejection_counts: dict[str, int] = {}
        for candidate in pending:
            if candidate is anchor:
                continue
            for name, left, right in zip(
                self._COMPATIBILITY_FIELDS,
                anchor.compatibility_key,
                candidate.compatibility_key,
            ):
                if left != right:
                    rejection_counts[name] = rejection_counts.get(name, 0) + 1
        minimum_ready = min(item.ready_at_ns for item in pending)
        maximum_ready = max(item.ready_at_ns for item in pending)
        potential_batch_size = min(int(max_batch_size), len(compatible))
        snapshot = FlowBatchOpportunitySnapshot(
            timestamp_monotonic_ns=now,
            request_ids=tuple(item.request_id for item in pending),
            compatible_request_ids=tuple(item.request_id for item in compatible),
            max_compatible_request_ids=tuple(item.request_id for item in largest_group),
            candidate_count=len(pending),
            compatible_count=len(compatible),
            max_compatible_count=len(largest_group),
            compatibility_group_sizes=tuple(sorted((len(group) for group in ordered_groups), reverse=True)),
            potential_batch_size=potential_batch_size,
            potential_b2=potential_batch_size >= 2,
            potential_b4=potential_batch_size >= 4,
            rejection_counts=dict(rejection_counts),
            step_index_distribution=self._count([str(item.step_index) for item in pending]),
            shape_signature_distribution=self._count(
                [self._shape_digest(item.shape_signature) for item in pending]
            ),
            last_chunk_distribution=self._count([str(item.last_chunk) for item in pending]),
            ready_age_ms=max(0.0, (now - minimum_ready) / 1_000_000.0),
            inter_ready_gap_ms=max(0.0, (maximum_ready - minimum_ready) / 1_000_000.0),
            max_batch_size=int(max_batch_size),
            wait_budget_ms=float(wait_budget_ns) / 1_000_000.0,
            logical_concurrency=self.logical_concurrency,
        )
        payload = {
            "event": "FLOW_BATCH_OPPORTUNITY",
            "event_type": "FLOW_BATCH_OPPORTUNITY",
            "timestamp_monotonic_ns": snapshot.timestamp_monotonic_ns,
            "request_ids": list(snapshot.request_ids),
            "compatible_request_ids": list(snapshot.compatible_request_ids),
            "candidate_count": snapshot.candidate_count,
            "compatible_count": snapshot.compatible_count,
            "max_compatible_count": snapshot.max_compatible_count,
            "max_compatible_request_ids": list(snapshot.max_compatible_request_ids),
            "compatibility_group_sizes": list(snapshot.compatibility_group_sizes),
            "potential_batch_size": snapshot.potential_batch_size,
            "potential_b2": snapshot.potential_b2,
            "potential_b4": snapshot.potential_b4,
            "max_batch_size": snapshot.max_batch_size,
            "wait_budget_ms": snapshot.wait_budget_ms,
            "ready_age_ms": snapshot.ready_age_ms,
            "inter_ready_gap_ms": snapshot.inter_ready_gap_ms,
            "step_index_distribution": snapshot.step_index_distribution,
            "shape_signature_distribution": snapshot.shape_signature_distribution,
            "last_chunk_distribution": snapshot.last_chunk_distribution,
            "compatibility_rejection_counts": snapshot.rejection_counts,
            "logical_concurrency": snapshot.logical_concurrency,
        }
        self._emit(payload)
        if rejection_counts:
            self._emit(
                {
                    "event": "FLOW_BATCH_REJECTION_SUMMARY",
                    "event_type": "FLOW_BATCH_REJECTION_SUMMARY",
                    "timestamp_monotonic_ns": now,
                    "request_ids": list(snapshot.request_ids),
                    "anchor_request_id": anchor.request_id,
                    "candidate_count": snapshot.candidate_count,
                    "compatible_count": snapshot.compatible_count,
                    "compatibility_rejection_counts": dict(rejection_counts),
                    "logical_concurrency": snapshot.logical_concurrency,
                }
            )
        return snapshot


class DeadlineBoundedFlowScheduler:
    """Form compatible batches without extending the latency deadline."""

    _COMPATIBILITY_FIELDS = (
        "model_identity",
        "device",
        "dtype",
        "step_index",
        "shape_signature",
        "last_chunk",
        "n_timesteps",
    )

    def __init__(
        self,
        *,
        max_batch_size: int = 2,
        max_batch_wait_ms: float | None = None,
        rolling_b1_step_median_ms: float | None = None,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
        time_ns: Callable[[], int] | None = None,
        opportunity_observer: FlowBatchOpportunityObserver | None = None,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise DeadlineBoundedFlowSchedulerError("max_batch_size must be an integer")
        if max_batch_size <= 0:
            raise DeadlineBoundedFlowSchedulerError("max_batch_size must be positive")
        if max_batch_wait_ms is None:
            if rolling_b1_step_median_ms is None:
                max_batch_wait_ms = 2.0
            else:
                if rolling_b1_step_median_ms < 0:
                    raise DeadlineBoundedFlowSchedulerError(
                        "rolling_b1_step_median_ms must be non-negative"
                    )
                max_batch_wait_ms = min(2.0, rolling_b1_step_median_ms * 0.10)
        if isinstance(max_batch_wait_ms, bool) or not isinstance(max_batch_wait_ms, (int, float)):
            raise DeadlineBoundedFlowSchedulerError("max_batch_wait_ms must be numeric")
        if max_batch_wait_ms < 0:
            raise DeadlineBoundedFlowSchedulerError("max_batch_wait_ms must be non-negative")
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = float(max_batch_wait_ms)
        self._wait_ns = int(round(self.max_batch_wait_ms * 1_000_000.0))
        self._pending: list[FlowStepItem] = []
        self._event_sink = event_sink
        self._time_ns = time_ns or time.monotonic_ns
        self._opportunity_observer = opportunity_observer

    def _emit(self, event: str, items: tuple[FlowStepItem, ...], **fields: Any) -> None:
        if self._event_sink is None:
            return
        first = items[0] if items else None
        payload = {
            "event": event,
            "event_type": event,
            "timestamp_monotonic_ns": int(self._time_ns()),
            "request_ids": [item.request_id for item in items],
            "batch_size": len(items),
        }
        if first is not None:
            payload.update(
                {
                    "generation_ids": [item.generation_id for item in items],
                    "versions": [item.version for item in items],
                    "step_index": first.step_index,
                    "shape_signature": list(first.shape_signature),
                    "wait_ms": max(
                        0.0,
                        (payload["timestamp_monotonic_ns"] - min(item.ready_at_ns for item in items))
                        / 1_000_000.0,
                    ),
                }
            )
        payload.update(fields)
        try:
            self._event_sink(payload)
        except Exception:
            return

    def submit(self, item: FlowStepItem) -> None:
        if not isinstance(item, FlowStepItem):
            raise DeadlineBoundedFlowSchedulerError("submit requires FlowStepItem")
        if any(existing.request_id == item.request_id for existing in self._pending):
            raise DeadlineBoundedFlowSchedulerError(
                f"request already has a pending Flow step: {item.request_id}"
            )
        self._pending.append(item)
        self._pending.sort(key=lambda value: (value.ready_at_ns, value.request_id))
        self._emit(
            "FLOW_STEP_READY",
            (item,),
            shape_signature=list(item.shape_signature),
        )

    def cancel(self, request_id: str) -> tuple[FlowStepItem, ...]:
        removed = tuple(item for item in self._pending if item.request_id == request_id)
        self._pending = [item for item in self._pending if item.request_id != request_id]
        return removed

    def pending_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self._pending)

    def next_batch(self, *, now_ns: int | None = None) -> tuple[FlowStepItem, ...] | None:
        if not self._pending:
            return None
        now = int(self._time_ns() if now_ns is None else now_ns)
        if self._opportunity_observer is not None:
            self._opportunity_observer.observe(
                tuple(self._pending),
                max_batch_size=self.max_batch_size,
                wait_budget_ns=self._wait_ns,
                now_ns=now,
            )
        oldest = min(self._pending, key=lambda value: (value.ready_at_ns, value.request_id))
        compatible = sorted(
            (
                item
                for item in self._pending
                if item.compatibility_key == oldest.compatibility_key
            ),
            key=lambda value: (value.ready_at_ns, value.request_id),
        )
        selected = tuple(compatible[: self.max_batch_size])
        age_ns = max(0, now - min(item.ready_at_ns for item in selected))
        if len(selected) < 2 and age_ns < self._wait_ns:
            return None
        for item in selected:
            self._pending.remove(item)
        if len(selected) >= 2:
            self._emit("FLOW_BATCH_FORMED", selected, reason="compatible_ready")
        else:
            rejected: dict[str, int] = {}
            reference = oldest.compatibility_key
            for candidate in self._pending:
                if candidate is oldest:
                    continue
                candidate_key = candidate.compatibility_key
                for name, left, right in zip(
                    self._COMPATIBILITY_FIELDS, reference, candidate_key
                ):
                    if left != right:
                        rejected[name] = rejected.get(name, 0) + 1
            self._emit(
                "FLOW_BATCH_FALLBACK",
                selected,
                reason="max_batch_wait",
                compatibility_rejection_counts=rejected,
            )
        return selected
