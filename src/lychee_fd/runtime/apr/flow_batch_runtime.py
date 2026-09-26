"""APR-owned Flow-step execution using only the public backend contract."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .deadline_bounded_flow_scheduler import (
    DeadlineBoundedFlowScheduler,
    FlowBatchOpportunityObserver,
    FlowStepItem,
)


class FlowBatchRuntimeError(RuntimeError):
    """Raised when a Flow-step transition cannot be committed safely."""


class FlowStepBackend(Protocol):
    def advance_step(self, state: Any) -> Any:
        ...

    def advance_step_batch(self, states: tuple[Any, ...]) -> tuple[Any, ...]:
        ...

    def advance_step_mixed_batch(self, states: tuple[Any, ...]) -> tuple[Any, ...]:
        ...

    def finish(self, state: Any) -> Any:
        ...


@dataclass(frozen=True)
class FlowBatchExecutionResult:
    items: tuple[FlowStepItem, ...]
    states: tuple[Any, ...]
    committed: bool
    batch_size: int
    fallback: bool = False
    cancellation_reason: str | None = None


class FlowBatchRuntime:
    """Execute one exact-shape Euler step and commit by logical identity.

    The runtime never reads decoder or Token2Wav private members.  The backend
    is responsible for implementing the public prepare/advance/finalize APIs.
    """

    def __init__(
        self,
        *,
        backend: FlowStepBackend,
        max_batch_size: int = 2,
        max_batch_wait_ms: float | None = None,
        rolling_b1_step_median_ms: float | None = None,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
        time_ns: Callable[[], int] | None = None,
        commit: Callable[[FlowStepItem, Any], Any] | None = None,
        timing_enabled: bool = False,
        opportunity_observer: FlowBatchOpportunityObserver | None = None,
        allow_mixed_step: bool = False,
    ) -> None:
        for name in ("advance_step", "advance_step_batch", "finish"):
            if not callable(getattr(backend, name, None)):
                raise FlowBatchRuntimeError(f"backend must expose public {name}()")
        self._backend = backend
        if not isinstance(allow_mixed_step, bool):
            raise FlowBatchRuntimeError("allow_mixed_step must be boolean")
        if allow_mixed_step and not bool(getattr(backend, "mixed_step", False)):
            raise FlowBatchRuntimeError(
                "mixed-step runtime requires a backend explicitly marked mixed_step"
            )
        if allow_mixed_step and not callable(
            getattr(backend, "advance_step_mixed_batch", None)
        ):
            raise FlowBatchRuntimeError(
                "mixed-step backend must expose advance_step_mixed_batch()"
            )
        mixed_cap = getattr(backend, "mixed_step_max_batch_size", 2)
        if (
            isinstance(mixed_cap, bool)
            or not isinstance(mixed_cap, int)
            or mixed_cap < 2
        ):
            raise FlowBatchRuntimeError(
                "mixed-step backend must declare an integer batch-size cap"
            )
        if allow_mixed_step and max_batch_size > mixed_cap:
            raise FlowBatchRuntimeError(
                "configured mixed-step batch size exceeds backend cap"
            )
        self._mixed_step_max_batch_size = int(mixed_cap)
        self._allow_mixed_step = allow_mixed_step
        self._event_sink = event_sink
        self._time_ns = time_ns or time.monotonic_ns
        self._commit_callback = commit
        if not isinstance(timing_enabled, bool):
            raise FlowBatchRuntimeError("timing_enabled must be boolean")
        self._timing_enabled = timing_enabled
        self._scheduler = DeadlineBoundedFlowScheduler(
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            rolling_b1_step_median_ms=rolling_b1_step_median_ms,
            event_sink=event_sink,
            time_ns=self._time_ns,
            opportunity_observer=opportunity_observer,
            allow_mixed_step=allow_mixed_step,
        )
        self._current: dict[str, tuple[int, int]] = {}
        self._committed: dict[str, Any] = {}
        self._canceled: set[str] = set()

    @property
    def scheduler(self) -> DeadlineBoundedFlowScheduler:
        return self._scheduler

    def _emit(self, event: str, items: tuple[FlowStepItem, ...], **fields: Any) -> None:
        if self._event_sink is None:
            return
        payload = {
            "event": event,
            "event_type": event,
            "timestamp_monotonic_ns": int(self._time_ns()),
            "request_ids": [item.request_id for item in items],
            "batch_size": len(items),
            "generation_ids": [item.generation_id for item in items],
            "versions": [item.version for item in items],
            "step_index": items[0].step_index if items else None,
            "step_indices": [item.step_index for item in items],
            "mixed_step": len({item.step_index for item in items}) > 1,
            "shape_signature": list(items[0].shape_signature) if items else None,
            "last_chunk": [item.last_chunk for item in items],
            "checkpoint_ids": [
                f"{item.request_id}:{item.generation_id}:{item.version}:{item.step_index}"
                for item in items
            ],
        }
        payload.update(fields)
        try:
            self._event_sink(payload)
        except Exception:
            return

    def submit(self, item: FlowStepItem) -> None:
        current = self._current.get(item.request_id)
        expected = (item.generation_id, item.version)
        if current is not None and current != expected:
            raise FlowBatchRuntimeError(
                f"generation/version mismatch at submit for {item.request_id}"
            )
        self._current[item.request_id] = expected
        self._canceled.discard(item.request_id)
        self._scheduler.submit(item)

    def update_version(self, request_id: str, *, generation_id: int, version: int) -> None:
        if request_id not in self._current:
            raise FlowBatchRuntimeError(f"unknown request: {request_id}")
        self._current[request_id] = (generation_id, version)

    def cancel(self, request_id: str, *, generation_id: int) -> None:
        current = self._current.get(request_id)
        if current is None or current[0] != generation_id:
            raise FlowBatchRuntimeError(f"generation mismatch at cancel for {request_id}")
        removed = self._scheduler.cancel(request_id)
        self._canceled.add(request_id)
        if removed:
            self._emit(
                "FLOW_BATCH_CANCELLED",
                removed,
                reason="cancel_before_submit",
            )

    def reset_request(self, request_id: str) -> None:
        """Tear down all runtime-owned state before reusing a request ID.

        Cancellation prevents stale work from committing, while reset completes
        the request lifecycle so a later generation can start again at version 0.
        The operation is idempotent and also removes a pending scheduler item.
        """
        request_id = str(request_id)
        removed = self._scheduler.cancel(request_id)
        self._current.pop(request_id, None)
        self._committed.pop(request_id, None)
        self._canceled.discard(request_id)
        if removed:
            self._emit(
                "FLOW_BATCH_CANCELLED",
                removed,
                reason="request_reset",
            )

    def _is_current(self, item: FlowStepItem) -> bool:
        return (
            item.request_id not in self._canceled
            and self._current.get(item.request_id)
            == (item.generation_id, item.version)
        )

    @staticmethod
    def _validate_identity(item: FlowStepItem, state: Any) -> None:
        for name, expected in (
            ("request_id", item.request_id),
            ("generation_id", item.generation_id),
            ("version", item.version),
        ):
            actual = getattr(state, name, expected)
            if actual != expected:
                raise FlowBatchRuntimeError(
                    f"returned state identity mismatch for {item.request_id}: {name}"
                )

    @staticmethod
    def _variable_batch_fields(items: tuple[FlowStepItem, ...]) -> dict[str, Any]:
        """Return bounded metadata for an opt-in variable-length batch.

        Only public state shapes and identity metadata are inspected.  Tensor
        contents are intentionally excluded from telemetry and checkpoint
        accounting.
        """
        row_lengths: list[int] = []
        current_lengths: list[int] = []
        sequence_numbers: list[int | None] = []
        for item in items:
            state = item.state
            x = getattr(state, "x", None)
            current_lengths.append(
                int(x.shape[-1]) if getattr(x, "ndim", 0) and x.ndim >= 1 else 0
            )
            sequence = getattr(state, "sequence_no", None)
            sequence_numbers.append(int(sequence) if isinstance(sequence, int) else None)
            cache = getattr(state, "input_att_cache", None)
            step_index = int(getattr(state, "step_index", item.step_index))
            if isinstance(cache, (tuple, list)):
                cache = cache[step_index] if step_index < len(cache) else None
            row_lengths.append(
                int(cache.shape[3])
                if getattr(cache, "ndim", 0) >= 4
                else 0
            )
        max_length = max(row_lengths, default=0)
        current_length = max(current_lengths, default=0)
        physical_rows = len(items) * 2
        mask_bytes = physical_rows * current_length * (current_length + max_length)
        return {
            "sequence_no": sequence_numbers,
            "step_indices": [int(item.step_index) for item in items],
            "current_lengths": current_lengths,
            "row_attention_cache_lengths": row_lengths,
            "max_attention_cache_length": max_length,
            "mask_bytes": int(mask_bytes),
        }

    def run_next(self, *, now_ns: int | None = None) -> FlowBatchExecutionResult | None:
        items = self._scheduler.next_batch(now_ns=now_ns)
        if items is None:
            return None
        fallback = len(items) == 1
        variable_batch = len(items) > 1 and bool(
            getattr(self._backend, "variable_length", False)
        )
        mixed_batch = variable_batch and self._allow_mixed_step and bool(
            getattr(self._backend, "mixed_step", False)
        )
        variable_fields = self._variable_batch_fields(items) if variable_batch else {}
        self._emit(
            "FLOW_BATCH_SUBMIT",
            items,
            fallback=fallback,
        )
        if variable_batch:
            self._emit(
                "FLOW_MIXED_STEP_BATCH_ATTEMPT" if mixed_batch else "FLOW_VARIABLE_BATCH_ATTEMPT",
                items,
                **variable_fields,
            )
        if not all(self._is_current(item) for item in items):
            self._emit(
                "FLOW_BATCH_CANCELLED",
                items,
                reason="stale_before_execution",
            )
            if variable_batch:
                self._emit(
                    "FLOW_MIXED_STEP_BATCH_STALE_DROP"
                    if mixed_batch
                    else "FLOW_VARIABLE_BATCH_STALE_DROP",
                    items,
                    reason="stale_before_execution",
                    **variable_fields,
                )
            return FlowBatchExecutionResult(
                items=items,
                states=(),
                committed=False,
                batch_size=len(items),
                fallback=fallback,
                cancellation_reason="stale_before_execution",
            )
        try:
            step_wall_start_ns = self._time_ns()
            self._emit(
                "FLOW_STEP_START",
                items,
                step_index=items[0].step_index,
                remaining_steps=[max(0, item.n_timesteps - item.step_index) for item in items],
            )
            wall_start_ns = time.monotonic_ns() if self._timing_enabled else 0
            cuda_start = None
            cuda_end = None
            cuda_time_ms = 0.0
            if self._timing_enabled:
                try:
                    import torch

                    if torch.cuda.is_available():
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_end = torch.cuda.Event(enable_timing=True)
                        cuda_start.record()
                except Exception:
                    cuda_start = None
                    cuda_end = None
            if len(items) == 1:
                next_states = (self._backend.advance_step(items[0].state),)
            elif mixed_batch:
                next_states = tuple(
                    self._backend.advance_step_mixed_batch(
                        tuple(item.state for item in items)
                    )
                )
            else:
                next_states = tuple(
                    self._backend.advance_step_batch(tuple(item.state for item in items))
                )
            if cuda_start is not None and cuda_end is not None:
                cuda_end.record()
                cuda_end.synchronize()
            if self._timing_enabled:
                cuda_time_ms = (
                    float(cuda_start.elapsed_time(cuda_end))
                    if cuda_start is not None and cuda_end is not None
                    else 0.0
                )
                self._emit(
                    "FLOW_BATCH_TIMING",
                    items,
                    fallback=fallback,
                    wall_time_ms=(time.monotonic_ns() - wall_start_ns) / 1_000_000.0,
                    cuda_time_ms=cuda_time_ms,
                )
            if len(next_states) != len(items):
                raise FlowBatchRuntimeError("backend returned the wrong number of states")
            for item, state in zip(items, next_states):
                self._validate_identity(item, state)
            self._emit(
                "FLOW_STEP_END",
                items,
                step_index=items[0].step_index,
                next_step_indices=[
                    int(getattr(state, "step_index", -1)) for state in next_states
                ],
                wall_time_ms=max(
                    0.0, (self._time_ns() - step_wall_start_ns) / 1_000_000.0
                ),
                cuda_time_ms=cuda_time_ms,
                remaining_steps=[
                    max(
                        0,
                        item.n_timesteps
                        - int(getattr(state, "step_index", item.step_index + 1)),
                    )
                    for item, state in zip(items, next_states)
                ],
            )
        except FlowBatchRuntimeError as exc:
            if variable_batch:
                self._emit(
                    "FLOW_MIXED_STEP_BATCH_REJECTED"
                    if mixed_batch
                    else "FLOW_VARIABLE_BATCH_REJECTED",
                    items,
                    reason=str(exc),
                    **variable_fields,
                )
            raise
        except Exception as exc:
            if variable_batch:
                self._emit(
                    "FLOW_MIXED_STEP_BATCH_REJECTED"
                    if mixed_batch
                    else "FLOW_VARIABLE_BATCH_REJECTED",
                    items,
                    reason=f"{type(exc).__name__}: {exc}",
                    **variable_fields,
                )
            raise FlowBatchRuntimeError(f"Flow batch execution failed: {exc}") from exc

        if not all(self._is_current(item) for item in items):
            self._emit(
                "FLOW_BATCH_CANCELLED",
                items,
                reason="stale_after_execution",
            )
            if variable_batch:
                self._emit(
                    "FLOW_MIXED_STEP_BATCH_STALE_DROP"
                    if mixed_batch
                    else "FLOW_VARIABLE_BATCH_STALE_DROP",
                    items,
                    reason="stale_after_execution",
                    **variable_fields,
                )
            return FlowBatchExecutionResult(
                items=items,
                states=next_states,
                committed=False,
                batch_size=len(items),
                fallback=fallback,
                cancellation_reason="stale_after_execution",
            )
        for item, state in zip(items, next_states):
            if self._commit_callback is not None:
                self._commit_callback(item, state)
            self._committed[item.request_id] = state
        self._emit(
            "FLOW_BATCH_COMPLETE",
            items,
            fallback=fallback,
            next_step_indices=[int(getattr(state, "step_index", -1)) for state in next_states],
        )
        if variable_batch:
            self._emit(
                "FLOW_MIXED_STEP_BATCH_COMPLETE"
                if mixed_batch
                else "FLOW_VARIABLE_BATCH_COMPLETE",
                items,
                **variable_fields,
                next_step_indices=[
                    int(getattr(state, "step_index", -1)) for state in next_states
                ],
            )
        return FlowBatchExecutionResult(
            items=items,
            states=next_states,
            committed=True,
            batch_size=len(items),
            fallback=fallback,
        )

    def committed_state(self, request_id: str) -> Any | None:
        return self._committed.get(request_id)

    def finish(self, request_id: str) -> Any:
        state = self._committed.get(request_id)
        if state is None:
            raise FlowBatchRuntimeError(f"no committed Flow state for {request_id}")
        return self._backend.finish(state)
