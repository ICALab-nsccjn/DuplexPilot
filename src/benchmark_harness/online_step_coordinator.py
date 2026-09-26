"""Online, deadline-bounded Flow-step coordination for APR acoustic lanes.

The coordinator owns only public Flow-step messages.  A backend adapter owns
Token2Wav preparation, state updates, and PCM finalization; this module never
reaches into decoder or Token2Wav private members.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from .contracts import AcousticTokenBatch
from .deadline_bounded_flow_scheduler import (
    FlowBatchOpportunityObserver,
    FlowStepItem,
)
from .flow_batch_runtime import FlowBatchExecutionResult, FlowBatchRuntime, FlowBatchRuntimeError


class OnlineFlowStepCoordinatorError(RuntimeError):
    """Raised when online Flow-step ownership or lifecycle is invalid."""


@dataclass
class PreparedFlowChunk:
    """Prepared public Flow state belonging to one logical acoustic chunk."""

    batch: AcousticTokenBatch
    state: Any
    n_timesteps: int
    worker_id: int


class FlowChunkExecutionAdapter(Protocol):
    """Public adapter boundary used by the online coordinator."""

    def register(self, request_id: str, *, stream_id: str, generation_id: int) -> None:
        ...

    def prepare_chunk(self, batch: AcousticTokenBatch) -> PreparedFlowChunk | Mapping[str, Any]:
        ...

    def make_step_item(self, chunk: PreparedFlowChunk) -> FlowStepItem:
        ...

    def advance_step(self, state: Any) -> Any:
        ...

    def advance_step_batch(self, states: tuple[Any, ...]) -> tuple[Any, ...]:
        ...

    def update_step(self, chunk: PreparedFlowChunk, next_state: Any) -> None:
        ...

    def finalize_chunk(self, chunk: PreparedFlowChunk) -> Any:
        ...

    def cancel(self, request_id: str) -> None:
        ...


@dataclass
class _Submission:
    batch: AcousticTokenBatch
    future: Future[Any]


@dataclass
class _ActiveChunk:
    batch: AcousticTokenBatch
    future: Future[Any]
    chunk: Any | None = None
    prepared: bool = False
    cancelled: bool = False
    last_batch_size: int = 1


class _JsonlTrace:
    """Process-local, thread-safe JSONL sink enabled only by environment."""

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._handle: Any | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(dict(payload), sort_keys=True, default=str)
        with self._lock:
            if self._handle is None:
                self._handle = self.path.open("a", encoding="utf-8")
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
                self._handle.close()


class _RuntimeStepBackend:
    """Adapt public step methods to the legacy runtime backend shape."""

    def __init__(self, adapter: FlowChunkExecutionAdapter) -> None:
        self.adapter = adapter

    def advance_step(self, state: Any) -> Any:
        return self.adapter.advance_step(state)

    def advance_step_batch(self, states: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.adapter.advance_step_batch(states)

    def finish(self, state: Any) -> Any:
        return None


class OnlineFlowStepCoordinator:
    """Schedule one exact-shape Euler step at a time across logical chunks."""

    def __init__(
        self,
        *,
        adapter: FlowChunkExecutionAdapter,
        max_batch_size: int = 2,
        max_batch_wait_ms: float = 2.0,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
        time_ns: Callable[[], int] | None = None,
        start: bool = True,
    ) -> None:
        if isinstance(max_batch_size, bool) or max_batch_size not in (1, 2, 4):
            raise OnlineFlowStepCoordinatorError("max_batch_size must be one of 1, 2, or 4")
        if isinstance(max_batch_wait_ms, bool) or float(max_batch_wait_ms) < 0:
            raise OnlineFlowStepCoordinatorError("max_batch_wait_ms must be non-negative")
        for name in ("prepare_chunk", "make_step_item", "advance_step", "advance_step_batch", "update_step", "finalize_chunk", "cancel"):
            if not callable(getattr(adapter, name, None)):
                raise OnlineFlowStepCoordinatorError(f"adapter must expose public {name}()")
        self.adapter = adapter
        self.max_batch_size = int(max_batch_size)
        self.max_batch_wait_ms = float(max_batch_wait_ms)
        self._time_ns = time_ns or time.monotonic_ns
        self._external_sink = event_sink
        configured_n_raw = os.environ.get("DUPLEXPILOT_APR_LOGICAL_N", "").strip()
        if configured_n_raw:
            try:
                configured_n = int(configured_n_raw)
            except ValueError as exc:
                raise OnlineFlowStepCoordinatorError(
                    "DUPLEXPILOT_APR_LOGICAL_N must be a positive integer"
                ) from exc
            if configured_n <= 0:
                raise OnlineFlowStepCoordinatorError(
                    "DUPLEXPILOT_APR_LOGICAL_N must be a positive integer"
                )
            self._configured_logical_n: int | None = configured_n
        else:
            self._configured_logical_n = None
        self._trace = _JsonlTrace(os.environ.get("DUPLEXPILOT_APR_ONLINE_TRACE_PATH"))
        self._preemption_trace = _JsonlTrace(
            os.environ.get("DUPLEXPILOT_APR_PREEMPTION_TRACE_PATH")
        )
        self._opportunity_trace = _JsonlTrace(
            os.environ.get("DUPLEXPILOT_APR_OPPORTUNITY_TRACE_PATH")
        )
        opportunity_observer = None
        if self._opportunity_trace.path is not None:
            opportunity_observer = FlowBatchOpportunityObserver(
                event_sink=self._emit_opportunity,
                time_ns=self._time_ns,
            )
        self._condition = threading.Condition()
        self._ingress: deque[_Submission] = deque()
        self._active: dict[str, _ActiveChunk] = {}
        # A realtime session can be registered long before its next acoustic
        # chunk is ready.  Keep that logical-session cardinality separate from
        # the transient ready/in-flight set used by the scheduler telemetry.
        self._registered_requests: set[str] = set()
        self._stopping = False
        self._started = False
        self._auto_start = bool(start)
        self._thread: threading.Thread | None = None
        self._runtime = FlowBatchRuntime(
            backend=_RuntimeStepBackend(adapter),
            max_batch_size=self.max_batch_size,
            max_batch_wait_ms=self.max_batch_wait_ms,
            event_sink=self._emit,
            time_ns=self._time_ns,
            timing_enabled=self._trace.path is not None
            or self._preemption_trace.path is not None
            or self._opportunity_trace.path is not None,
            opportunity_observer=opportunity_observer,
        )
        if start:
            self.start()

    @property
    def scheduler(self):
        return self._runtime.scheduler

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            if self._stopping:
                raise OnlineFlowStepCoordinatorError("coordinator is closed")
            self._started = True
            self._thread = threading.Thread(target=self._run, name="apr-online-flow-step", daemon=True)
            self._thread.start()
            self._condition.notify_all()

    def register(self, request_id: str, *, stream_id: str, generation_id: int) -> None:
        request_id = str(request_id)
        self._registered_requests.add(request_id)
        if callable(getattr(self.adapter, "register", None)):
            self.adapter.register(request_id, stream_id=str(stream_id), generation_id=int(generation_id))
        self._emit(
            "REQUEST_REGISTER",
            (),
            request_ids=[request_id],
            generation_id=[int(generation_id)],
            sequence_no=[None],
            token_length=[0],
            last_chunk=[False],
            step_index=-1,
            shape_signature=[],
            stream_id=str(stream_id),
            pending_depth=len(self._ingress) + len(self._runtime.scheduler.pending_ids()),
        )

    def submit_async(self, batch: AcousticTokenBatch) -> Future[Any]:
        if not isinstance(batch, AcousticTokenBatch):
            raise OnlineFlowStepCoordinatorError("submit_async requires AcousticTokenBatch")
        future: Future[Any] = Future()
        with self._condition:
            self._registered_requests.add(batch.request_id)
            if self._stopping:
                raise OnlineFlowStepCoordinatorError("coordinator is closed")
            if batch.request_id in self._active:
                raise OnlineFlowStepCoordinatorError(
                    f"request already has an acoustic step in flight: {batch.request_id}"
                )
            self._active[batch.request_id] = _ActiveChunk(batch=batch, future=future)
            self._ingress.append(_Submission(batch=batch, future=future))
            self._emit("ONLINE_CHUNK_ENQUEUE", (), batch=batch, pending_depth=len(self._ingress))
            self._emit(
                "ACOUSTIC_CHUNK_READY",
                (),
                batch=batch,
                step_index=0,
                remaining_steps=None,
                first_audio=bool(batch.sequence_no == 0),
            )
            if self._auto_start and not self._started:
                self.start()
            self._condition.notify_all()
        return future

    def cancel(self, request_id: str, generation_id: int) -> bool:
        request_id = str(request_id)
        with self._condition:
            active = self._active.get(request_id)
            if active is None:
                self._registered_requests.discard(request_id)
                self._runtime.reset_request(request_id)
                self.adapter.cancel(request_id)
                return False
            if active.batch.generation_id != int(generation_id):
                raise OnlineFlowStepCoordinatorError(f"generation mismatch at cancel for {request_id}")
            active.cancelled = True
            if not active.future.done():
                active.future.cancel()
            try:
                self._runtime.cancel(request_id, generation_id=int(generation_id))
            except FlowBatchRuntimeError:
                pass
            self.adapter.cancel(request_id)
            self._active.pop(request_id, None)
            self._registered_requests.discard(request_id)
            self._emit("ONLINE_CANCEL_ACK", (), batch=active.batch, pending_depth=len(self._ingress))
            self._condition.notify_all()
            return True

    def reset(self, request_id: str, stream_id: str, generation_id: int) -> None:
        request_id = str(request_id)
        with self._condition:
            active = self._active.get(request_id)
            if active is not None:
                if not active.future.done():
                    active.future.cancel()
                active.cancelled = True
                self.adapter.cancel(request_id)
                self._active.pop(request_id, None)
            self._registered_requests.discard(request_id)
            self._ingress = deque(item for item in self._ingress if item.batch.request_id != request_id)
            self._runtime.reset_request(request_id)
        self.register(request_id, stream_id=stream_id, generation_id=generation_id)

    def close(self) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            for submission in tuple(self._ingress):
                if not submission.future.done():
                    submission.future.cancel()
            self._ingress.clear()
            for active in self._active.values():
                active.cancelled = True
                if not active.future.done():
                    active.future.cancel()
            self._condition.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10.0)
        for request_id in tuple(self._active):
            try:
                self.adapter.cancel(request_id)
            except Exception:
                pass
        self._active.clear()
        self._registered_requests.clear()
        self._trace.close()
        self._preemption_trace.close()
        self._opportunity_trace.close()

    def _normalize_prepared(self, batch: AcousticTokenBatch, value: Any) -> Any:
        if isinstance(value, PreparedFlowChunk):
            if value.batch != batch:
                raise OnlineFlowStepCoordinatorError("adapter returned a different batch identity")
            return value
        if isinstance(value, Mapping):
            required = ("batch", "state", "n_timesteps", "worker_id")
            if any(name not in value for name in required):
                raise OnlineFlowStepCoordinatorError("prepared chunk is missing public fields")
            if value["batch"] != batch:
                raise OnlineFlowStepCoordinatorError("adapter returned a different batch identity")
            return value
        raise OnlineFlowStepCoordinatorError("adapter returned an invalid prepared chunk")

    @staticmethod
    def _chunk_value(chunk: Any, name: str) -> Any:
        if isinstance(chunk, Mapping):
            return chunk[name]
        return getattr(chunk, name)

    @staticmethod
    def _set_chunk_state(chunk: Any, state: Any) -> None:
        if isinstance(chunk, Mapping):
            chunk["state"] = state
        else:
            chunk.state = state

    def _prepare(self, submission: _Submission) -> None:
        batch = submission.batch
        active = self._active.get(batch.request_id)
        if active is None or active.cancelled or submission.future.cancelled():
            return
        try:
            chunk = self._normalize_prepared(batch, self.adapter.prepare_chunk(batch))
            item = self.adapter.make_step_item(chunk)
            if not isinstance(item, FlowStepItem):
                raise OnlineFlowStepCoordinatorError("adapter returned an invalid FlowStepItem")
            item = FlowStepItem(
                request_id=item.request_id, generation_id=item.generation_id, version=item.version,
                state=item.state, ready_at_ns=self._time_ns(), model_identity=item.model_identity,
                device=item.device, dtype=item.dtype, step_index=item.step_index,
                shape_signature=item.shape_signature, last_chunk=item.last_chunk, n_timesteps=item.n_timesteps,
            )
            if item.request_id != batch.request_id or item.generation_id != batch.generation_id:
                raise OnlineFlowStepCoordinatorError("adapter FlowStepItem identity mismatch")
            active.chunk = chunk
            active.prepared = True
            self._runtime.submit(item)
            self._emit("ONLINE_CHUNK_PREPARED", (item,), batch=batch, pending_depth=len(self._ingress))
        except Exception as exc:
            self._fail_request(batch.request_id, exc)

    def _fail_request(self, request_id: str, error: BaseException) -> None:
        active = self._active.get(request_id)
        if active is None:
            return
        try:
            self.adapter.cancel(request_id)
        except Exception:
            pass
        self._runtime.reset_request(request_id)
        self._active.pop(request_id, None)
        if not active.future.done():
            active.future.set_exception(error)

    def _complete_result(self, active: _ActiveChunk) -> None:
        if active.cancelled or active.future.cancelled():
            return
        chunk = active.chunk
        if chunk is None:
            raise OnlineFlowStepCoordinatorError("cannot finalize an unprepared chunk")
        finalize_start_ns = self._time_ns()
        self._emit(
            "FLOW_FINALIZE_START",
            (),
            batch=active.batch,
            step_index=int(self._chunk_value(chunk, "n_timesteps")),
        )
        result = self.adapter.finalize_chunk(chunk)
        finalize_end_ns = self._time_ns()
        self._emit(
            "FLOW_FINALIZE_END",
            (),
            batch=active.batch,
            step_index=int(self._chunk_value(chunk, "n_timesteps")),
            wall_time_ms=max(0.0, (finalize_end_ns - finalize_start_ns) / 1_000_000.0),
        )
        if getattr(result, "request_id", active.batch.request_id) != active.batch.request_id:
            raise OnlineFlowStepCoordinatorError("adapter finalized the wrong request")
        try:
            result = replace(result, flow_batch_size=active.last_batch_size)
        except (TypeError, ValueError):
            pass
        next_version = int(
            getattr(result, "state_version", active.batch.state_version + 1)
        )
        self._runtime.update_version(
            active.batch.request_id,
            generation_id=active.batch.generation_id,
            version=next_version,
        )
        self._active.pop(active.batch.request_id, None)
        active.future.set_result(result)
        self._emit(
            "PCM_READY",
            (),
            batch=active.batch,
            step_index=int(self._chunk_value(chunk, "n_timesteps")),
            pcm_nonempty=bool(getattr(result, "pcm_records", ())),
        )
        self._emit(
            "REQUEST_FINISH",
            (),
            batch=active.batch,
            step_index=int(self._chunk_value(chunk, "n_timesteps")),
        )
        self._emit("ONLINE_CHUNK_FINALIZE", (), batch=active.batch, pending_depth=len(self._ingress))

    def _handle_execution(self, result: FlowBatchExecutionResult) -> None:
        if not result.committed:
            for item in result.items:
                active = self._active.get(item.request_id)
                if active is not None and not active.cancelled and not active.future.done():
                    self._fail_request(item.request_id, OnlineFlowStepCoordinatorError("stale Flow result was dropped"))
            return
        for item, next_state in zip(result.items, result.states):
            active = self._active.get(item.request_id)
            if active is None or active.cancelled or active.future.cancelled():
                continue
            if active.chunk is None:
                raise OnlineFlowStepCoordinatorError("missing prepared chunk")
            active.last_batch_size = int(result.batch_size)
            self.adapter.update_step(active.chunk, next_state)
            next_index = int(getattr(next_state, "step_index", item.step_index + 1))
            n_timesteps = int(self._chunk_value(active.chunk, "n_timesteps"))
            if next_index >= n_timesteps:
                self._complete_result(active)
                continue
            next_item = self.adapter.make_step_item(active.chunk)
            if not isinstance(next_item, FlowStepItem):
                raise OnlineFlowStepCoordinatorError("adapter returned an invalid requeued FlowStepItem")
            next_item = FlowStepItem(
                request_id=next_item.request_id, generation_id=next_item.generation_id,
                version=next_item.version, state=next_item.state, ready_at_ns=self._time_ns(),
                model_identity=next_item.model_identity, device=next_item.device, dtype=next_item.dtype,
                step_index=next_item.step_index, shape_signature=next_item.shape_signature,
                last_chunk=next_item.last_chunk, n_timesteps=next_item.n_timesteps,
            )
            self._runtime.submit(next_item)
            self._emit("ONLINE_STEP_REQUEUED", (next_item,), batch=active.batch, pending_depth=len(self._ingress))

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopping and not self._ingress and not self._runtime.scheduler.pending_ids():
                    return
                submissions = []
                # Bound one admission cohort.  In particular, do not let an
                # unbounded burst postpone already-ready Flow work.
                while self._ingress and len(submissions) < self.max_batch_size:
                    submissions.append(self._ingress.popleft())
            # Preparation can be materially slower than the batch wait
            # window.  Admit work that arrives while the first state is being
            # prepared before launching its step; otherwise the late request
            # can only meet the already-advanced next Euler step and loses the
            # same-step batching opportunity.  The cohort is bounded by the
            # physical batch cap, so this does not create an unbounded prepare
            # loop under continuous ingress.
            index = 0
            while index < len(submissions):
                self._prepare(submissions[index])
                index += 1
                if len(submissions) >= self.max_batch_size:
                    continue
                with self._condition:
                    while self._ingress and len(submissions) < self.max_batch_size:
                        submissions.append(self._ingress.popleft())
            try:
                result = self._runtime.run_next(now_ns=self._time_ns())
                if result is not None:
                    self._handle_execution(result)
                    continue
            except Exception as exc:
                request_ids = tuple(self._runtime.scheduler.pending_ids())
                if not request_ids:
                    request_ids = tuple(self._active)
                for request_id in request_ids:
                    self._fail_request(request_id, exc)
            with self._condition:
                if self._stopping and not self._ingress and not self._runtime.scheduler.pending_ids():
                    return
                self._condition.wait(timeout=0.001)

    def _emit(
        self,
        event: str | Mapping[str, Any],
        items: tuple[FlowStepItem, ...] = (),
        **fields: Any,
    ) -> dict[str, Any] | None:
        # FlowBatchRuntime sends a complete event mapping to its event sink,
        # while the coordinator emits named events directly.  Normalize both
        # forms before adding the online lifecycle fields.
        if isinstance(event, Mapping):
            incoming = dict(event)
            event = str(incoming.pop("event", incoming.pop("event_type", "FLOW_EVENT")))
            incoming.pop("event_type", None)
            incoming.update(fields)
            fields = incoming
        items = tuple(items or ())
        batch = fields.get("batch")
        batches = [self._active[item.request_id].batch for item in items if item.request_id in self._active]
        if not batches:
            batches = [
                self._active[request_id].batch
                for request_id in fields.get("request_ids", ())
                if request_id in self._active
            ]
        if not batches and batch is not None:
            batches = [batch]
        request_ids = [item.request_id for item in items]
        if not request_ids:
            request_ids = list(fields.get("request_ids", ()))
        if not request_ids and batch is not None:
            request_ids = [batch.request_id]
        generation_ids = [item.generation_id for item in items]
        if not generation_ids and batches:
            generation_ids = [item.generation_id for item in batches]
        if not generation_ids and batch is not None:
            generation_ids = [batch.generation_id]
        if not generation_ids and fields.get("generation_ids") is not None:
            generation_ids = list(fields["generation_ids"])
        state_versions = [item.version for item in items]
        if not state_versions and batches:
            state_versions = [item.state_version for item in batches]
        if not state_versions and fields.get("versions") is not None:
            state_versions = list(fields["versions"])
        last_chunks = [item.last_chunk for item in items]
        if not last_chunks and batches:
            last_chunks = [item.last_chunk for item in batches]
        token_lengths = [len(batch.stoken_ids) for batch in batches]
        sequence_numbers = [batch.sequence_no for batch in batches]
        payload: dict[str, Any] = {
            "event": event,
            "event_type": event,
            "timestamp_monotonic_ns": int(self._time_ns()),
            "request_ids": request_ids,
            "generation_id": generation_ids,
            "sequence_no": sequence_numbers,
            "state_version": state_versions,
            "token_length": token_lengths,
            "last_chunk": last_chunks,
            "step_index": items[0].step_index if items else fields.get("step_index"),
            "shape_signature": list(items[0].shape_signature) if items else fields.get("shape_signature"),
            "pending_depth": len(self._ingress) + len(self._runtime.scheduler.pending_ids()),
            "batch_size": len(items) if items else int(fields.get("batch_size", 0)),
            "wait_ms": fields.get("wait_ms", 0.0),
            "CUDA_time": fields.get("cuda_time_ms", fields.get("CUDA_time", 0.0)),
            "compatibility_rejection_counts": fields.get("compatibility_rejection_counts", {}),
            "N": self._configured_logical_n or len(self._active),
            "configured_N": self._configured_logical_n,
            "registered_sessions": len(self._registered_requests),
            "logical_concurrency": len(self._registered_requests) or len(self._active),
            "active_sessions": len(self._registered_requests) or len(self._active),
        }
        payload.update({key: value for key, value in fields.items() if key not in {"batch"}})
        if payload.get("logical_concurrency") is None:
            payload["logical_concurrency"] = len(self._registered_requests) or len(self._active)
        if payload.get("active_sessions") is None:
            payload["active_sessions"] = len(self._registered_requests) or len(self._active)
        self._trace.write(payload)
        if self._preemption_trace.path != self._trace.path:
            self._preemption_trace.write(payload)
        if self._external_sink is not None:
            try:
                self._external_sink(payload)
            except Exception:
                pass
        return payload

    def _emit_opportunity(self, event: Mapping[str, Any]) -> None:
        """Normalize observer telemetry and persist it in its own JSONL sink."""

        payload = self._emit(event)
        if payload is None:
            return
        if self._opportunity_trace.path in {
            self._trace.path,
            self._preemption_trace.path,
        }:
            return
        self._opportunity_trace.write(payload)


__all__ = [
    "FlowChunkExecutionAdapter",
    "OnlineFlowStepCoordinator",
    "OnlineFlowStepCoordinatorError",
    "PreparedFlowChunk",
]
