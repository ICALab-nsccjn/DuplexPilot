"""Small APR coordinator bridging token ingress to acoustic worker progress.

The runtime is intentionally a polling coordinator for the prototype.  Enqueueing
an acoustic token batch only records a request-owned item and marks its queue
ready; acoustic work happens later in process_one().  APR is opt-in so the
legacy synchronous handoff remains the default integration path.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Callable

from .contracts import AcousticTokenBatch, _require_nonempty_text, _require_nonnegative_int
from .queue import APRBackpressure, AcousticIngressQueue
from .profiling import StageProfiler, StageSpan
from .scheduler import AcousticScheduler
from .state_store import AcousticCommitResult, AcousticStateStore
from .worker import AcousticBackend, AcousticWorker
from .worker_pool import AcousticWorkerPool


class APRRuntimeError(RuntimeError):
    """Raised when the APR coordinator cannot preserve request ownership."""


@dataclass
class _RequestRuntime:
    request_id: str
    generation_id: int
    queue: AcousticIngressQueue
    backend: AcousticBackend


class AprRuntime:
    """Coordinate request-owned acoustic queues and isolated worker progress."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        worker_count: int = 1,
        queue_capacity: int = 32,
        state_store: AcousticStateStore | None = None,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
        profiler: StageProfiler | None = None,
        legacy_submit: Callable[[AcousticTokenBatch], Any] | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise APRRuntimeError("enabled must be a boolean")
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int) or queue_capacity <= 0:
            raise APRRuntimeError("queue_capacity must be a positive integer")
        self._enabled = enabled
        self._queue_capacity = queue_capacity
        self._state_store = state_store or AcousticStateStore()
        self._worker_pool = AcousticWorkerPool(
            worker_count=worker_count,
            event_sink=event_sink,
        )
        self._scheduler = AcousticScheduler(worker_count=worker_count)
        self._event_sink = event_sink
        if profiler is not None and not isinstance(profiler, StageProfiler):
            raise APRRuntimeError("profiler must be a StageProfiler")
        self._profiler = profiler
        self._legacy_submit = legacy_submit
        self._requests: dict[str, _RequestRuntime] = {}
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def state_store(self) -> AcousticStateStore:
        return self._state_store

    def _record_schedule_span(
        self,
        batch: AcousticTokenBatch,
        *,
        start_monotonic_ns: int,
        end_monotonic_ns: int,
        worker_id: int,
        queue_depth: int,
    ) -> None:
        profiler = self._profiler
        if profiler is None or not profiler.enabled:
            return
        context = profiler.run_context
        profiler.record(
            StageSpan(
                run_id=str(context["run_id"]),
                system=str(context["system"]),
                workload=str(context["workload"]),
                concurrency=int(context["concurrency"]),
                repeat=int(context["repeat"]),
                session_id=batch.request_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                state_version=batch.state_version,
                worker_id=worker_id,
                stage="APR_SCHEDULE_WAIT",
                start_monotonic_ns=start_monotonic_ns,
                end_monotonic_ns=end_monotonic_ns,
                queue_depth=queue_depth,
            )
        )

    def _emit_event(
        self,
        event_type: str,
        *,
        batch: AcousticTokenBatch | None = None,
        request_id: str | None = None,
        **fields: Any,
    ) -> None:
        if self._event_sink is None:
            return
        event = dict(fields)
        event["event_type"] = event_type
        event["timestamp_monotonic_ns"] = time.monotonic_ns()
        if batch is not None:
            event.update({
                "request_id": batch.request_id,
                "stream_id": batch.stream_id,
                "generation_id": batch.generation_id,
                "sequence_no": batch.sequence_no,
                "state_version": batch.state_version,
            })
        elif request_id is not None:
            event["request_id"] = request_id
        try:
            self._event_sink(event)
        except Exception:
            # Observability must not alter the serving result.
            return

    def start_request(
        self,
        request_id: str,
        backend: AcousticBackend,
        *,
        generation_id: int = 0,
        queue_capacity: int | None = None,
    ) -> None:
        """Register one logical request before its first token handoff."""
        if not self._enabled:
            return
        request_id = _require_nonempty_text("request_id", request_id)
        if not isinstance(backend, AcousticBackend):
            raise APRRuntimeError("APR request requires an AcousticBackend")
        generation_id = _require_nonnegative_int("generation_id", generation_id)
        capacity = self._queue_capacity if queue_capacity is None else queue_capacity
        queue = AcousticIngressQueue(request_id=request_id, capacity=capacity)
        with self._lock:
            if request_id in self._requests:
                raise APRRuntimeError(f"request already started: {request_id}")
            self._requests[request_id] = _RequestRuntime(
                request_id=request_id,
                generation_id=generation_id,
                queue=queue,
                backend=backend,
            )
            try:
                self._scheduler.register(request_id, queue)
            except Exception:
                del self._requests[request_id]
                raise

    def enqueue_token_batch(self, batch: AcousticTokenBatch) -> Any:
        """Enqueue without doing acoustic work when APR is enabled."""
        if not isinstance(batch, AcousticTokenBatch):
            raise APRRuntimeError("APR runtime accepts AcousticTokenBatch values only")
        if not self._enabled:
            if self._legacy_submit is None:
                raise APRRuntimeError("APR is disabled and no legacy_submit callback was provided")
            return self._legacy_submit(batch)
        with self._lock:
            request = self._requests.get(batch.request_id)
            if request is None:
                raise APRRuntimeError(f"request is not registered: {batch.request_id}")
            if batch.generation_id != request.generation_id:
                raise APRRuntimeError(
                    f"generation mismatch for {batch.request_id}: "
                    f"expected {request.generation_id}, got {batch.generation_id}"
                )
            try:
                request.queue.put(batch)
            except APRBackpressure as exc:
                self._emit_event(
                    "APR_BACKPRESSURE",
                    batch=batch,
                    error=str(exc),
                    queue_depth=request.queue.depth(),
                )
                raise
            self._scheduler.mark_ready(batch.request_id)
            self._emit_event("APR_TOKEN_ENQUEUED", batch=batch)
            self._emit_event(
                "APR_QUEUE_DEPTH",
                batch=batch,
                queue_depth=request.queue.depth(),
            )
            return None

    def process_one(self) -> AcousticCommitResult | None:
        """Consume at most one queued batch; return None when no work is ready."""
        if not self._enabled:
            return None
        schedule_start_ns = time.monotonic_ns()
        request_id = self._scheduler.next_ready()
        schedule_end_ns = time.monotonic_ns()
        if request_id is None:
            return None
        with self._lock:
            request = self._requests.get(request_id)
        if request is None:
            self._scheduler.release(request_id)
            raise APRRuntimeError(f"scheduled request disappeared: {request_id}")
        lease = self._worker_pool.acquire()
        try:
            self._emit_event(
                "APR_ACOUSTIC_SCHEDULE",
                request_id=request_id,
                worker_id=lease.worker_id,
                queue_depth=request.queue.depth(),
            )
            batch = request.queue.get()
            if batch is None:
                return None
            self._record_schedule_span(
                batch,
                start_monotonic_ns=schedule_start_ns,
                end_monotonic_ns=schedule_end_ns,
                worker_id=lease.worker_id,
                queue_depth=request.queue.depth(),
            )
            self._emit_event(
                "APR_TOKEN_DEQUEUED",
                batch=batch,
                queue_depth=request.queue.depth(),
            )
            worker = AcousticWorker(
                self._state_store,
                request.backend,
                worker_slot=lease.worker_id,
                event_sink=self._event_sink,
                profiler=self._profiler,
            )
            result = worker.process(batch)
            return replace(result, worker_id=lease.worker_id)
        finally:
            self._worker_pool.release(lease)
            self._scheduler.release(request_id)

    def queue_depth(self, request_id: str) -> int:
        request_id = _require_nonempty_text("request_id", request_id)
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise APRRuntimeError(f"request is not registered: {request_id}")
            return request.queue.depth()

    def shutdown_request(self, request_id: str, *, generation_id: int) -> None:
        """Cancel queued generation and remove logical state after terminal cleanup."""
        request_id = _require_nonempty_text("request_id", request_id)
        generation_id = _require_nonnegative_int("generation_id", generation_id)
        if not self._enabled:
            return
        with self._lock:
            request = self._requests.pop(request_id, None)
            if request is None:
                return
            self._emit_event(
                "APR_CANCEL",
                request_id=request_id,
                generation_id=generation_id,
            )
            request.queue.cancel_generation(generation_id)
            request.queue.close()
            self._scheduler.unregister(request_id)
            snapshot = self._state_store.snapshot(request_id)
            if snapshot is not None:
                self._state_store.cancel(request_id, generation_id)
                self._state_store.remove(request_id)

    def close(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            request_ids = list(self._requests)
        for request_id in request_ids:
            request = self._requests.get(request_id)
            generation_id = request.generation_id if request is not None else 0
            self.shutdown_request(request_id, generation_id=generation_id)
