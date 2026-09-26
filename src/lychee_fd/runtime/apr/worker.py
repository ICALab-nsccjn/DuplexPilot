"""One isolated acoustic progression worker slot."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Callable, Mapping

from .acoustic_backend import AcousticBackend
from .contracts import AcousticTokenBatch
from .nvtx import range as nvtx_range
from .profiling import StageProfiler, estimate_state_size_bytes
from .state_store import APRStateError, AcousticCommitResult, AcousticStateStore


class APRWorkerError(RuntimeError):
    """Raised when an acoustic transition cannot be committed safely."""


class AcousticWorker:
    """Materialize logical state, process one batch, and commit atomically."""

    def __init__(
        self,
        state_store: AcousticStateStore,
        backend: AcousticBackend,
        *,
        worker_slot: int = 0,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if isinstance(worker_slot, bool) or not isinstance(worker_slot, int) or worker_slot < 0:
            raise APRWorkerError("worker_slot must be a non-negative integer")
        if not isinstance(backend, AcousticBackend):
            raise APRWorkerError("backend must implement AcousticBackend")
        self._state_store = state_store
        self._backend = backend
        self._worker_slot = worker_slot
        self._event_sink = event_sink
        if profiler is not None and not isinstance(profiler, StageProfiler):
            raise APRWorkerError("profiler must be a StageProfiler")
        self._profiler = profiler

    def _profile_span(
        self,
        stage: str,
        batch: AcousticTokenBatch,
        *,
        queue_depth: int | None = None,
    ):
        profiler = self._profiler
        if profiler is None:
            return nullcontext()
        return profiler.span(
            stage,
            session_id=batch.request_id,
            generation_id=batch.generation_id,
            sequence_no=batch.sequence_no,
            state_version=batch.state_version,
            worker_id=self._worker_slot,
            queue_depth=queue_depth,
        )

    def _emit(self, event_type: str, batch: AcousticTokenBatch, **fields: Any) -> None:
        if self._event_sink is None:
            return
        event = {
            "event_type": event_type,
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "request_id": batch.request_id,
            "stream_id": batch.stream_id,
            "generation_id": batch.generation_id,
            "sequence_no": batch.sequence_no,
            "state_version": batch.state_version,
            "worker_slot": self._worker_slot,
        }
        event.update(fields)
        try:
            self._event_sink(event)
        except Exception:
            # Observability must not alter the serving result.
            return

    def process(self, batch: AcousticTokenBatch) -> AcousticCommitResult:
        if not isinstance(batch, AcousticTokenBatch):
            raise APRWorkerError("APR worker accepts AcousticTokenBatch values only")
        lease = None
        try:
            with self._profile_span("STATE_ACQUIRE", batch):
                lease = self._state_store.acquire(
                    batch.request_id,
                    generation_id=batch.generation_id,
                    expected_version=batch.state_version,
                )
            self._emit("APR_STATE_ACQUIRE", batch, lease_state_version=lease.state_version)
            snapshot = self._state_store.snapshot(batch.request_id)
            if snapshot is None:
                raise APRWorkerError("logical acoustic state disappeared after acquire")
            self._emit("APR_ACOUSTIC_START", batch)
            with self._profile_span("STATE_RESTORE", batch):
                self._backend.restore_state(snapshot["state"])
            self._emit("APR_STATE_RESTORE", batch, restored_state_version=snapshot["state_version"])
            self._backend.resume()
            with self._profile_span("BACKEND_PROCESS", batch):
                self._backend.process(batch.stoken_ids)
            with self._profile_span("PCM_COMMIT", batch):
                pcm_records = self._backend.commit_pcm()
            capture_start_ns = time.monotonic_ns()
            next_state = None
            try:
                with nvtx_range("APR_CHECKPOINT"):
                    next_state = self._backend.capture_state(batch.request_id)
            finally:
                if self._profiler is not None:
                    try:
                        self._profiler.record_external_span(
                            "STATE_CAPTURE",
                            start_monotonic_ns=capture_start_ns,
                            end_monotonic_ns=time.monotonic_ns(),
                            session_id=batch.request_id,
                            generation_id=batch.generation_id,
                            sequence_no=batch.sequence_no,
                            state_version=batch.state_version,
                            worker_id=self._worker_slot,
                            state_size_bytes=(
                                estimate_state_size_bytes(next_state)
                                if isinstance(next_state, Mapping)
                                else None
                            ),
                        )
                    except Exception:
                        pass
            if not isinstance(next_state, Mapping):
                raise APRWorkerError("acoustic backend capture_state must return a mapping")
            with self._profile_span("STATE_COMMIT", batch):
                result = self._state_store.commit(lease, next_state, pcm_records)
            self._emit("APR_STATE_COMMIT", batch, committed_state_version=result.state_version)
            self._emit("APR_PCM_COMMIT", batch, pcm_count=len(result.pcm_records))
            self._emit(
                "APR_ACOUSTIC_END",
                batch,
                committed_state_version=result.state_version,
                pcm_count=len(result.pcm_records),
            )
            return result
        except Exception as exc:
            if isinstance(exc, APRStateError):
                self._emit("APR_STALE_OUTPUT", batch, error_type=type(exc).__name__, error=str(exc))
            self._emit("APR_ERROR", batch, error_type=type(exc).__name__, error=str(exc))
            if isinstance(exc, APRWorkerError):
                raise
            if isinstance(exc, APRStateError):
                raise APRWorkerError(str(exc)) from exc
            raise APRWorkerError(f"acoustic worker failed: {exc}") from exc
        finally:
            if lease is not None:
                self._state_store.release(batch.request_id)
