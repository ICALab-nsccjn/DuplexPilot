"""Benchmark-only physical-affinity and elastic acoustic lanes.

The lanes in this module are deliberately additive.  They share the model
weights, but keep caller-owned Token2Wav continuation state in a logical
request object and execute it under a real :class:`AcousticWorkerPoolV2`
context.  ``ElasticAcousticLaneV2`` performs handoff only at a completed
chunk boundary; it never interrupts a CUDA kernel.

This is a narrow capacity/SLO experiment boundary.  The historical lanes and
the default paper systems are left untouched unless a new system spec
explicitly selects one of these lanes.
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from lychee_fd.runtime.apr.contracts import AcousticPcmRecord, AcousticTokenBatch
from lychee_fd.runtime.apr.elastic_worker_pool import (
    AcousticWorkerPoolV2,
    PhysicalAcousticWorkerContext,
    PhysicalWorkerError,
    PhysicalWorkerLease,
)
from lychee_fd.runtime.apr.worker_handoff import (
    HandoffError,
    LogicalAcousticSessionState,
    WorkerHandoffController,
)
from lychee_fd.runtime.token2wav_checkpoint import clone_state_value
from tools.apr.real_acoustic_lanes import (
    AcousticLaneProgress,
    _acoustic_cuda_device_context,
    resolve_lane_identities,
)


class CapacitySloLaneError(RuntimeError):
    """Raised when a capacity lane cannot preserve logical ownership."""


@dataclass(frozen=True)
class CapacityLaneProgress:
    """One atomically committed chunk plus physical execution provenance."""

    request_id: str
    worker_id: int
    pcm_records: tuple[AcousticPcmRecord, ...]
    state_version: int
    checkpoint_count: int
    restore_count: int
    worker_switch: bool
    worker_instance_id: str
    execution_context_id: str
    source_worker_instance_id: str
    source_execution_context_id: str
    handoff_latency_ns: int = 0
    capture_latency_ns: int = 0
    restore_latency_ns: int = 0
    flow_steps: int = 10
    flow_wall_time_ns: int = 0
    flow_cuda_time_ns: int = 0
    flow_batch_size: int = 1

    @property
    def target_worker_instance_id(self) -> str:
        """Stable instance identity of the context that executed the chunk."""
        return self.worker_instance_id

    @property
    def target_execution_context_id(self) -> str:
        """Stable execution-context identity of the target context."""
        return self.execution_context_id


@dataclass
class _LogicalRequest:
    request_id: str
    stream_id: str
    generation_id: int
    prompt_wav: str
    stream_state: dict[str, Any]
    logical_state: LogicalAcousticSessionState
    assigned_worker_id: int | None = None
    last_worker_id: int | None = None
    last_worker_instance_id: str | None = None
    last_execution_context_id: str | None = None
    last_checkpoint: Any | None = None
    next_sequence: int = 0
    next_version: int = 0
    pcm_seq: int = 0
    cancelled: bool = False


def _fingerprint(model: Any, prompt_wav: str, device: str) -> str:
    payload = f"{type(model).__module__}.{type(model).__qualname__}|{prompt_wav}|{device}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_summary(stream_state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep metadata separate from the one full logical stream snapshot."""
    flow = stream_state.get("flow_cache")
    hift = stream_state.get("hift_cache")
    return {
        "flow_cache_present": isinstance(flow, Mapping),
        "hift_cache_present": isinstance(hift, Mapping),
    }


def _sync_context(context: PhysicalAcousticWorkerContext) -> None:
    stream = getattr(context, "stream", None)
    synchronize = getattr(stream, "synchronize", None)
    if callable(synchronize):
        synchronize()


@contextmanager
def _model_execution(model: Any):
    """Use inference mode without changing the model's precision contract."""
    try:
        import torch
    except ImportError:  # pragma: no cover - the real container has torch
        yield
        return
    if torch.cuda.is_available():
        with torch.inference_mode():
            yield
    else:
        with torch.inference_mode():
            yield


class _CapacityLaneBase:
    """Shared request, queue, state, and public Token2Wav execution logic."""

    def __init__(
        self,
        *,
        worker_count: int,
        model: Any,
        prompt_wav: str,
        device: str | None = None,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
            raise CapacitySloLaneError("worker_count must be a positive integer")
        if not callable(getattr(model, "create_stream_state", None)):
            raise CapacitySloLaneError("model must expose create_stream_state()")
        if not callable(getattr(model, "stream_with_state", None)) and not callable(
            getattr(model, "begin_chunk_steps", None)
        ):
            raise CapacitySloLaneError(
                "model must expose stream_with_state() or public Flow-step APIs"
            )
        self.worker_count = worker_count
        self.model = model
        self.prompt_wav = str(prompt_wav)
        self.device = str(device) if device is not None else None
        self.events: list[dict[str, Any]] = []
        self._event_sink = event_sink
        self._model_lock = threading.RLock()
        self.pool = AcousticWorkerPoolV2(
            worker_count=worker_count,
            device=self.device,
            event_sink=self._emit_event,
            shared_execution_lock=self._model_lock,
        )
        self.handoff_controller = WorkerHandoffController(
            self.pool,
            event_sink=self._emit_event,
        )
        self.backends: dict[str, _LogicalRequest] = {}
        self.identity_for: dict[str, tuple[str, int]] = {}
        self.worker_for: dict[str, int] = {}
        self.last_worker: dict[str, int] = {}
        self._pending: deque[AcousticTokenBatch] = deque()
        self._pending_by_worker: dict[int, deque[AcousticTokenBatch]] = defaultdict(deque)
        self._pending_ids: set[str] = set()
        self._worker_cursor = 0
        self._closed = False
        self.ownership_errors = 0
        self.cleanup_ok = False
        self.handoff_count = 0
        self.checkpoint_count = 0
        self.restore_count = 0

    def _emit_event(self, event: dict[str, Any] | str, **fields: Any) -> None:
        if isinstance(event, str):
            payload = {"event_type": event, **fields}
        else:
            payload = dict(event)
            payload.update(fields)
            payload.setdefault("event_type", payload.get("event"))
        payload.setdefault("timestamp_monotonic_ns", time.monotonic_ns())
        self.events.append(payload)
        if self._event_sink is not None:
            try:
                self._event_sink(dict(payload))
            except Exception:
                # Telemetry cannot change the serving result.
                pass

    def _request(self, request_id: str) -> _LogicalRequest:
        try:
            return self.backends[str(request_id)]
        except KeyError as exc:
            raise CapacitySloLaneError(
                f"request is not admitted: {request_id}"
            ) from exc

    def start(
        self,
        request_ids: Iterable[str],
        *,
        identity_by_request: Mapping[str, tuple[str, int]] | None = None,
    ) -> tuple[str, ...]:
        ids = tuple(str(item) for item in request_ids)
        if not ids or len(set(ids)) != len(ids) or any(not item for item in ids):
            raise CapacitySloLaneError("request IDs must be unique and non-empty")
        if self._closed:
            raise CapacitySloLaneError("lane is closed")
        identities = resolve_lane_identities(ids, identity_by_request)
        with _acoustic_cuda_device_context():
            for index, request_id in enumerate(ids):
                if request_id in self.backends:
                    raise CapacitySloLaneError(f"request already started: {request_id}")
                stream_id, generation_id = identities[request_id]
                stream_state = self.model.create_stream_state(self.prompt_wav)
                if not isinstance(stream_state, dict):
                    raise CapacitySloLaneError(
                        "create_stream_state must return a mutable dictionary"
                    )
                logical = LogicalAcousticSessionState(
                    request_id=request_id,
                    stream_id=stream_id,
                    generation_id=generation_id,
                    sequence_no=0,
                    state_version=0,
                    flow_state=_state_summary(stream_state),
                    token2wav_state={
                        "prompt_wav": self.prompt_wav,
                        # The request object is the single owner of the full
                        # CUDA continuation state.  Keep a shallow logical
                        # view rather than a second tensor clone; checkpoint
                        # capture makes a detached copy only for an actual
                        # handoff transaction.
                        "stream_state": stream_state,
                    },
                    hift_state={"present": isinstance(stream_state.get("hift_cache"), Mapping)},
                    flush_state={"event_active": False, "last_chunk": False},
                )
                request = _LogicalRequest(
                    request_id=request_id,
                    stream_id=stream_id,
                    generation_id=generation_id,
                    prompt_wav=self.prompt_wav,
                    stream_state=stream_state,
                    logical_state=logical,
                )
                self._assign_initial_worker(request, index)
                self.backends[request_id] = request
                self.identity_for[request_id] = (stream_id, generation_id)
                self.worker_for[request_id] = int(request.assigned_worker_id or 0)
                self._emit_event(
                    "APR_CAPACITY_REQUEST_START",
                    request_id=request_id,
                    stream_id=stream_id,
                    generation_id=generation_id,
                    worker_id=request.assigned_worker_id,
                )
        return ids

    def _assign_initial_worker(self, request: _LogicalRequest, index: int) -> None:
        # ``OnlineAcousticRouter`` registers requests one at a time.  Use the
        # lane cursor rather than the per-call index so physical-affinity-v2
        # remains balanced for both batched and incremental registration.
        request.assigned_worker_id = self._worker_cursor % self.worker_count
        self._worker_cursor = (self._worker_cursor + 1) % self.worker_count

    def submit(self, batch: AcousticTokenBatch) -> None:
        if not isinstance(batch, AcousticTokenBatch):
            raise CapacitySloLaneError("lane accepts AcousticTokenBatch values only")
        request = self._request(batch.request_id)
        if request.cancelled:
            raise CapacitySloLaneError("cancelled request cannot accept tokens")
        if (batch.stream_id, batch.generation_id) != (
            request.stream_id,
            request.generation_id,
        ):
            raise CapacitySloLaneError(
                f"identity mismatch for {batch.request_id}"
            )
        if batch.request_id in self._pending_ids:
            raise CapacitySloLaneError(
                f"request already has a pending acoustic chunk: {batch.request_id}"
            )
        if batch.sequence_no != request.next_sequence:
            raise CapacitySloLaneError(
                f"sequence mismatch for {batch.request_id}: "
                f"expected={request.next_sequence} got={batch.sequence_no}"
            )
        if batch.state_version != request.next_version:
            raise CapacitySloLaneError(
                f"state version mismatch for {batch.request_id}: "
                f"expected={request.next_version} got={batch.state_version}"
            )
        self._enqueue(batch)
        self._pending_ids.add(batch.request_id)
        self._emit_event(
            "APR_CAPACITY_TOKEN_READY",
            request_id=batch.request_id,
            generation_id=batch.generation_id,
            sequence_no=batch.sequence_no,
            state_version=batch.state_version,
            last_chunk=batch.last_chunk,
            pending_depth=self.pending_depth,
        )

    @property
    def pending_depth(self) -> int:
        return len(self._pending_ids)

    def _enqueue(self, batch: AcousticTokenBatch) -> None:
        self._pending.append(batch)

    def _take_batch(self, target_worker_id: int | None) -> tuple[AcousticTokenBatch, int] | None:
        if not self._pending:
            return None
        batch = self._pending.popleft()
        request = self._request(batch.request_id)
        target = self._choose_target(request, target_worker_id)
        self._pending_ids.discard(batch.request_id)
        return batch, target

    def _choose_target(self, request: _LogicalRequest, target_worker_id: int | None) -> int:
        if target_worker_id is not None:
            if isinstance(target_worker_id, bool) or not isinstance(target_worker_id, int):
                raise CapacitySloLaneError("target_worker_id must be an integer")
            if target_worker_id < 0 or target_worker_id >= self.worker_count:
                raise CapacitySloLaneError("target_worker_id is out of range")
            return target_worker_id
        if request.last_worker_id is not None:
            return (request.last_worker_id + 1) % self.worker_count
        target = self._worker_cursor % self.worker_count
        self._worker_cursor += 1
        return target

    def _make_checkpoint(
        self,
        request: _LogicalRequest,
        context: PhysicalAcousticWorkerContext,
        lease: PhysicalWorkerLease,
    ):
        from lychee_fd.runtime.acoustic_checkpoint_state_v2 import AcousticCheckpointStateV2

        cuda_rng: tuple[Any, ...] = ()
        try:
            import torch

            cpu_rng = torch.get_rng_state()
            if torch.cuda.is_available():
                cuda_rng = (torch.cuda.get_rng_state(device=context.device),)
        except Exception:
            cpu_rng = b"capacity-v2-cpu-rng-unavailable"
        # Keep one complete caller-owned stream snapshot.  The other v2
        # sections are explicit summaries/provenance and do not duplicate the
        # potentially large Flow tensors.
        stream_snapshot = clone_state_value(request.stream_state)
        return AcousticCheckpointStateV2.capture(
            request_id=request.request_id,
            stream_id=request.stream_id,
            generation_id=request.generation_id,
            version=request.next_version,
            sequence_no=request.next_sequence,
            pcm_seq=request.pcm_seq,
            flow_state=_state_summary(stream_snapshot),
            token2wav_state={
                "prompt_wav": request.prompt_wav,
                "stream_state": stream_snapshot,
            },
            hift_state={"present": isinstance(stream_snapshot.get("hift_cache"), Mapping)},
            token_buffer=request.logical_state.token_buffer,
            tts_initialized=request.logical_state.tts_initialized,
            flush_state=request.logical_state.flush_state,
            pending_pcm=(),
            cpu_rng_state=cpu_rng,
            cuda_rng_state=cuda_rng,
            explicit_generator_state={},
            quiescence={
                "boundary": "chunk_commit",
                "in_flight_steps": 0,
                "pending_output": 0,
            },
            source_worker_id=lease.worker_id,
            source_worker_instance_id=lease.worker_instance_id,
            source_execution_context_id=context.execution_context_id,
            source_lease_id=lease.lease_id,
            compatibility_fingerprint=_fingerprint(
                self.model, request.prompt_wav, context.device
            ),
            cancelled=request.cancelled,
        )

    def _restore_checkpoint(
        self,
        request: _LogicalRequest,
        checkpoint: Any,
    ) -> None:
        """Restore every request-owned field carried by checkpoint v2.

        The physical context is intentionally absent from this method: the
        handoff controller owns stream/lease activation, while this method
        restores only logical continuation state.
        """
        if checkpoint.request_id != request.request_id:
            raise CapacitySloLaneError("v2 checkpoint request identity mismatch")
        if checkpoint.stream_id != request.stream_id:
            raise CapacitySloLaneError("v2 checkpoint stream identity mismatch")
        if checkpoint.generation_id != request.generation_id:
            raise CapacitySloLaneError("v2 checkpoint generation identity mismatch")
        payload = checkpoint.token2wav_state
        if not isinstance(payload, Mapping):
            raise CapacitySloLaneError("v2 checkpoint token2wav_state is invalid")
        if payload.get("prompt_wav") != request.prompt_wav:
            raise CapacitySloLaneError("v2 checkpoint prompt identity mismatch")
        stream_state = payload.get("stream_state")
        if not isinstance(stream_state, Mapping):
            raise CapacitySloLaneError("v2 checkpoint stream_state is missing")
        request.stream_state = clone_state_value(dict(stream_state))
        request.pcm_seq = int(checkpoint.pcm_seq)
        request.next_sequence = int(checkpoint.sequence_no)
        request.next_version = int(checkpoint.version)
        request.logical_state.flow_state = clone_state_value(dict(checkpoint.flow_state))
        # Rebind the logical view to the one restored request-owned mapping;
        # retaining another full clone here would double GPU cache residency.
        request.logical_state.token2wav_state = {
            "prompt_wav": request.prompt_wav,
            "stream_state": request.stream_state,
        }
        request.logical_state.hift_state = clone_state_value(dict(checkpoint.hift_state))
        request.logical_state.flush_state = clone_state_value(dict(checkpoint.flush_state))
        request.logical_state.token_buffer = tuple(checkpoint.token_buffer)
        request.logical_state.tts_initialized = bool(checkpoint.tts_initialized)
        request.logical_state.pending_output = list(checkpoint.pending_pcm)
        request.logical_state.sequence_no = int(checkpoint.sequence_no)
        request.logical_state.state_version = int(checkpoint.version)
        request.cancelled = bool(checkpoint.cancelled)
        request.logical_state.cancelled = request.cancelled

    def _execute_chunk(
        self,
        request: _LogicalRequest,
        batch: AcousticTokenBatch,
        context: PhysicalAcousticWorkerContext,
    ) -> tuple[bytes, int, int]:
        """Execute one complete safe chunk using public Token2Wav APIs."""
        setter = getattr(self.model, "set_profile_context", None)
        if callable(setter):
            setter(
                session_id=request.request_id,
                generation_id=request.generation_id,
                sequence_no=batch.sequence_no,
                state_version=batch.state_version,
                worker_id=context.worker_id,
                worker_instance_id=context.worker_instance_id,
                execution_context_id=context.execution_context_id,
            )
        start_ns = time.monotonic_ns()
        use_step_api = all(
            callable(getattr(self.model, name, None))
            for name in (
                "begin_chunk_steps",
                "advance_chunk_step",
                "finish_chunk_steps",
                "render_chunk_pcm",
            )
        )
        with _model_execution(self.model):
            if use_step_api:
                step_state = self.model.begin_chunk_steps(
                    batch.stoken_ids,
                    request.prompt_wav,
                    request.stream_state,
                    last_chunk=batch.last_chunk,
                    n_timesteps=10,
                    request_id=request.request_id,
                    generation_id=request.generation_id,
                    sequence_no=batch.sequence_no,
                    version=batch.state_version,
                )
                t_span = getattr(step_state, "t_span", None)
                steps = int(t_span.numel() - 1) if t_span is not None else 10
                if steps <= 0:
                    raise CapacitySloLaneError("Flow state has no executable steps")
                for _ in range(steps):
                    step_state = self.model.advance_chunk_step(step_state)
                finished = self.model.finish_chunk_steps(step_state)
                if not isinstance(finished, tuple) or len(finished) not in (2, 3):
                    raise CapacitySloLaneError("invalid public Flow finish result")
                mel = finished[0]
                if len(finished) == 2 and isinstance(finished[1], Mapping):
                    estimator_cache = dict(finished[1])
                elif len(finished) == 3:
                    estimator_cache = {
                        "estimator_cnn_cache": finished[1],
                        "estimator_att_cache": finished[2],
                    }
                else:
                    raise CapacitySloLaneError("invalid Flow estimator cache result")
                current_flow_cache = request.stream_state.get("flow_cache")
                if not isinstance(current_flow_cache, dict):
                    raise CapacitySloLaneError("stream state is missing flow_cache")
                updated_flow_cache = dict(current_flow_cache)
                updated_flow_cache.update(estimator_cache)
                request.stream_state["flow_cache"] = updated_flow_cache
                pcm = self.model.render_chunk_pcm(
                    mel,
                    request.stream_state,
                    last_chunk=batch.last_chunk,
                )
            else:
                steps = 10
                pcm = self.model.stream_with_state(
                    list(batch.stoken_ids),
                    request.prompt_wav,
                    request.stream_state,
                    last_chunk=batch.last_chunk,
                )
        _sync_context(context)
        elapsed_ns = time.monotonic_ns() - start_ns
        if not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise CapacitySloLaneError("Token2Wav returned non-bytes PCM")
        pcm_bytes = bytes(pcm)
        if batch.last_chunk and not pcm_bytes:
            raise CapacitySloLaneError("final chunk produced empty PCM")
        return pcm_bytes, steps, elapsed_ns

    def _commit_progress(
        self,
        request: _LogicalRequest,
        batch: AcousticTokenBatch,
        *,
        lease: PhysicalWorkerLease,
        context: PhysicalAcousticWorkerContext,
        pcm_bytes: bytes,
        flow_steps: int,
        flow_elapsed_ns: int,
        handoff_result: Any | None,
    ) -> CapacityLaneProgress:
        records: tuple[AcousticPcmRecord, ...]
        if pcm_bytes:
            records = (
                AcousticPcmRecord(
                    request_id=batch.request_id,
                    stream_id=batch.stream_id,
                    generation_id=batch.generation_id,
                    sequence_no=batch.sequence_no,
                    pcm_bytes=pcm_bytes,
                    sample_rate=24000,
                    pcm_seq=request.pcm_seq,
                ),
            )
            request.pcm_seq += 1
        else:
            records = ()
        for record in records:
            if (
                record.request_id != batch.request_id
                or record.stream_id != batch.stream_id
                or record.generation_id != batch.generation_id
                or record.sequence_no != batch.sequence_no
            ):
                self.ownership_errors += 1
                raise CapacitySloLaneError("PCM ownership mismatch")

        old_version = request.next_version
        request.next_version += 1
        request.next_sequence += 1
        logical = request.logical_state
        logical.commit_step(
            generation_id=batch.generation_id,
            expected_version=old_version,
            sequence_no=request.next_sequence,
            flow_state=_state_summary(request.stream_state),
            token2wav_state={
                "prompt_wav": request.prompt_wav,
                "stream_state": request.stream_state,
            },
        )
        logical.hift_state = {
            "present": isinstance(request.stream_state.get("hift_cache"), Mapping)
        }
        logical.tts_initialized = not batch.last_chunk
        logical.flush_state = {
            "event_active": not batch.last_chunk,
            "last_chunk": bool(batch.last_chunk),
            "pending_output": 0,
        }
        logical.pending_output.clear()
        # A physical-affinity request has no reason to snapshot its logical
        # state on every chunk.  A v2 snapshot is captured by the handoff
        # transaction itself, only when a request actually changes context.
        checkpoint = (
            handoff_result.checkpoint if handoff_result is not None else None
        )
        checkpoint_elapsed_ns = (
            int(handoff_result.capture_latency_ns)
            if handoff_result is not None
            else 0
        )
        checkpoint_count = 1 if handoff_result is not None else 0
        self.checkpoint_count += checkpoint_count
        # Do not retain a second full CUDA snapshot between chunks.  Future
        # handoffs capture the current state at their safe boundary.
        request.last_checkpoint = None
        previous_worker = request.last_worker_id
        previous_instance = request.last_worker_instance_id
        previous_context = request.last_execution_context_id
        request.last_worker_id = lease.worker_id
        request.last_worker_instance_id = lease.worker_instance_id
        request.last_execution_context_id = context.execution_context_id
        self.worker_for[request.request_id] = lease.worker_id
        self.last_worker[request.request_id] = lease.worker_id
        switched = previous_context is not None and previous_context != context.execution_context_id
        self._emit_event(
            "APR_CAPACITY_CHUNK_COMMIT",
            request_id=request.request_id,
            generation_id=request.generation_id,
            sequence_no=batch.sequence_no,
            state_version=request.next_version,
            worker_id=lease.worker_id,
            worker_instance_id=lease.worker_instance_id,
            execution_context_id=context.execution_context_id,
            source_worker_instance_id=previous_instance or lease.worker_instance_id,
            source_execution_context_id=previous_context or context.execution_context_id,
            worker_switch=switched,
            checkpoint_bytes=(
                checkpoint.logical_state_size_bytes() if checkpoint is not None else 0
            ),
            checkpoint_latency_ns=checkpoint_elapsed_ns,
        )
        return CapacityLaneProgress(
            request_id=request.request_id,
            worker_id=lease.worker_id,
            pcm_records=records,
            state_version=request.next_version,
            checkpoint_count=checkpoint_count,
            restore_count=1 if handoff_result is not None else 0,
            worker_switch=switched,
            worker_instance_id=lease.worker_instance_id,
            execution_context_id=context.execution_context_id,
            source_worker_instance_id=previous_instance or lease.worker_instance_id,
            source_execution_context_id=previous_context or context.execution_context_id,
            handoff_latency_ns=(
                int(handoff_result.total_latency_ns) if handoff_result is not None else 0
            ),
            capture_latency_ns=(
                int(handoff_result.capture_latency_ns) if handoff_result is not None else 0
            ),
            restore_latency_ns=(
                int(handoff_result.restore_latency_ns) if handoff_result is not None else 0
            ),
            flow_steps=flow_steps,
            flow_wall_time_ns=flow_elapsed_ns,
            flow_cuda_time_ns=0,
            flow_batch_size=1,
        )

    def _direct_lease(self, request: _LogicalRequest, target: int) -> PhysicalWorkerLease:
        try:
            return self.pool.acquire(target_worker_id=target, request_id=request.request_id)
        except PhysicalWorkerError as exc:
            raise CapacitySloLaneError(str(exc)) from exc

    def _process_request(
        self,
        batch: AcousticTokenBatch,
        target: int,
    ) -> CapacityLaneProgress:
        request = self._request(batch.request_id)
        if request.cancelled:
            raise CapacitySloLaneError("request was cancelled before execution")
        handoff_result = None
        lease: PhysicalWorkerLease | None = None
        try:
            if self._needs_handoff(request, target):
                try:
                    source_lease = self.pool.acquire(
                        target_worker_id=int(request.last_worker_id),
                        request_id=request.request_id,
                    )
                except PhysicalWorkerError as exc:
                    raise CapacitySloLaneError(
                        f"source context acquisition failed: {exc}"
                    ) from exc
                request.logical_state.lease = source_lease
                def capture(context: PhysicalAcousticWorkerContext, state: LogicalAcousticSessionState):
                    return self._make_checkpoint(request, context, source_lease)

                def restore(
                    _context: PhysicalAcousticWorkerContext,
                    checkpoint: Any,
                    _state: LogicalAcousticSessionState,
                ):
                    self._restore_checkpoint(request, checkpoint)

                handoff_result = self.handoff_controller.handoff(
                    state=request.logical_state,
                    source_lease=source_lease,
                    target_worker_id=target,
                    capture=capture,
                    restore=restore,
                    quiesce=lambda _context, _state: {
                        "boundary": "chunk_commit",
                        "in_flight_steps": 0,
                    },
                    drain_pending_output=lambda _context, state: state.pending_output.clear(),
                )
                lease = request.logical_state.lease
                self.handoff_count += 1
                self.restore_count += 1
                if lease is None:
                    raise CapacitySloLaneError("handoff returned no target lease")
            else:
                lease = self._direct_lease(request, target)
                request.logical_state.lease = lease
            context = self.pool.context_for(lease)
            self._emit_event(
                "APR_CAPACITY_EXECUTION_START",
                request_id=batch.request_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                worker_id=context.worker_id,
                worker_instance_id=context.worker_instance_id,
                execution_context_id=context.execution_context_id,
                handoff=handoff_result is not None,
            )
            with self.pool.execution(lease):
                pcm_bytes, flow_steps, flow_elapsed_ns = self._execute_chunk(
                    request, batch, context
                )
            progress = self._commit_progress(
                request,
                batch,
                lease=lease,
                context=context,
                pcm_bytes=pcm_bytes,
                flow_steps=flow_steps,
                flow_elapsed_ns=flow_elapsed_ns,
                handoff_result=handoff_result,
            )
            self._emit_event(
                "APR_CAPACITY_EXECUTION_END",
                request_id=batch.request_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                state_version=progress.state_version,
                worker_id=progress.worker_id,
                execution_context_id=progress.execution_context_id,
                worker_switch=progress.worker_switch,
            )
            return progress
        except (CapacitySloLaneError, HandoffError):
            raise
        except Exception as exc:
            raise CapacitySloLaneError(
                f"acoustic execution failed for {batch.request_id}: {exc}"
            ) from exc
        finally:
            if lease is not None:
                try:
                    self.pool.release(lease)
                except Exception as exc:
                    self._emit_event(
                        "APR_CAPACITY_RELEASE_ERROR",
                        request_id=batch.request_id,
                        error=str(exc),
                    )
                request.logical_state.lease = None

    def _needs_handoff(self, request: _LogicalRequest, target: int) -> bool:
        return False

    def process_one(self, *, target_worker_id: int | None = None) -> CapacityLaneProgress | None:
        selected = self._take_batch(target_worker_id)
        if selected is None:
            return None
        batch, target = selected
        return self._process_request(batch, target)

    def cancel(self, request_id: str) -> None:
        request_id = str(request_id)
        request = self.backends.get(request_id)
        if request is not None:
            request.cancelled = True
            request.logical_state.cancel()
        self._pending = deque(
            batch for batch in self._pending if batch.request_id != request_id
        )
        for worker_id, queue in tuple(self._pending_by_worker.items()):
            self._pending_by_worker[worker_id] = deque(
                batch for batch in queue if batch.request_id != request_id
            )
        self._pending_ids.discard(request_id)
        self.backends.pop(request_id, None)
        self.identity_for.pop(request_id, None)
        self.worker_for.pop(request_id, None)
        self.last_worker.pop(request_id, None)
        self._emit_event("APR_CAPACITY_CANCEL", request_id=request_id)

    def close(self) -> None:
        if self._closed:
            self.cleanup_ok = True
            return
        for request_id in tuple(self.backends):
            self.cancel(request_id)
        self._pending.clear()
        try:
            self.pool.close()
        except PhysicalWorkerError as exc:
            raise CapacitySloLaneError(str(exc)) from exc
        self._closed = True
        self.cleanup_ok = not self.backends and not self._pending and self.pool.active_count == 0


class PhysicalAffinityAcousticLaneV2(_CapacityLaneBase):
    """Physical affinity-v2 control using stable worker contexts."""

    def _enqueue(self, batch: AcousticTokenBatch) -> None:
        """Keep an independent FIFO for each fixed physical worker."""
        request = self._request(batch.request_id)
        worker_id = int(request.assigned_worker_id or 0)
        self._pending_by_worker[worker_id].append(batch)

    @property
    def pending_depth(self) -> int:
        return sum(len(queue) for queue in self._pending_by_worker.values())

    def _take_batch(
        self, target_worker_id: int | None
    ) -> tuple[AcousticTokenBatch, int] | None:
        if target_worker_id is not None:
            if (
                isinstance(target_worker_id, bool)
                or not isinstance(target_worker_id, int)
                or target_worker_id < 0
                or target_worker_id >= self.worker_count
            ):
                raise CapacitySloLaneError("target_worker_id is out of range")
            target = int(target_worker_id)
            queue = self._pending_by_worker.get(target)
            if not queue:
                return None
            batch = queue.popleft()
            self._pending_ids.discard(batch.request_id)
            return batch, target
        for offset in range(self.worker_count):
            worker_id = (self._worker_cursor + offset) % self.worker_count
            queue = self._pending_by_worker.get(worker_id)
            if queue:
                self._worker_cursor = (worker_id + 1) % self.worker_count
                batch = queue.popleft()
                self._pending_ids.discard(batch.request_id)
                return batch, worker_id
        return None

    def _choose_target(self, request: _LogicalRequest, target_worker_id: int | None) -> int:
        if target_worker_id is not None:
            return super()._choose_target(request, target_worker_id)
        return int(request.assigned_worker_id or 0)


class ElasticAcousticLaneV2(_CapacityLaneBase):
    """Elastic lane that hands off logical state at chunk-safe boundaries."""

    def _needs_handoff(self, request: _LogicalRequest, target: int) -> bool:
        return request.last_worker_id is not None and int(request.last_worker_id) != int(target)


__all__ = [
    "CapacityLaneProgress",
    "CapacitySloLaneError",
    "ElasticAcousticLaneV2",
    "PhysicalAffinityAcousticLaneV2",
]
