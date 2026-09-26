"""Real local Token2Wav Flow-step batching lane for APR evaluation.

The lane is deliberately separate from the frozen original-affinity lane. It
uses only Token2Wav's public Flow-step and PCM-finalization APIs and keeps the
caller-owned stream state keyed by logical request ID.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Mapping

from lychee_fd.runtime.apr.contracts import AcousticPcmRecord, AcousticTokenBatch
from lychee_fd.runtime.apr.online_step_coordinator import PreparedFlowChunk
from lychee_fd.runtime.apr.flow_batch_runtime import FlowBatchRuntime, FlowStepItem
from tools.apr.real_acoustic_lanes import (
    _acoustic_cuda_device_context,
    resolve_lane_identities,
)
from tools.apr.real_acoustic_lanes import AcousticLaneProgress


class FlowBatchAcousticLaneError(RuntimeError):
    """Raised when a Flow-batched acoustic transition cannot be committed."""


@contextmanager
def _flow_context(model: Any):
    try:
        import torch
    except Exception:
        yield
        return
    if not torch.cuda.is_available():
        yield
        return
    dtype = torch.float16 if bool(getattr(model, "float16", False)) else torch.float32
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        yield


@dataclass
class _Request:
    request_id: str
    stream_id: str
    generation_id: int
    prompt_wav: str
    stream_state: dict[str, Any]
    next_sequence: int = 0
    next_version: int = 0
    pcm_seq: int = 0
    cancelled: bool = False


class Token2WavFlowStepBackend:
    """Public-method-only bridge consumed by FlowBatchRuntime."""

    def __init__(self, model: Any) -> None:
        required = (
            "advance_chunk_step",
            "advance_chunk_step_batch",
            "finish_chunk_steps",
        )
        missing = [name for name in required if not callable(getattr(model, name, None))]
        if missing:
            raise FlowBatchAcousticLaneError(
                f"Token2Wav model lacks public Flow methods: {', '.join(missing)}"
            )
        self._model = model

    def advance_step(self, state: Any) -> Any:
        with _flow_context(self._model):
            return self._model.advance_chunk_step(state)

    def advance_step_batch(self, states: tuple[Any, ...]) -> tuple[Any, ...]:
        with _flow_context(self._model):
            return tuple(self._model.advance_chunk_step_batch(states))

    def finish(self, state: Any) -> Any:
        with _flow_context(self._model):
            return self._model.finish_chunk_steps(state)


class VariableLengthToken2WavFlowStepBackend(Token2WavFlowStepBackend):
    """Public backend for the opt-in unequal-attention-cache path.

    The single-state operation deliberately remains the frozen public method;
    only a physical multi-state call is routed to the new variable-length API.
    """

    variable_length = True

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        if not callable(getattr(model, "advance_chunk_step_variable_batch", None)):
            raise FlowBatchAcousticLaneError(
                "Token2Wav model lacks public advance_chunk_step_variable_batch()"
            )

    def advance_step_batch(self, states: tuple[Any, ...]) -> tuple[Any, ...]:
        with _flow_context(self._model):
            return tuple(self._model.advance_chunk_step_variable_batch(states))


class APRFlowBatchAcousticLane:
    """Execute compatible logical Flow states in bounded dynamic batches."""

    def __init__(
        self,
        *,
        worker_count: int,
        model: Any,
        prompt_wav: str,
        max_batch_size: int = 2,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
            raise FlowBatchAcousticLaneError("worker_count must be a positive integer")
        if not callable(getattr(model, "create_stream_state", None)):
            raise FlowBatchAcousticLaneError("model must expose create_stream_state()")
        if not callable(getattr(model, "begin_chunk_steps", None)):
            raise FlowBatchAcousticLaneError("model must expose begin_chunk_steps()")
        if not callable(getattr(model, "render_chunk_pcm", None)):
            raise FlowBatchAcousticLaneError("model must expose render_chunk_pcm()")
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise FlowBatchAcousticLaneError("max_batch_size must be an integer")
        if max_batch_size not in (1, 2, 4):
            raise FlowBatchAcousticLaneError("max_batch_size must be one of 1, 2, or 4")

        self.worker_count = worker_count
        self.model = model
        self.prompt_wav = str(prompt_wav)
        self.max_batch_size = max_batch_size
        self.events: list[dict[str, Any]] = []
        self.ownership_errors = 0
        self.cleanup_ok = False
        self.backends: dict[str, _Request] = {}
        self.worker_for: dict[str, int] = {}
        self.last_worker: dict[str, int] = {}
        self.queue: deque[AcousticTokenBatch] = deque()
        self._progress: deque[AcousticLaneProgress] = deque()
        self._worker_cursor = 0
        sink = event_sink or self.events.append
        self.flow_runtime = FlowBatchRuntime(
            backend=Token2WavFlowStepBackend(model),
            max_batch_size=max_batch_size,
            max_batch_wait_ms=0.0,
            event_sink=sink,
            timing_enabled=True,
        )

    def _request(self, request_id: str) -> _Request:
        try:
            return self.backends[request_id]
        except KeyError as exc:
            raise FlowBatchAcousticLaneError(
                f"request is not admitted: {request_id}"
            ) from exc

    def start(
        self,
        request_ids: Iterable[str],
        *,
        identity_by_request: Mapping[str, tuple[str, int]] | None = None,
    ) -> tuple[str, ...]:
        ids = tuple(str(request_id) for request_id in request_ids)
        if not ids or len(set(ids)) != len(ids) or any(not request_id for request_id in ids):
            raise FlowBatchAcousticLaneError("request IDs must be unique and non-empty")
        identities = resolve_lane_identities(ids, identity_by_request)
        with _acoustic_cuda_device_context():
            for index, request_id in enumerate(ids):
                if request_id in self.backends:
                    raise FlowBatchAcousticLaneError(f"request already started: {request_id}")
                stream_id, generation_id = identities[request_id]
                stream_state = self.model.create_stream_state(self.prompt_wav)
                if not isinstance(stream_state, dict):
                    raise FlowBatchAcousticLaneError("create_stream_state must return a dict")
                self.backends[request_id] = _Request(
                    request_id=request_id,
                    stream_id=stream_id,
                    generation_id=generation_id,
                    prompt_wav=self.prompt_wav,
                    stream_state=stream_state,
                )
                self.worker_for[request_id] = index % self.worker_count
                self.events.append(
                    {
                        "event": "ACOUSTIC_REQUEST_START",
                        "event_type": "ACOUSTIC_REQUEST_START",
                        "request_id": request_id,
                        "stream_id": stream_id,
                        "generation_id": generation_id,
                        "worker_id": self.worker_for[request_id],
                    }
                )
        return ids

    def submit(self, batch: AcousticTokenBatch) -> None:
        if not isinstance(batch, AcousticTokenBatch):
            raise FlowBatchAcousticLaneError("lane accepts AcousticTokenBatch values only")
        request = self._request(batch.request_id)
        if request.cancelled:
            raise FlowBatchAcousticLaneError("cancelled request cannot accept tokens")
        if batch.stream_id != request.stream_id or batch.generation_id != request.generation_id:
            raise FlowBatchAcousticLaneError("acoustic batch identity mismatch")
        if batch.sequence_no != request.next_sequence:
            raise FlowBatchAcousticLaneError(
                f"sequence mismatch for {batch.request_id}: "
                f"expected={request.next_sequence} got={batch.sequence_no}"
            )
        if batch.state_version != request.next_version:
            raise FlowBatchAcousticLaneError(
                f"state version mismatch for {batch.request_id}: "
                f"expected={request.next_version} got={batch.state_version}"
            )
        self.queue.append(batch)
        self.events.append(
            {
                "event": "ACOUSTIC_TOKEN_SUBMIT",
                "event_type": "ACOUSTIC_TOKEN_SUBMIT",
                "request_id": batch.request_id,
                "sequence_no": batch.sequence_no,
                "state_version": batch.state_version,
            }
        )

    def _take_group(self) -> tuple[AcousticTokenBatch, ...]:
        first = self.queue.popleft()
        group = [first]
        remaining: deque[AcousticTokenBatch] = deque()
        for candidate in self.queue:
            compatible = (
                len(candidate.stoken_ids) == len(first.stoken_ids)
                and candidate.generation_id == first.generation_id
                and candidate.last_chunk == first.last_chunk
            )
            if compatible and len(group) < self.max_batch_size:
                group.append(candidate)
            else:
                remaining.append(candidate)
        self.queue = remaining
        return tuple(group)

    @staticmethod
    def _state_shape(
        state: Any,
        fallback_tokens: tuple[int, ...],
        *,
        wildcard_attention_time: bool = False,
    ) -> tuple[int, ...]:
        """Return the exact public tensor-shape signature needed for batching.

        ``x`` alone is insufficient: two logical chunks can have the same
        acoustic length while their causal attention histories have different
        time extents.  ``CausalConditionalCFM._advance_states`` concatenates
        those histories along only the logical-batch axis, so every other
        dimension must match before a physical batch is formed.  Read only
        public state fields and the current per-step cache slot; worker-local
        buffers and cache capacity are deliberately excluded.
        """

        signature: list[int] = []

        def add_shape(tag: int, value: Any, *, wildcard_dims: tuple[int, ...] = ()) -> None:
            shape = getattr(value, "shape", None)
            if shape is None:
                signature.extend((tag, -1))
                return
            dimensions = [int(item) for item in shape]
            for dimension in wildcard_dims:
                if 0 <= dimension < len(dimensions):
                    dimensions[dimension] = -1
            signature.extend((tag, len(dimensions), *dimensions))

        # These fields are concatenated by the public Flow batch contract.
        for tag, name in enumerate(
            ("x", "t", "dt", "t_span", "mu", "speaker", "condition"),
            start=1,
        ):
            add_shape(tag, getattr(state, name, None))

        step_index = int(getattr(state, "step_index", 0))
        for tag, name in ((101, "input_cnn_cache"), (102, "input_att_cache")):
            cache = getattr(state, name, None)
            if isinstance(cache, (tuple, list)):
                cache = cache[step_index] if step_index < len(cache) else None
            wildcard_dims = (3,) if wildcard_attention_time and name == "input_att_cache" else ()
            add_shape(tag, cache, wildcard_dims=wildcard_dims)

        if not signature:
            return (1, len(fallback_tokens))
        return tuple(signature)

    @staticmethod
    def _state_device(state: Any) -> str:
        value = getattr(state, "x", None)
        device = getattr(value, "device", None)
        return str(device) if device is not None else "cpu"

    @staticmethod
    def _state_dtype(state: Any) -> str:
        value = getattr(state, "x", None)
        dtype = getattr(value, "dtype", None)
        return str(dtype) if dtype is not None else "unknown"

    def _begin_state(self, batch: AcousticTokenBatch, request: _Request) -> Any:
        with _flow_context(self.model):
            return self.model.begin_chunk_steps(
                batch.stoken_ids,
                request.prompt_wav,
                request.stream_state,
                last_chunk=batch.last_chunk,
                n_timesteps=10,
                request_id=batch.request_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                version=batch.state_version,
            )

    def _run_flow_group(self, batches: tuple[AcousticTokenBatch, ...]) -> None:
        requests = {batch.request_id: self._request(batch.request_id) for batch in batches}
        states = {
            batch.request_id: self._begin_state(batch, requests[batch.request_id])
            for batch in batches
        }
        first_state = next(iter(states.values()))
        t_span = getattr(first_state, "t_span", None)
        step_count = (
            int(t_span.numel() - 1)
            if t_span is not None
            else int(getattr(first_state, "n_steps", 10))
        )
        for step_index in range(step_count):
            for batch in batches:
                state = states[batch.request_id]
                item = FlowStepItem(
                    request_id=batch.request_id,
                    generation_id=batch.generation_id,
                    version=batch.state_version,
                    state=state,
                    ready_at_ns=time.monotonic_ns(),
                    model_identity=type(self.model).__name__,
                    device=self._state_device(state),
                    dtype=self._state_dtype(state),
                    step_index=int(getattr(state, "step_index", step_index)),
                    shape_signature=self._state_shape(state, batch.stoken_ids),
                    last_chunk=batch.last_chunk,
                    n_timesteps=step_count,
                )
                self.flow_runtime.submit(item)
            result = self.flow_runtime.run_next(now_ns=time.monotonic_ns())
            if result is None or not result.committed:
                raise FlowBatchAcousticLaneError("Flow batch runtime made no progress")
            states.update({state.request_id: state for state in result.states})

        worker_id = self._worker_cursor % self.worker_count
        self._worker_cursor += 1
        for batch in batches:
            request = requests[batch.request_id]
            with _flow_context(self.model):
                mel, estimator_cache = self.flow_runtime.finish(batch.request_id)
            if not isinstance(estimator_cache, dict):
                raise FlowBatchAcousticLaneError("Flow finish must return a cache mapping")
            next_flow_cache = dict(request.stream_state["flow_cache"])
            next_flow_cache.update(estimator_cache)
            request.stream_state["flow_cache"] = next_flow_cache
            setter = getattr(self.model, "set_profile_context", None)
            if callable(setter):
                setter(
                    session_id=batch.request_id,
                    generation_id=batch.generation_id,
                    sequence_no=batch.sequence_no,
                    state_version=batch.state_version,
                    worker_id=worker_id,
                )
            pcm_bytes = self.model.render_chunk_pcm(
                mel,
                request.stream_state,
                last_chunk=batch.last_chunk,
            )
            if not isinstance(pcm_bytes, (bytes, bytearray, memoryview)):
                raise FlowBatchAcousticLaneError(
                    f"Token2Wav returned an invalid PCM value for {batch.request_id}"
                )
            if batch.last_chunk and not pcm_bytes:
                raise FlowBatchAcousticLaneError(
                    f"Token2Wav returned empty final PCM for {batch.request_id}"
                )
            pcm_records: tuple[AcousticPcmRecord, ...]
            if pcm_bytes:
                record = AcousticPcmRecord(
                    request_id=batch.request_id,
                    stream_id=batch.stream_id,
                    generation_id=batch.generation_id,
                    sequence_no=batch.sequence_no,
                    pcm_bytes=bytes(pcm_bytes),
                    sample_rate=24000,
                    pcm_seq=request.pcm_seq,
                )
                request.pcm_seq += 1
                pcm_records = (record,)
            else:
                pcm_records = ()
            request.next_sequence += 1
            request.next_version += 1
            self.flow_runtime.update_version(
                batch.request_id,
                generation_id=batch.generation_id,
                version=request.next_version,
            )
            previous = self.last_worker.get(batch.request_id)
            self.last_worker[batch.request_id] = worker_id
            self._progress.append(
                AcousticLaneProgress(
                    request_id=batch.request_id,
                    worker_id=worker_id,
                    pcm_records=pcm_records,
                    state_version=request.next_version,
                    checkpoint_count=1,
                    restore_count=1,
                    worker_switch=previous is not None and previous != worker_id,
                )
            )

    def process_one(self) -> AcousticLaneProgress | None:
        if self._progress:
            return self._progress.popleft()
        if not self.queue:
            return None
        batches = self._take_group()
        self._run_flow_group(batches)
        return self._progress.popleft() if self._progress else None

    def cancel(self, request_id: str) -> None:
        request_id = str(request_id)
        request = self.backends.pop(request_id, None)
        if request is not None:
            request.cancelled = True
        self.queue = deque(batch for batch in self.queue if batch.request_id != request_id)
        self._progress = deque(
            progress for progress in self._progress if progress.request_id != request_id
        )
        self.flow_runtime.reset_request(request_id)
        self.worker_for.pop(request_id, None)
        self.events.append(
            {
                "event": "ACOUSTIC_CANCEL",
                "event_type": "ACOUSTIC_CANCEL",
                "request_id": request_id,
            }
        )

    def online_step_adapter(self) -> "Token2WavFlowChunkExecutionAdapter":
        """Return the public adapter used by the online step coordinator."""
        return Token2WavFlowChunkExecutionAdapter(self)

    def variable_length_online_step_adapter(
        self,
    ) -> "VariableLengthToken2WavFlowChunkExecutionAdapter":
        """Return the opt-in adapter that ignores only attention time length."""
        return VariableLengthToken2WavFlowChunkExecutionAdapter(self)

    def close(self) -> None:
        for request_id in tuple(self.backends):
            self.cancel(request_id)
        self.queue.clear()
        self._progress.clear()
        self.cleanup_ok = not self.backends and not self.queue and not self._progress


class Token2WavFlowChunkExecutionAdapter:
    """Expose one logical Flow chunk through the public Token2Wav API."""

    def __init__(self, lane: APRFlowBatchAcousticLane) -> None:
        self.lane = lane
        self.model = lane.model
        self.backend = Token2WavFlowStepBackend(self.model)

    def register(self, request_id: str, *, stream_id: str, generation_id: int) -> None:
        request = self.lane._request(request_id)
        if request.stream_id != str(stream_id) or request.generation_id != int(generation_id):
            raise FlowBatchAcousticLaneError(
                f"online adapter identity mismatch for {request_id}"
            )

    def prepare_chunk(self, batch: AcousticTokenBatch) -> PreparedFlowChunk:
        request = self.lane._request(batch.request_id)
        state = self.lane._begin_state(batch, request)
        t_span = getattr(state, "t_span", None)
        n_timesteps = (
            int(t_span.numel() - 1)
            if t_span is not None
            else int(getattr(state, "n_steps", 10))
        )
        if n_timesteps <= 0:
            raise FlowBatchAcousticLaneError("Flow state has no Euler steps")
        worker_id = self.lane._worker_cursor % self.lane.worker_count
        self.lane._worker_cursor += 1
        return PreparedFlowChunk(
            batch=batch,
            state=state,
            n_timesteps=n_timesteps,
            worker_id=worker_id,
        )

    def make_step_item(self, chunk: PreparedFlowChunk) -> FlowStepItem:
        state = chunk.state
        batch = chunk.batch
        return FlowStepItem(
            request_id=batch.request_id,
            generation_id=batch.generation_id,
            version=batch.state_version,
            state=state,
            ready_at_ns=time.monotonic_ns(),
            model_identity=type(self.model).__name__,
            device=self.lane._state_device(state),
            dtype=self.lane._state_dtype(state),
            step_index=int(getattr(state, "step_index", 0)),
            shape_signature=self.lane._state_shape(state, batch.stoken_ids),
            last_chunk=batch.last_chunk,
            n_timesteps=chunk.n_timesteps,
        )

    def advance_step(self, state: Any) -> Any:
        return self.backend.advance_step(state)

    def advance_step_batch(self, states: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.backend.advance_step_batch(states)

    def update_step(self, chunk: PreparedFlowChunk, next_state: Any) -> None:
        chunk.state = next_state

    def schedule_terminal_step(self, state: Any) -> Any:
        return self.model.schedule_terminal_step(state)

    def wait_for_step(self, completion_event: Any) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.current_stream().wait_event(completion_event)

    def _finish_flow(self, state: Any) -> tuple[Any, dict[str, Any]]:
        finished = self.model.finish_chunk_steps(state)
        if not isinstance(finished, tuple):
            raise FlowBatchAcousticLaneError("Flow finish must return a tuple")
        if len(finished) == 3:
            mel, cnn_cache, att_cache = finished
            return mel, {
                "estimator_cnn_cache": cnn_cache,
                "estimator_att_cache": att_cache,
            }
        if len(finished) == 2 and isinstance(finished[1], Mapping):
            return finished[0], dict(finished[1])
        raise FlowBatchAcousticLaneError(
            "Flow finish must return (mel, cache) or (mel, cnn_cache, att_cache)"
        )

    def finalize_chunk(self, chunk: PreparedFlowChunk) -> AcousticLaneProgress:
        batch = chunk.batch
        request = self.lane._request(batch.request_id)
        with _flow_context(self.model):
            mel, estimator_cache = self._finish_flow(chunk.state)
        if not isinstance(estimator_cache, dict):
            raise FlowBatchAcousticLaneError("Flow finish must return a cache mapping")
        current_cache = request.stream_state.get("flow_cache")
        if not isinstance(current_cache, dict):
            raise FlowBatchAcousticLaneError("stream state is missing flow_cache")
        next_flow_cache = dict(current_cache)
        next_flow_cache.update(estimator_cache)
        request.stream_state["flow_cache"] = next_flow_cache

        setter = getattr(self.model, "set_profile_context", None)
        if callable(setter):
            setter(
                session_id=batch.request_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                state_version=batch.state_version,
                worker_id=chunk.worker_id,
            )
        with _flow_context(self.model):
            pcm_bytes = self.model.render_chunk_pcm(
                mel,
                request.stream_state,
                last_chunk=batch.last_chunk,
            )
        if not isinstance(pcm_bytes, (bytes, bytearray, memoryview)):
            raise FlowBatchAcousticLaneError(
                f"Token2Wav returned an invalid PCM value for {batch.request_id}"
            )
        if batch.last_chunk and not pcm_bytes:
            raise FlowBatchAcousticLaneError(
                f"Token2Wav returned empty final PCM for {batch.request_id}"
            )

        if pcm_bytes:
            record = AcousticPcmRecord(
                request_id=batch.request_id,
                stream_id=batch.stream_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                pcm_bytes=bytes(pcm_bytes),
                sample_rate=24000,
                pcm_seq=request.pcm_seq,
            )
            request.pcm_seq += 1
            records = (record,)
        else:
            records = ()
        try:
            for record in records:
                if (
                    record.request_id != batch.request_id
                    or record.stream_id != batch.stream_id
                    or record.generation_id != batch.generation_id
                    or record.sequence_no != batch.sequence_no
                    or not record.pcm_bytes
                ):
                    raise FlowBatchAcousticLaneError(
                        f"PCM ownership mismatch for {batch.request_id}"
                    )
        except FlowBatchAcousticLaneError:
            self.lane.ownership_errors += 1
            raise
        request.next_sequence += 1
        request.next_version += 1
        previous = self.lane.last_worker.get(batch.request_id)
        self.lane.last_worker[batch.request_id] = chunk.worker_id
        return AcousticLaneProgress(
            request_id=batch.request_id,
            worker_id=chunk.worker_id,
            pcm_records=records,
            state_version=request.next_version,
            checkpoint_count=1,
            restore_count=1,
            worker_switch=previous is not None and previous != chunk.worker_id,
        )

    def cancel(self, request_id: str) -> None:
        self.lane.cancel(request_id)


class VariableLengthToken2WavFlowChunkExecutionAdapter(
    Token2WavFlowChunkExecutionAdapter
):
    """Online adapter for narrow variable-length Flow B=2 execution."""

    def __init__(self, lane: APRFlowBatchAcousticLane) -> None:
        super().__init__(lane)
        self.backend = VariableLengthToken2WavFlowStepBackend(self.model)

    def make_step_item(self, chunk: PreparedFlowChunk) -> FlowStepItem:
        state = chunk.state
        batch = chunk.batch
        return FlowStepItem(
            request_id=batch.request_id,
            generation_id=batch.generation_id,
            version=batch.state_version,
            state=state,
            ready_at_ns=time.monotonic_ns(),
            model_identity=type(self.model).__name__,
            device=self.lane._state_device(state),
            dtype=self.lane._state_dtype(state),
            step_index=int(getattr(state, "step_index", 0)),
            shape_signature=self.lane._state_shape(
                state,
                batch.stoken_ids,
                wildcard_attention_time=True,
            ),
            last_chunk=batch.last_chunk,
            n_timesteps=chunk.n_timesteps,
        )


__all__ = [
    "APRFlowBatchAcousticLane",
    "FlowBatchAcousticLaneError",
    "Token2WavFlowStepBackend",
    "Token2WavFlowChunkExecutionAdapter",
    "VariableLengthToken2WavFlowStepBackend",
    "VariableLengthToken2WavFlowChunkExecutionAdapter",
]
