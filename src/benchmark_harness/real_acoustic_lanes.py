"""Local Token2Wav lanes for real-model APR bring-up.

Both lanes consume the same public AcousticBackend contract.  The fixed
lane keeps a logical request on one physical slot; the APR lane delegates
checkpoint capture/restore and worker selection to AprRuntime.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from lychee_fd.runtime.apr.contracts import AcousticPcmRecord, AcousticTokenBatch
from lychee_fd.runtime.apr.local_backend import LocalToken2WavAPRBackend
from lychee_fd.runtime.apr.profiling import StageProfiler
from lychee_fd.runtime.apr.runtime import AprRuntime
from lychee_fd.runtime.apr.worker import AcousticBackend


class RealAcousticLaneError(RuntimeError):
    """Raised when a local acoustic lane cannot preserve ownership."""


@contextmanager
def _acoustic_cuda_device_context():
    """Use the acoustic CUDA device only while creating per-request state."""
    import os

    device_raw = os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE", "").strip()
    if not device_raw:
        yield
        return
    try:
        device_index = int(device_raw)
    except ValueError as exc:
        raise RuntimeError(
            "LYCHEEFD_TOKEN2WAV_DEVICE must be a non-negative CUDA index"
        ) from exc
    if device_index < 0:
        raise RuntimeError(
            "LYCHEEFD_TOKEN2WAV_DEVICE must be a non-negative CUDA index"
        )
    import torch

    previous_device = int(torch.cuda.current_device())
    torch.cuda.set_device(device_index)
    try:
        yield
    finally:
        if previous_device != device_index:
            torch.cuda.set_device(previous_device)


@dataclass(frozen=True)
class AcousticLaneProgress:
    request_id: str
    worker_id: int
    pcm_records: tuple[AcousticPcmRecord, ...]
    state_version: int
    checkpoint_count: int
    restore_count: int
    worker_switch: bool


def _default_backend_factory(
    model: Any,
    prompt_wav: str,
    stream_state_factory: Callable[[str], Mapping[str, Any]] | None,
    request_id: str,
) -> AcousticBackend:
    if stream_state_factory is not None:
        stream_state = stream_state_factory(request_id)
    elif hasattr(model, "create_stream_state"):
        stream_state = model.create_stream_state(prompt_wav)
    else:
        raise RealAcousticLaneError(
            "real Token2Wav model must expose create_stream_state"
        )
    return LocalToken2WavAPRBackend(
        model,
        request_id=request_id,
        stream_id=f"stream-{request_id}",
        generation_id=0,
        prompt_wav=prompt_wav,
        stream_state=stream_state,
    )


class _LocalLaneBase:
    def __init__(
        self,
        *,
        worker_count: int,
        backend_factory: Callable[[str], AcousticBackend] | None = None,
        model: Any | None = None,
        prompt_wav: str | None = None,
        stream_state_factory: Callable[[str], Mapping[str, Any]] | None = None,
        profiler: StageProfiler | None = None,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if isinstance(worker_count, bool) or not isinstance(worker_count, int):
            raise RealAcousticLaneError("worker_count must be an integer")
        if worker_count <= 0:
            raise RealAcousticLaneError("worker_count must be positive")
        if backend_factory is None:
            if model is None or not prompt_wav:
                raise RealAcousticLaneError(
                    "backend_factory or model plus prompt_wav is required"
                )
            backend_factory = lambda request_id: _default_backend_factory(
                model, prompt_wav, stream_state_factory, request_id
            )
        self.worker_count = worker_count
        self._backend_factory = backend_factory
        if profiler is not None and not isinstance(profiler, StageProfiler):
            raise RealAcousticLaneError("profiler must be a StageProfiler")
        self._profiler = profiler
        self.backends: dict[str, AcousticBackend] = {}
        self.worker_for: dict[str, int] = {}
        self.last_worker: dict[str, int] = {}
        self._next_versions: dict[str, int] = {}
        self._next_sequences: dict[str, int] = {}
        self._admission_cursor = 0
        self.queue: deque[AcousticTokenBatch] = deque()
        self.events: list[dict[str, Any]] = []
        self._event_sink = event_sink
        self.ownership_errors = 0
        self.cleanup_ok = False

    def _emit_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)

    def _profile_span(
        self,
        stage: str,
        *,
        session_id: str | None = None,
        generation_id: int | None = None,
        sequence_no: int | None = None,
        state_version: int | None = None,
        worker_id: int | None = None,
        queue_depth: int | None = None,
    ):
        if self._profiler is None:
            return nullcontext()
        return self._profiler.span(
            stage,
            session_id=session_id,
            generation_id=generation_id,
            sequence_no=sequence_no,
            state_version=state_version,
            worker_id=worker_id,
            queue_depth=queue_depth,
        )

    def start(self, request_ids: Iterable[str]) -> tuple[str, ...]:
        ids = tuple(str(request_id) for request_id in request_ids)
        if not ids or len(set(ids)) != len(ids) or any(not request_id for request_id in ids):
            raise RealAcousticLaneError("request IDs must be unique and non-empty")
        with _acoustic_cuda_device_context():
            for index, request_id in enumerate(ids):
                if request_id in self.backends:
                    raise RealAcousticLaneError(f"request already started: {request_id}")
                with self._profile_span(
                    "REQUEST_ADMISSION",
                    session_id=request_id,
                    generation_id=0,
                    sequence_no=0,
                    state_version=0,
                ):
                    backend = self._backend_factory(request_id)
                    if not isinstance(backend, AcousticBackend):
                        raise RealAcousticLaneError(
                            "backend_factory must return an AcousticBackend"
                        )
                    self.backends[request_id] = backend
                    self.worker_for[request_id] = self._admission_cursor % self.worker_count
                    self._admission_cursor += 1
                    self._next_versions[request_id] = 0
                    self._next_sequences[request_id] = 0
                self._emit_event(
                    {
                        "event": "ACOUSTIC_REQUEST_START",
                        "request_id": request_id,
                        "worker_id": self.worker_for[request_id],
                    }
                )
        return ids

    def submit(self, batch: AcousticTokenBatch) -> None:
        if not isinstance(batch, AcousticTokenBatch):
            raise RealAcousticLaneError("lane accepts AcousticTokenBatch values only")
        if batch.request_id not in self.backends:
            raise RealAcousticLaneError(
                f"request is not admitted: {batch.request_id}"
            )
        expected_version = self._next_versions[batch.request_id]
        expected_sequence = self._next_sequences[batch.request_id]
        if batch.state_version != expected_version:
            raise RealAcousticLaneError(
                f"state version mismatch for {batch.request_id}: "
                f"expected={expected_version} got={batch.state_version}"
            )
        if batch.sequence_no != expected_sequence:
            raise RealAcousticLaneError(
                f"sequence mismatch for {batch.request_id}: "
                f"expected={expected_sequence} got={batch.sequence_no}"
            )
        self.queue.append(batch)
        self._emit_event(
            {
                "event": "ACOUSTIC_TOKEN_SUBMIT",
                "request_id": batch.request_id,
                "sequence_no": batch.sequence_no,
                "state_version": batch.state_version,
            }
        )

    @staticmethod
    def _validate_pcm(
        batch: AcousticTokenBatch,
        records: tuple[AcousticPcmRecord, ...],
    ) -> None:
        for record in records:
            if (
                record.request_id != batch.request_id
                or record.stream_id != batch.stream_id
                or record.generation_id != batch.generation_id
                or record.sequence_no != batch.sequence_no
                or not record.pcm_bytes
            ):
                raise RealAcousticLaneError(
                    f"PCM ownership mismatch for {batch.request_id}"
                )

    def cancel(self, request_id: str) -> None:
        request_id = str(request_id)
        backend = self.backends.pop(request_id, None)
        if backend is not None:
            backend.cancel()
        self.queue = deque(
            batch for batch in self.queue if batch.request_id != request_id
        )
        self.worker_for.pop(request_id, None)
        self._next_versions.pop(request_id, None)
        self._next_sequences.pop(request_id, None)
        self._emit_event({"event": "ACOUSTIC_CANCEL", "request_id": request_id})

    def close(self) -> None:
        for request_id in tuple(self.backends):
            with self._profile_span("CLEANUP", session_id=request_id):
                self.cancel(request_id)
        self.queue.clear()
        self.cleanup_ok = not self.backends and not self.queue


class FixedAffinityAcousticLane(_LocalLaneBase):
    """Control lane with one permanent worker assignment per logical request."""

    def process_one(self) -> AcousticLaneProgress | None:
        if not self.queue:
            return None
        batch = self.queue.popleft()
        backend = self.backends.get(batch.request_id)
        if backend is None:
            raise RealAcousticLaneError(
                f"request disappeared before acoustic progress: {batch.request_id}"
            )
        worker_id = self.worker_for[batch.request_id]
        backend.resume()
        backend.process(batch.stoken_ids, last_chunk=batch.last_chunk)
        records = tuple(backend.commit_pcm())
        try:
            with self._profile_span(
                "PCM_EGRESS",
                session_id=batch.request_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                state_version=batch.state_version,
                worker_id=worker_id,
            ):
                self._validate_pcm(batch, records)
        except RealAcousticLaneError:
            self.ownership_errors += 1
            raise
        self._next_versions[batch.request_id] += 1
        self._next_sequences[batch.request_id] += 1
        previous = self.last_worker.get(batch.request_id)
        self.last_worker[batch.request_id] = worker_id
        self._emit_event(
            {
                "event": "ACOUSTIC_PROGRESS",
                "request_id": batch.request_id,
                "worker_id": worker_id,
                "pcm_count": len(records),
            }
        )
        return AcousticLaneProgress(
            request_id=batch.request_id,
            worker_id=worker_id,
            pcm_records=records,
            state_version=self._next_versions[batch.request_id],
            checkpoint_count=0,
            restore_count=0,
            worker_switch=previous is not None and previous != worker_id,
        )


class APRLocalAcousticLane(_LocalLaneBase):
    """APR lane using the explicit checkpoint contract and local backend."""

    def __init__(self, *, worker_count: int, **kwargs: Any) -> None:
        super().__init__(worker_count=worker_count, **kwargs)
        self.runtime = AprRuntime(
            enabled=True,
            worker_count=worker_count,
            event_sink=self.events.append,
            profiler=self._profiler,
        )

    def start(self, request_ids: Iterable[str]) -> tuple[str, ...]:
        ids = super().start(request_ids)
        try:
            for request_id in ids:
                self.runtime.start_request(
                    request_id,
                    self.backends[request_id],
                    generation_id=0,
                )
        except Exception:
            for request_id in ids:
                self.backends.pop(request_id, None)
            raise
        return ids

    def process_one(self) -> AcousticLaneProgress | None:
        if not self.backends:
            return None
        before = len(self.events)
        result = self.runtime.process_one()
        if result is None:
            return None
        submit_events = [
            event for event in self.events[:before]
            if event.get("event") == "ACOUSTIC_TOKEN_SUBMIT"
            and event.get("request_id") == result.request_id
        ]
        if not submit_events:
            raise RealAcousticLaneError(
                f"no public batch identity for APR result {result.request_id}"
            )
        submit_event = submit_events[-1]
        batch = AcousticTokenBatch(
            request_id=result.request_id,
            stream_id=f"stream-{result.request_id}",
            generation_id=0,
            sequence_no=int(submit_event["sequence_no"]),
            stoken_ids=(0,),
            source_execution_id="apr-public-result",
            state_version=int(submit_event["state_version"]),
            created_monotonic_ns=0,
        )
        records = tuple(result.pcm_records)
        worker_id = int(result.worker_id)
        try:
            with self._profile_span(
                "PCM_EGRESS",
                session_id=batch.request_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                state_version=batch.state_version,
                worker_id=worker_id,
            ):
                self._validate_pcm(batch, records)
        except RealAcousticLaneError:
            self.ownership_errors += 1
            raise
        self._next_versions[batch.request_id] = int(result.state_version)
        self._next_sequences[batch.request_id] = batch.sequence_no + 1
        previous = self.last_worker.get(batch.request_id)
        self.last_worker[batch.request_id] = worker_id
        event_slice = self.events[before:]
        checkpoint_count = sum(
            event.get("event_type") == "APR_STATE_COMMIT"
            for event in event_slice
        )
        restore_count = sum(
            event.get("event_type") == "APR_STATE_RESTORE"
            for event in event_slice
        )
        return AcousticLaneProgress(
            request_id=result.request_id,
            worker_id=worker_id,
            pcm_records=records,
            state_version=int(result.state_version),
            checkpoint_count=checkpoint_count,
            restore_count=restore_count,
            worker_switch=previous is not None and previous != worker_id,
        )

    def submit(self, batch: AcousticTokenBatch) -> None:
        super().submit(batch)
        self.queue.pop()
        self.runtime.enqueue_token_batch(batch)

    def cancel(self, request_id: str) -> None:
        request_id = str(request_id)
        if request_id in self.backends:
            self.runtime.shutdown_request(request_id, generation_id=0)
        super().cancel(request_id)

    def close(self) -> None:
        with self._profile_span("CLEANUP"):
            self.runtime.close()
            self.backends.clear()
            self.queue.clear()
            self.cleanup_ok = True
