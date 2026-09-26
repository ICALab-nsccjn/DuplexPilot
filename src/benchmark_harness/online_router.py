"""Benchmark-only online acoustic routing for the B0--B3 common harness.

The router is the narrow bridge between the realtime Token2Wav token stream and
the public APR acoustic lanes.  It owns no model state: the lane remains the
only component allowed to create, advance, and finalize acoustic state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import inspect
import os
import threading
import time
from typing import Any, Callable, Mapping

from .contracts import AcousticPcmRecord, AcousticTokenBatch
from .paper_systems import PaperSystemSpec, get_system_spec


class OnlineAcousticRouterError(RuntimeError):
    """Raised when an online acoustic request violates the public contract."""


def resolve_paper_system_payload(
    payload: Mapping[str, Any], *, runtime_mode: str
) -> PaperSystemSpec | None:
    """Resolve benchmark-only fields and fail closed on a mismatched spec."""
    if not isinstance(payload, Mapping):
        raise ValueError("paper system payload must be a mapping")
    system_id = payload.get("paper_system_id")
    if system_id in (None, ""):
        return None
    spec = get_system_spec(str(system_id))
    supplied = {
        "runtime_mode": runtime_mode,
        "acoustic_mode": payload.get("acoustic_mode"),
        "max_flow_batch_size": payload.get("max_flow_batch_size"),
    }
    expected = {
        "runtime_mode": spec.model_runtime_mode,
        "acoustic_mode": spec.acoustic_mode,
        "max_flow_batch_size": spec.max_flow_batch_size,
    }
    for name, value in supplied.items():
        if value is not None and str(value) != str(expected[name]):
            raise ValueError(
                f"paper system field {name}={value!r} does not match "
                f"{spec.system_id} ({expected[name]!r})"
            )
    return spec


@dataclass
class _RequestState:
    stream_id: str
    generation_id: int
    next_sequence: int = 0
    next_version: int = 0
    in_flight: bool = False


@dataclass
class _Pending:
    batch: AcousticTokenBatch
    submitted_ns: int
    done: threading.Event
    result: dict[str, Any] | None = None
    error: BaseException | None = None


def _configure_acoustic_cuda_device() -> None:
    device_raw = os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE", "").strip()
    if not device_raw:
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
    torch.cuda.set_device(device_index)


class OnlineAcousticRouter:
    """Share one public acoustic lane across realtime logical sessions."""

    def __init__(
        self,
        spec: PaperSystemSpec,
        *,
        model: Any,
        prompt_wav: str,
        worker_count: int,
        max_batch_wait_ms: float | None = None,
        lane_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(spec, PaperSystemSpec):
            raise TypeError("spec must be a PaperSystemSpec")
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
            raise ValueError("worker_count must be a positive integer")
        if max_batch_wait_ms is None:
            max_batch_wait_ms = 0.0 if spec.max_flow_batch_size == 1 else 2.0
        if isinstance(max_batch_wait_ms, bool) or float(max_batch_wait_ms) < 0:
            raise ValueError("max_batch_wait_ms must be non-negative")
        self.spec = spec
        self.worker_count = int(worker_count)
        self.max_batch_wait_ms = float(max_batch_wait_ms)
        self.events: list[dict[str, Any]] = []
        self._condition = threading.Condition()
        self._lane_lock = threading.Lock()
        self._requests: dict[str, _RequestState] = {}
        self._pending: deque[_Pending] = deque()
        self._stopping = False

        if lane_factory is None:
            from tools.benchmarks.paper_backend_selector import build_acoustic_lane

            self.lane = build_acoustic_lane(
                spec,
                worker_count=self.worker_count,
                model=model,
                prompt_wav=prompt_wav,
                event_sink=self._record_lane_event,
            )
        else:
            self.lane = lane_factory(
                spec=spec,
                worker_count=self.worker_count,
                model=model,
                prompt_wav=prompt_wav,
                event_sink=self._record_lane_event,
            )
        self._coordinator = None
        online_adapter_factory = getattr(self.lane, "online_step_adapter", None)
        if (
            spec.acoustic_mode == "apr_step"
            and spec.flow_batch_variant == "variable_length"
        ):
            # Variable-length execution is an explicit opt-in system variant;
            # silently falling back to the exact-shape adapter would invalidate
            # the comparison, so fail closed when the lane lacks the boundary.
            online_adapter_factory = getattr(
                self.lane, "variable_length_online_step_adapter", None
            )
            if not callable(online_adapter_factory):
                raise OnlineAcousticRouterError(
                    "variable-length system requires the public "
                    "variable_length_online_step_adapter() boundary"
                )
        if spec.acoustic_mode == "apr_step" and callable(online_adapter_factory):
            adapter = online_adapter_factory()
            if spec.flow_execution_mode == "async":
                from .async_online_flow_coordinator import AsyncOnlineFlowStepCoordinator

                adaptive_policy = None
                if spec.flow_step_policy == "residual_adaptive":
                    threshold_raw = os.environ.get(
                        "DUPLEXPILOT_ADAPTIVE_RESIDUAL_THRESHOLD", ""
                    ).strip()
                    if not threshold_raw:
                        raise OnlineAcousticRouterError(
                            "residual-adaptive Flow requires "
                            "DUPLEXPILOT_ADAPTIVE_RESIDUAL_THRESHOLD"
                        )
                    from .adaptive_flow_policy import ResidualAdaptiveFlowPolicy

                    adaptive_policy = ResidualAdaptiveFlowPolicy(
                        threshold=float(threshold_raw)
                    )
                self._coordinator = AsyncOnlineFlowStepCoordinator(
                    adapter=adapter,
                    max_inflight=spec.max_inflight_flow_steps,
                    schedule_policy=spec.flow_schedule_policy,
                    step_policy=spec.flow_step_policy,
                    adaptive_policy=adaptive_policy,
                    event_sink=self._record_lane_event,
                )
            else:
                from .online_step_coordinator import OnlineFlowStepCoordinator

                self._coordinator = OnlineFlowStepCoordinator(
                    adapter=adapter,
                    max_batch_size=spec.max_flow_batch_size,
                    max_batch_wait_ms=self.max_batch_wait_ms,
                    event_sink=self._record_lane_event,
                )
            self._thread = None
        else:
            self._thread = threading.Thread(
                target=self._run,
                name=f"paper-acoustic-router-{spec.system_id}",
                daemon=True,
            )
            self._thread.start()

    def _record_lane_event(self, event: dict[str, Any]) -> None:
        if isinstance(event, dict):
            self.events.append(dict(event))

    def register(self, request_id: str, *, stream_id: str, generation_id: int) -> None:
        request_id = str(request_id)
        stream_id = str(stream_id)
        if not request_id or not stream_id or int(generation_id) < 0:
            raise ValueError("request identity is invalid")
        with self._condition:
            if self._stopping:
                raise OnlineAcousticRouterError("router is closed")
            if request_id in self._requests:
                raise OnlineAcousticRouterError(f"request already registered: {request_id}")
            with self._lane_lock:
                start = self.lane.start
                parameters = inspect.signature(start).parameters
                accepts_identity = (
                    "identity_by_request" in parameters
                    or any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                )
                if accepts_identity:
                    start(
                        (request_id,),
                        identity_by_request={
                            request_id: (stream_id, int(generation_id))
                        },
                    )
                else:
                    start((request_id,))
                if self._coordinator is not None:
                    self._coordinator.register(
                        request_id,
                        stream_id=stream_id,
                        generation_id=int(generation_id),
                    )
            self._requests[request_id] = _RequestState(
                stream_id=stream_id,
                generation_id=int(generation_id),
            )
            self.events.append({
                "event": "FLOW_REQUEST_REGISTER",
                "request_id": request_id,
                "stream_id": stream_id,
                "generation_id": int(generation_id),
            })

    def _validate_request_locked(
        self, request_id: str, stream_id: str, generation_id: int
    ) -> _RequestState:
        state = self._requests.get(request_id)
        if state is None:
            raise OnlineAcousticRouterError(f"request is not registered: {request_id}")
        if state.stream_id != str(stream_id) or state.generation_id != int(generation_id):
            raise OnlineAcousticRouterError(
                f"acoustic identity mismatch for {request_id}: "
                f"expected=({state.stream_id},{state.generation_id}) "
                f"got=({stream_id},{generation_id})"
            )
        if state.in_flight:
            raise OnlineAcousticRouterError(f"request already has an acoustic step in flight: {request_id}")
        return state

    def submit(
        self,
        request_id: str,
        *,
        stream_id: str,
        generation_id: int,
        tokens: tuple[int, ...] | list[int],
        last_chunk: bool,
    ) -> dict[str, Any]:
        request_id = str(request_id)
        token_tuple = tuple(int(token) for token in tokens)
        if not token_tuple or any(token < 0 for token in token_tuple):
            raise ValueError("tokens must contain at least one non-negative integer")
        with self._condition:
            state = self._validate_request_locked(request_id, stream_id, generation_id)
            batch = AcousticTokenBatch(
                request_id=request_id,
                stream_id=str(stream_id),
                generation_id=int(generation_id),
                sequence_no=state.next_sequence,
                stoken_ids=token_tuple,
                source_execution_id=f"paper-online-{request_id}-{state.next_sequence}",
                state_version=state.next_version,
                created_monotonic_ns=time.monotonic_ns(),
                last_chunk=bool(last_chunk),
            )
            state.in_flight = True
            pending = _Pending(batch=batch, submitted_ns=time.monotonic_ns(), done=threading.Event())
            self.events.append({
                "event": "FLOW_STEP_READY",
                "request_id": request_id,
                "sequence_no": batch.sequence_no,
                "state_version": batch.state_version,
            })
            if self._coordinator is None:
                self._pending.append(pending)
                self._condition.notify_all()
        if self._coordinator is not None:
            started = time.perf_counter()
            try:
                progress = self._coordinator.submit_async(batch).result(timeout=120.0)
                self._complete(
                    pending,
                    progress,
                    int(getattr(progress, "flow_batch_size", 1)),
                    time.perf_counter() - started,
                )
            except BaseException as exc:
                self._fail(pending, exc)
        if not pending.done.wait(timeout=120.0):
            raise TimeoutError(f"online acoustic step timed out: {request_id}")
        if pending.error is not None:
            raise pending.error
        if pending.result is None:
            raise OnlineAcousticRouterError("online acoustic step completed without a result")
        return dict(pending.result)

    @staticmethod
    def _compatible(first: AcousticTokenBatch, candidate: AcousticTokenBatch) -> bool:
        return (
            len(first.stoken_ids) == len(candidate.stoken_ids)
            and first.last_chunk == candidate.last_chunk
            and first.generation_id == candidate.generation_id
        )

    def _take_group_locked(self) -> tuple[_Pending, ...]:
        first = self._pending.popleft()
        group = [first]
        remaining: deque[_Pending] = deque()
        for pending in self._pending:
            if (
                len(group) < self.spec.max_flow_batch_size
                and self._compatible(first.batch, pending.batch)
            ):
                group.append(pending)
            else:
                remaining.append(pending)
        self._pending = remaining
        return tuple(group)

    def _complete(
        self, pending: _Pending, progress: Any, batch_size: int, synth_sec: float
    ) -> None:
        batch = pending.batch
        with self._condition:
            state = self._requests.get(batch.request_id)
            if (
                state is None
                or state.stream_id != batch.stream_id
                or state.generation_id != batch.generation_id
            ):
                pending.error = OnlineAcousticRouterError(
                    f"stale acoustic completion dropped for {batch.request_id}"
                )
            else:
                records = tuple(getattr(progress, "pcm_records", ()) or ())
                for record in records:
                    if not isinstance(record, AcousticPcmRecord):
                        pending.error = OnlineAcousticRouterError("lane returned an invalid PCM record")
                        break
                    if (
                        record.request_id != batch.request_id
                        or record.stream_id != batch.stream_id
                        or record.generation_id != batch.generation_id
                        or record.sequence_no != batch.sequence_no
                    ):
                        pending.error = OnlineAcousticRouterError(
                            f"PCM identity mismatch for {batch.request_id}"
                        )
                        break
                if pending.error is None:
                    state.next_sequence = batch.sequence_no + 1
                    state.next_version = int(getattr(progress, "state_version", batch.state_version + 1))
                    pending.result = {
                        "pcm_bytes": b"".join(record.pcm_bytes for record in records),
                        "sample_rate": records[0].sample_rate if records else 24000,
                        "worker_id": int(getattr(progress, "worker_id", -1)),
                        "flow_batch_size": int(batch_size),
                        "sequence_no": batch.sequence_no,
                        "state_version": state.next_version,
                        "checkpoint_count": int(getattr(progress, "checkpoint_count", 0)),
                        "restore_count": int(getattr(progress, "restore_count", 0)),
                        "synth_sec": float(synth_sec),
                    }
                state.in_flight = False
            pending.done.set()
            self._condition.notify_all()

    def _fail(self, pending: _Pending, error: BaseException) -> None:
        with self._condition:
            state = self._requests.get(pending.batch.request_id)
            if state is not None:
                state.in_flight = False
            pending.error = error
            pending.done.set()
            self._condition.notify_all()

    def _run_group(self, group: tuple[_Pending, ...]) -> None:
        batch_size = len(group)
        started = time.perf_counter()
        self.events.append({
            "event": "FLOW_BATCH_FORMED",
            "batch_size": batch_size,
            "request_ids": [pending.batch.request_id for pending in group],
            "step_index": 0,
        })
        try:
            with self._lane_lock:
                for pending in group:
                    self.lane.submit(pending.batch)
                progress_by_id = {}
                while len(progress_by_id) < batch_size:
                    progress = self.lane.process_one()
                    if progress is None:
                        raise OnlineAcousticRouterError("acoustic lane made no progress")
                    progress_by_id[progress.request_id] = progress
            self.events.append({
                "event": "FLOW_BATCH_COMPLETE",
                "batch_size": batch_size,
                "request_ids": [pending.batch.request_id for pending in group],
            })
            synth_sec = time.perf_counter() - started
            for pending in group:
                progress = progress_by_id.get(pending.batch.request_id)
                if progress is None:
                    self._fail(pending, OnlineAcousticRouterError("missing lane progress"))
                else:
                    self._complete(pending, progress, batch_size, synth_sec)
        except BaseException as exc:
            self.events.append({
                "event": "FLOW_BATCH_FAILED",
                "batch_size": batch_size,
                "request_ids": [pending.batch.request_id for pending in group],
                "error": f"{type(exc).__name__}: {exc}",
            })
            for pending in group:
                self._fail(pending, exc)

    def _run(self) -> None:
        _configure_acoustic_cuda_device()
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._pending:
                    return
                if self.spec.max_flow_batch_size > 1 and self.max_batch_wait_ms > 0:
                    deadline = time.monotonic() + self.max_batch_wait_ms / 1000.0
                    while len(self._pending) < self.spec.max_flow_batch_size and not self._stopping:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                group = self._take_group_locked()
            self._run_group(group)

    def unregister(self, request_id: str) -> None:
        request_id = str(request_id)
        with self._condition:
            state = self._requests.pop(request_id, None)
            remaining: deque[_Pending] = deque()
            for pending in self._pending:
                if pending.batch.request_id == request_id:
                    pending.error = OnlineAcousticRouterError(
                        f"request unregistered before acoustic completion: {request_id}"
                    )
                    pending.done.set()
                else:
                    remaining.append(pending)
            self._pending = remaining
            self.events.append({"event": "FLOW_REQUEST_UNREGISTER", "request_id": request_id})
            self._condition.notify_all()
        with self._lane_lock:
            if self._coordinator is not None and state is not None:
                self._coordinator.cancel(
                    request_id,
                    generation_id=state.generation_id,
                )
            else:
                self.lane.cancel(request_id)

    def reset(self, request_id: str, *, stream_id: str, generation_id: int) -> None:
        self.unregister(request_id)
        self.register(request_id, stream_id=stream_id, generation_id=generation_id)

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            for pending in self._pending:
                pending.error = OnlineAcousticRouterError("router closed")
                pending.done.set()
            self._pending.clear()
            self._condition.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        with self._lane_lock:
            if self._coordinator is not None:
                self._coordinator.close()
            self.lane.close()


@dataclass
class _RegistryEntry:
    router: OnlineAcousticRouter
    model: Any
    references: int


class OnlineAcousticRouterRegistry:
    """Share benchmark routers across sessions within one service process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], _RegistryEntry] = {}

    def acquire(
        self,
        spec: PaperSystemSpec,
        *,
        model: Any,
        prompt_wav: str,
        worker_count: int,
        max_batch_wait_ms: float | None = None,
        lane_factory: Callable[..., Any] | None = None,
    ) -> OnlineAcousticRouter:
        key = (spec.system_id, str(prompt_wav))
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                if entry.model is not model:
                    raise OnlineAcousticRouterError(
                        f"router key {key!r} is already bound to another acoustic model"
                    )
                entry.references += 1
                return entry.router
            router = OnlineAcousticRouter(
                spec,
                model=model,
                prompt_wav=str(prompt_wav),
                worker_count=worker_count,
                max_batch_wait_ms=max_batch_wait_ms,
                lane_factory=lane_factory,
            )
            self._entries[key] = _RegistryEntry(router=router, model=model, references=1)
            return router

    def release(self, spec: PaperSystemSpec, *, prompt_wav: str) -> None:
        key = (spec.system_id, str(prompt_wav))
        router = None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.references -= 1
            if entry.references <= 0:
                router = entry.router
                del self._entries[key]
        if router is not None:
            router.close()

    def close_all(self) -> None:
        with self._lock:
            routers = [entry.router for entry in self._entries.values()]
            self._entries.clear()
        for router in routers:
            router.close()
