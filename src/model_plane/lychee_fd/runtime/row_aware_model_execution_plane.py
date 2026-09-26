"""Opt-in request-aware model execution plane.

This module owns only scheduling and output routing.  It deliberately knows
nothing about vLLM internals, acoustic state, or model tensors.  A caller
supplies a single ``step_fn``; the plane invokes it from one driver thread with
up to ``max_batch_size`` request rounds.  The returned objects are routed back
to the request-owned futures by ``output_request_id``.

The implementation is intentionally small and benchmark-only.  The default
Lychee runtime does not construct it; production integration must explicitly
opt in and provide a request-safe step function.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Deque, Mapping, Sequence


class ModelExecutionPlaneError(RuntimeError):
    """Base error for fail-closed execution-plane operations."""


class ModelExecutionPlaneClosed(ModelExecutionPlaneError):
    """Raised when work is submitted after the plane has closed."""


class ModelExecutionPlaneBackpressure(ModelExecutionPlaneError):
    """Raised when the bounded demand queue is full."""


class ModelExecutionPlaneStale(ModelExecutionPlaneError):
    """Raised when a generation fence invalidates a request round."""


@dataclass(frozen=True)
class ModelRequestHandle:
    request_id: str
    generation_id: int


@dataclass(frozen=True)
class ModelOutputEnvelope:
    """Metadata-only request-owned output contract.

    ``raw_output`` is intentionally absent: real engine objects must remain
    inside the injected adapter and are never serialized by this plane.
    """

    request_id: str
    generation_id: int
    sequence_no: int
    text_token: int
    stoken_token: int
    control_token: int
    phase: str
    finished: bool
    finish_reason: str | None = None


@dataclass(frozen=True)
class ModelRound:
    """One request's demand passed to the single driver."""

    request_id: str
    generation_id: int
    sequence_no: int
    round_input: Any


@dataclass
class _PendingRound:
    round: ModelRound
    future: Future


@dataclass
class _DriverCommand:
    """A non-round operation that must run on the engine driver thread."""

    fn: Callable[[], Any]
    future: Future
    description: str


@dataclass
class _RegisteredRequest:
    generation_id: int
    next_sequence_no: int = 0
    cancelled: bool = False
    active: bool = False
    finished: bool = False
    cancel_callback_sent: bool = False


def _request_field(request_spec: Any, name: str, default: Any = None) -> Any:
    if isinstance(request_spec, Mapping):
        return request_spec.get(name, default)
    return getattr(request_spec, name, default)


class RowAwareModelExecutionPlane:
    """Run request rounds through one central, row-aware driver.

    Parameters:
        step_fn: Called only by the driver thread with a tuple of ``ModelRound``
            objects.  It must return one output object per selected request.
        output_request_id: Extracts the logical request ID from an output.
        output_generation_id: Optional generation extractor.  If omitted, the
            submitted generation is used and the raw output is not inspected.
        cancel_fn: Optional callback invoked by the driver after a cancelled
            request's in-flight step has returned.

    There is deliberately no artificial coalescing sleep.  Requests that are
    already waiting when the driver takes a scheduling turn can form a model
    batch; otherwise the step is a singleton.  This keeps arrival timestamps
    and backpressure semantics explicit.
    """

    def __init__(
        self,
        step_fn: Callable[[tuple[ModelRound, ...]], Sequence[Any]],
        *,
        max_batch_size: int = 2,
        max_pending_rounds: int = 1024,
        output_request_id: Callable[[Any], str] | None = None,
        output_generation_id: Callable[[Any], int | None] | None = None,
        cancel_fn: Callable[[str, int], None] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        clock_ns: Callable[[], int] | None = None,
        start_gate: threading.Event | None = None,
    ) -> None:
        if not callable(step_fn):
            raise TypeError("step_fn must be callable")
        if isinstance(max_batch_size, bool) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if isinstance(max_pending_rounds, bool) or max_pending_rounds <= 0:
            raise ValueError("max_pending_rounds must be positive")
        self._step_fn = step_fn
        self._max_batch_size = int(max_batch_size)
        self._max_pending_rounds = int(max_pending_rounds)
        self._output_request_id = output_request_id or self._default_output_id
        self._output_generation_id = output_generation_id
        self._cancel_fn = cancel_fn
        self._event_sink = event_sink
        self._clock_ns = clock_ns or time.monotonic_ns
        # Test-only deterministic gate.  It is intentionally absent from the
        # production integration; it lets contract tests queue two demands
        # before the driver takes its first scheduling turn.
        self._start_gate = start_gate
        self._condition = threading.Condition()
        self._requests: dict[str, _RegisteredRequest] = {}
        self._pending: Deque[_PendingRound] = deque()
        self._commands: Deque[_DriverCommand] = deque()
        self._active: tuple[_PendingRound, ...] = ()
        self._closed = False
        self._driver_error: BaseException | None = None
        self._driver = threading.Thread(
            target=self._run,
            name="lychee-row-aware-model-driver",
            daemon=True,
        )
        self._driver.start()

    @staticmethod
    def _default_output_id(output: Any) -> str:
        if isinstance(output, Mapping):
            value = output.get("request_id")
        else:
            value = getattr(output, "request_id", None)
        request_id = str(value or "")
        if not request_id:
            raise ModelExecutionPlaneError("model output is missing request_id")
        return request_id

    def _emit(self, event: str, **payload: Any) -> None:
        if self._event_sink is None:
            return
        record = {
            "event": str(event),
            "timestamp_monotonic_ns": int(self._clock_ns()),
        }
        record.update(payload)
        try:
            self._event_sink(record)
        except Exception:
            # Telemetry must not alter execution semantics.
            pass

    def register(self, request_spec: Any) -> ModelRequestHandle:
        request_id = str(_request_field(request_spec, "request_id", "") or "")
        generation_raw = _request_field(request_spec, "generation_id", 0)
        if not request_id:
            raise ValueError("request_spec.request_id is required")
        if isinstance(generation_raw, bool):
            raise ValueError("generation_id must be a non-negative integer")
        try:
            generation_id = int(generation_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("generation_id must be a non-negative integer") from exc
        if generation_id < 0:
            raise ValueError("generation_id must be non-negative")
        with self._condition:
            if self._closed:
                raise ModelExecutionPlaneClosed("model execution plane is closed")
            if request_id in self._requests:
                raise ValueError(f"request already registered: {request_id}")
            self._requests[request_id] = _RegisteredRequest(generation_id)
            self._emit(
                "MODEL_REQUEST_REGISTER",
                request_id=request_id,
                generation_id=generation_id,
            )
            self._condition.notify_all()
        return ModelRequestHandle(request_id, generation_id)

    def submit_round(
        self,
        request_id: str,
        generation_id: int,
        round_input: Any,
    ) -> Future:
        request_key = str(request_id)
        future: Future = Future()
        try:
            generation = int(generation_id)
        except (TypeError, ValueError) as exc:
            future.set_exception(ModelExecutionPlaneStale("invalid generation"))
            return future
        with self._condition:
            if self._closed:
                future.set_exception(ModelExecutionPlaneClosed("model execution plane is closed"))
                return future
            state = self._requests.get(request_key)
            if state is None:
                future.set_exception(ModelExecutionPlaneStale("request is not registered"))
                return future
            if state.generation_id != generation or state.cancelled:
                future.set_exception(ModelExecutionPlaneStale("generation fence rejected round"))
                return future
            if state.active or any(
                item.round.request_id == request_key for item in self._pending
            ):
                future.set_exception(
                    ModelExecutionPlaneError(
                        f"request already has a pending/in-flight round: {request_key}"
                    )
                )
                return future
            if len(self._pending) >= self._max_pending_rounds:
                future.set_exception(ModelExecutionPlaneBackpressure("demand queue is full"))
                self._emit(
                    "MODEL_EXECUTION_BACKPRESSURE",
                    request_id=request_key,
                    generation_id=generation,
                )
                return future
            sequence_no = state.next_sequence_no
            state.next_sequence_no += 1
            pending = _PendingRound(
                ModelRound(request_key, generation, sequence_no, round_input),
                future,
            )
            self._pending.append(pending)
            self._emit(
                "MODEL_DECODE_DEMAND",
                request_id=request_key,
                generation_id=generation,
                sequence_no=sequence_no,
                queue_depth=len(self._pending),
            )
            self._condition.notify_all()
        return future

    def call_on_driver(
        self,
        fn: Callable[[], Any],
        *,
        description: str = "driver_command",
    ) -> Future:
        """Run a control-plane operation on the single driver thread.

        vLLM's ``add_request`` and abort operations share the same mutable
        engine as ``step``.  The real integration uses this queue so those
        operations cannot race a model step.  The method is intentionally
        generic and contains no vLLM dependency, which keeps the fake-engine
        contract independently testable.
        """
        if not callable(fn):
            raise TypeError("driver command must be callable")
        future: Future = Future()
        with self._condition:
            if self._closed:
                future.set_exception(
                    ModelExecutionPlaneClosed("model execution plane is closed")
                )
                return future
            self._commands.append(
                _DriverCommand(fn, future, str(description or "driver_command"))
            )
            self._emit(
                "MODEL_DRIVER_COMMAND_ENQUEUE",
                description=str(description or "driver_command"),
                queue_depth=len(self._commands),
            )
            self._condition.notify_all()
        return future

    def _take_batch_locked(self) -> tuple[_PendingRound, ...]:
        selected: list[_PendingRound] = []
        remaining: Deque[_PendingRound] = deque()
        selected_ids: set[str] = set()
        while self._pending:
            item = self._pending.popleft()
            request_id = item.round.request_id
            if len(selected) < self._max_batch_size and request_id not in selected_ids:
                selected.append(item)
                selected_ids.add(request_id)
            else:
                remaining.append(item)
        self._pending = remaining
        for item in selected:
            state = self._requests.get(item.round.request_id)
            if state is not None:
                state.active = True
        return tuple(selected)

    def _fail_items(self, items: Sequence[_PendingRound], error: BaseException) -> None:
        with self._condition:
            for item in items:
                state = self._requests.get(item.round.request_id)
                if state is not None:
                    state.active = False
                if not item.future.done():
                    item.future.set_exception(error)
            self._condition.notify_all()

    def _run_one_batch(self, items: tuple[_PendingRound, ...]) -> None:
        rounds = tuple(item.round for item in items)
        self._active = items
        self._emit(
            "MODEL_ENGINE_BATCH_DISPATCH",
            request_ids=tuple(item.request_id for item in rounds),
            model_batch_size=len(rounds),
        )
        try:
            outputs = self._step_fn(rounds)
            if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
                raise ModelExecutionPlaneError("step_fn must return a sequence")
            by_id: dict[str, Any] = {}
            for output in outputs:
                output_id = str(self._output_request_id(output))
                if output_id in by_id:
                    raise ModelExecutionPlaneError(
                        f"duplicate model output for request {output_id}"
                    )
                by_id[output_id] = output
            selected_ids = {item.round.request_id for item in items}
            if set(by_id) != selected_ids:
                raise ModelExecutionPlaneError(
                    "model output set diverged from selected requests: "
                    f"expected={tuple(sorted(selected_ids))} got={tuple(sorted(by_id))}"
                )
            with self._condition:
                for item in items:
                    request_id = item.round.request_id
                    state = self._requests.get(request_id)
                    output = by_id[request_id]
                    stale = (
                        state is None
                        or state.cancelled
                        or state.generation_id != item.round.generation_id
                    )
                    if self._output_generation_id is not None and not stale:
                        output_generation = self._output_generation_id(output)
                        stale = output_generation is not None and int(output_generation) != item.round.generation_id
                    if stale:
                        if not item.future.done():
                            item.future.set_exception(ModelExecutionPlaneStale("stale model output dropped"))
                        self._emit(
                            "MODEL_STALE_OUTPUT_DROP",
                            request_id=request_id,
                            generation_id=item.round.generation_id,
                            sequence_no=item.round.sequence_no,
                        )
                    elif not item.future.done():
                        item.future.set_result(output)
                    if state is not None:
                        state.active = False
                        if state.finished:
                            self._requests.pop(request_id, None)
                self._condition.notify_all()
        except BaseException as exc:
            self._fail_items(items, exc)
            with self._condition:
                self._driver_error = exc
                # A backend failure is terminal for this execution plane.  Do
                # not silently accept more demands after a failed engine step.
                self._closed = True
                for pending in self._pending:
                    if not pending.future.done():
                        pending.future.set_exception(exc)
                self._pending.clear()
                self._condition.notify_all()
        finally:
            self._active = ()
            cancelled = []
            with self._condition:
                for item in items:
                    state = self._requests.get(item.round.request_id)
                    if state is not None and state.cancelled:
                        cancelled.append((item.round.request_id, item.round.generation_id))
            if self._cancel_fn is not None:
                for request_id, generation_id in cancelled:
                    with self._condition:
                        state = self._requests.get(request_id)
                        already_sent = state is None or state.cancel_callback_sent
                        if state is not None:
                            state.cancel_callback_sent = True
                    if not already_sent:
                        try:
                            self._cancel_fn(request_id, generation_id)
                        except Exception:
                            pass

    def _run(self) -> None:
        if self._start_gate is not None:
            while not self._start_gate.is_set():
                with self._condition:
                    if self._closed:
                        return
                self._start_gate.wait(timeout=0.01)
        while True:
            with self._condition:
                while not self._commands and not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._commands and not self._pending:
                    return
                command = self._commands.popleft() if self._commands else None
                items = () if command is not None else self._take_batch_locked()
            if command is not None:
                try:
                    result = command.fn()
                except BaseException as exc:
                    if not command.future.done():
                        command.future.set_exception(exc)
                    self._emit(
                        "MODEL_DRIVER_COMMAND_ERROR",
                        description=command.description,
                        error_type=type(exc).__name__,
                    )
                    # A control-plane failure (for example, an engine abort
                    # or add_request exception) leaves the underlying engine
                    # state unknown.  Keep the same fail-closed invariant as
                    # a failed model step: reject all queued work and make
                    # later submissions observe a terminal driver error.
                    with self._condition:
                        self._driver_error = exc
                        self._closed = True
                        for pending in self._pending:
                            if not pending.future.done():
                                pending.future.set_exception(exc)
                        self._pending.clear()
                        for queued_command in self._commands:
                            if not queued_command.future.done():
                                queued_command.future.set_exception(exc)
                        self._commands.clear()
                        self._condition.notify_all()
                else:
                    if not command.future.done():
                        command.future.set_result(result)
                    self._emit(
                        "MODEL_DRIVER_COMMAND_COMPLETE",
                        description=command.description,
                    )
                continue
            if items:
                self._run_one_batch(items)

    def cancel(self, request_id: str, generation_id: int) -> None:
        request_key = str(request_id)
        generation = int(generation_id)
        callback_now = False
        with self._condition:
            state = self._requests.get(request_key)
            if state is None or state.generation_id != generation:
                return
            state.cancelled = True
            kept: Deque[_PendingRound] = deque()
            for item in self._pending:
                if item.round.request_id == request_key and item.round.generation_id == generation:
                    if not item.future.done():
                        item.future.set_exception(ModelExecutionPlaneStale("round cancelled"))
                else:
                    kept.append(item)
            self._pending = kept
            callback_now = not state.active and not state.cancel_callback_sent
            if callback_now:
                state.cancel_callback_sent = True
            self._emit(
                "MODEL_REQUEST_CANCEL",
                request_id=request_key,
                generation_id=generation,
            )
            self._condition.notify_all()
        if callback_now and self._cancel_fn is not None:
            self._cancel_fn(request_key, generation)

    def reset(self, request_id: str, generation_id: int) -> ModelRequestHandle:
        request_key = str(request_id)
        generation = int(generation_id)
        with self._condition:
            state = self._requests.get(request_key)
            if state is None:
                raise ModelExecutionPlaneStale("cannot reset an unknown request")
            old_generation = state.generation_id
        self.cancel(request_key, old_generation)
        with self._condition:
            state = self._requests.get(request_key)
            if state is None:
                raise ModelExecutionPlaneStale("request disappeared during reset")
            state.generation_id = generation
            state.next_sequence_no = 0
            state.cancelled = False
            state.finished = False
            # If an old-generation step is still running, keep ``active`` set
            # until its driver turn completes.  This preserves the one-round
            # per-request invariant across a reset-generation fence.
            state.cancel_callback_sent = False
            self._emit(
                "MODEL_REQUEST_RESET",
                request_id=request_key,
                generation_id=generation,
            )
            self._condition.notify_all()
        return ModelRequestHandle(request_key, generation)

    def finish(self, request_id: str, generation_id: int) -> None:
        request_key = str(request_id)
        generation = int(generation_id)
        with self._condition:
            state = self._requests.get(request_key)
            if state is None or state.generation_id != generation:
                return
            state.cancelled = True
            state.finished = True
            kept: Deque[_PendingRound] = deque()
            for item in self._pending:
                if item.round.request_id == request_key:
                    if not item.future.done():
                        item.future.set_exception(ModelExecutionPlaneStale("request finished"))
                else:
                    kept.append(item)
            self._pending = kept
            if not state.active:
                self._requests.pop(request_key, None)
            self._emit(
                "MODEL_REQUEST_FINISH",
                request_id=request_key,
                generation_id=generation,
            )
            self._condition.notify_all()

    def pending_ids(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(item.round.request_id for item in self._pending)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for item in self._pending:
                if not item.future.done():
                    item.future.set_exception(ModelExecutionPlaneClosed("model execution plane closed"))
            self._pending.clear()
            for command in self._commands:
                if not command.future.done():
                    command.future.set_exception(
                        ModelExecutionPlaneClosed("model execution plane closed")
                    )
            self._commands.clear()
            self._condition.notify_all()
        if threading.current_thread() is not self._driver:
            self._driver.join(timeout=5.0)

    @property
    def driver_error(self) -> BaseException | None:
        with self._condition:
            return self._driver_error

    @property
    def driver_thread_id(self) -> int | None:
        """Return the driver identity once it has entered the thread."""
        ident = self._driver.ident
        return int(ident) if ident is not None else None


__all__ = [
    "ModelExecutionPlaneBackpressure",
    "ModelExecutionPlaneClosed",
    "ModelExecutionPlaneError",
    "ModelExecutionPlaneStale",
    "ModelOutputEnvelope",
    "ModelRequestHandle",
    "ModelRound",
    "RowAwareModelExecutionPlane",
]
