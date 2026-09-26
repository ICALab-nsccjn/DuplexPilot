"""Bounded per-request ingress queues for APR."""

from __future__ import annotations

import threading
from collections import deque

from .contracts import AcousticTokenBatch, APRContractError, _require_nonempty_text


class APRQueueError(RuntimeError):
    """Raised when a queue operation violates ownership or sequence rules."""


class APRBackpressure(APRQueueError):
    """Raised when APR would have to drop a token batch."""


class AcousticIngressQueue:
    """FIFO queue scoped to one logical request and one active generation."""

    def __init__(self, request_id: str | None = None, capacity: int = 32) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise APRQueueError("APR queue capacity must be a positive integer")
        self._lock = threading.RLock()
        self._request_id = _require_nonempty_text("request_id", request_id) if request_id is not None else None
        self._capacity = capacity
        self._items: deque[AcousticTokenBatch] = deque()
        self._generation_id: int | None = None
        self._next_sequence: int | None = None
        self._cancelled_generations: set[int] = set()
        self._closed = False

    def put(self, batch: AcousticTokenBatch) -> None:
        if not isinstance(batch, AcousticTokenBatch):
            raise APRQueueError("APR ingress accepts AcousticTokenBatch values only")
        with self._lock:
            if self._closed:
                raise APRQueueError("APR ingress queue is closed")
            if self._request_id is None:
                self._request_id = batch.request_id
            if batch.request_id != self._request_id:
                raise APRQueueError("token batch request_id does not match its queue")
            if batch.generation_id in self._cancelled_generations:
                raise APRQueueError("token batch belongs to a cancelled generation")
            if self._generation_id is None:
                if batch.sequence_no != 0:
                    raise APRQueueError("a new APR generation must start at sequence 0")
                self._generation_id = batch.generation_id
                self._next_sequence = 0
            elif batch.generation_id != self._generation_id:
                raise APRQueueError("generation transition requires explicit cancellation")
            if batch.sequence_no != self._next_sequence:
                raise APRQueueError(
                    f"expected sequence {self._next_sequence}, got {batch.sequence_no}"
                )
            if len(self._items) >= self._capacity:
                raise APRBackpressure(
                    f"APR queue capacity {self._capacity} reached for {self._request_id}"
                )
            self._items.append(batch)
            self._next_sequence += 1

    def get(self) -> AcousticTokenBatch | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()

    def cancel_generation(self, generation_id: int) -> int:
        if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 0:
            raise APRQueueError("generation_id must be a non-negative integer")
        with self._lock:
            self._cancelled_generations.add(generation_id)
            before = len(self._items)
            if self._items:
                self._items = deque(item for item in self._items if item.generation_id != generation_id)
            drained = before - len(self._items)
            if self._generation_id == generation_id:
                self._generation_id = None
                self._next_sequence = None
            return drained

    def depth(self) -> int:
        with self._lock:
            return len(self._items)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._items.clear()
            self._generation_id = None
            self._next_sequence = None
