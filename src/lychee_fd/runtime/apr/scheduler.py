"""Diagnostic round-robin scheduler for APR acoustic ingress queues."""

from __future__ import annotations

import threading
from collections import deque

from .queue import AcousticIngressQueue
from .contracts import _require_nonempty_text


class APRSchedulerError(RuntimeError):
    """Raised when an APR scheduler registration or lease is invalid."""


class AcousticScheduler:
    """Select ready logical requests without importing the model scheduler."""

    def __init__(self, worker_count: int = 1) -> None:
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
            raise APRSchedulerError("worker_count must be a positive integer")
        self._lock = threading.RLock()
        self._worker_count = worker_count
        self._queues: dict[str, AcousticIngressQueue] = {}
        self._ready: deque[str] = deque()
        self._ready_set: set[str] = set()
        self._leased: set[str] = set()

    def register(self, request_id: str, queue: AcousticIngressQueue) -> None:
        request_id = _require_nonempty_text("request_id", request_id)
        if not isinstance(queue, AcousticIngressQueue):
            raise APRSchedulerError("APR scheduler requires an AcousticIngressQueue")
        with self._lock:
            if request_id in self._queues:
                raise APRSchedulerError(f"request already registered: {request_id}")
            self._queues[request_id] = queue

    def unregister(self, request_id: str) -> None:
        request_id = _require_nonempty_text("request_id", request_id)
        with self._lock:
            if request_id not in self._queues:
                return
            del self._queues[request_id]
            self._leased.discard(request_id)
            self._ready_set.discard(request_id)
            self._ready = deque(item for item in self._ready if item != request_id)

    def mark_ready(self, request_id: str) -> None:
        request_id = _require_nonempty_text("request_id", request_id)
        with self._lock:
            queue = self._queues.get(request_id)
            if queue is None:
                raise APRSchedulerError(f"request is not registered: {request_id}")
            if request_id in self._leased or request_id in self._ready_set:
                return
            if queue.depth() <= 0:
                return
            self._ready.append(request_id)
            self._ready_set.add(request_id)

    def next_ready(self) -> str | None:
        with self._lock:
            if len(self._leased) >= self._worker_count:
                return None
            while self._ready:
                request_id = self._ready.popleft()
                self._ready_set.discard(request_id)
                queue = self._queues.get(request_id)
                if queue is None or request_id in self._leased or queue.depth() <= 0:
                    continue
                self._leased.add(request_id)
                return request_id
            return None

    def release(self, request_id: str) -> None:
        request_id = _require_nonempty_text("request_id", request_id)
        with self._lock:
            if request_id not in self._queues:
                return
            self._leased.discard(request_id)
            queue = self._queues[request_id]
            if queue.depth() > 0 and request_id not in self._ready_set:
                self._ready.append(request_id)
                self._ready_set.add(request_id)

