"""Opt-in lock timing for Lychee lifecycle/serialization audits."""

from __future__ import annotations

from contextlib import contextmanager
import json
import time
from typing import Dict, Iterator, List, Sequence


class LockEventRecorder:
    """Record a real lock scope without changing its ownership semantics."""

    def __init__(self) -> None:
        self.events: List[Dict] = []

    @contextmanager
    def hold(self, lock, lock_name: str, protected_operation: str,
             request_ids: Sequence[str] = (), batch_size: int = 0,
             physical_rows: Sequence[int] = ()) -> Iterator[None]:
        acquire_start = time.perf_counter()
        lock.acquire()
        acquired = time.perf_counter()
        try:
            yield
        finally:
            lock.release()
            released = time.perf_counter()
            self.events.append({
                "lock": str(lock_name),
                "acquire": acquired,
                "release": released,
                "duration": released - acquired,
                "wait_duration": acquired - acquire_start,
                "protected_operation": str(protected_operation),
                "request_ids": [str(x) for x in request_ids],
                "batch_size": int(batch_size),
                "physical_rows": [int(x) for x in physical_rows],
            })

    def to_jsonl(self) -> str:
        return "".join(json.dumps(event, sort_keys=True) + "\n"
                       for event in self.events)
