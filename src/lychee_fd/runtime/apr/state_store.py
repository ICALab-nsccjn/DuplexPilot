"""Versioned logical acoustic state for APR worker handoff."""

from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import AcousticPcmRecord, _require_nonempty_text, _require_nonnegative_int


class APRStateError(RuntimeError):
    """Raised when an APR state lease or commit cannot be proven safe."""


@dataclass(frozen=True)
class AcousticLease:
    request_id: str
    generation_id: int
    state_version: int
    lease_token: str


@dataclass(frozen=True)
class AcousticCommitResult:
    request_id: str
    generation_id: int
    state_version: int
    pcm_records: tuple[AcousticPcmRecord, ...]


@dataclass
class _StateEntry:
    generation_id: int
    state_version: int
    state: dict[str, Any]
    active_lease: AcousticLease | None = None
    cancelled_generations: set[int] | None = None
    pcm_records: int = 0

    def __post_init__(self) -> None:
        if self.cancelled_generations is None:
            self.cancelled_generations = set()


class AcousticStateStore:
    """Thread-safe registry of logical state independent of worker slots."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, _StateEntry] = {}

    def acquire(self, request_id: str, generation_id: int, expected_version: int) -> AcousticLease:
        request_id = _require_nonempty_text("request_id", request_id)
        generation_id = _require_nonnegative_int("generation_id", generation_id)
        expected_version = _require_nonnegative_int("expected_version", expected_version)
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                if expected_version != 0:
                    raise APRStateError("new APR request must start at state version 0")
                entry = _StateEntry(generation_id, 0, {})
                self._entries[request_id] = entry
            else:
                if entry.active_lease is not None:
                    raise APRStateError("request already has an active APR lease")
                if generation_id != entry.generation_id:
                    if generation_id <= entry.generation_id or entry.generation_id not in entry.cancelled_generations:
                        raise APRStateError("generation transition was not cancelled explicitly")
                    entry.generation_id = generation_id
                    entry.state_version = 0
                    entry.state = {}
                    entry.pcm_records = 0
                if generation_id in entry.cancelled_generations:
                    raise APRStateError("cancelled APR generation cannot be acquired")
                if expected_version != entry.state_version:
                    raise APRStateError(
                        f"state version mismatch: expected {entry.state_version}, got {expected_version}"
                    )
            lease = AcousticLease(request_id, generation_id, entry.state_version, uuid.uuid4().hex)
            entry.active_lease = lease
            return lease

    def commit(
        self,
        lease: AcousticLease,
        next_state: Mapping[str, Any],
        pcm_records: Sequence[AcousticPcmRecord],
    ) -> AcousticCommitResult:
        if not isinstance(lease, AcousticLease):
            raise APRStateError("commit requires an AcousticLease")
        if not isinstance(next_state, Mapping):
            raise APRStateError("next_state must be a mapping")
        records = tuple(pcm_records)
        if any(not isinstance(record, AcousticPcmRecord) for record in records):
            raise APRStateError("pcm_records must contain AcousticPcmRecord values")
        with self._lock:
            entry = self._entries.get(lease.request_id)
            if entry is None or entry.active_lease != lease:
                raise APRStateError("stale or unknown APR lease")
            if lease.generation_id in entry.cancelled_generations:
                raise APRStateError("cancelled APR generation cannot commit PCM")
            if entry.state_version != lease.state_version:
                raise APRStateError("APR lease state version is stale")
            for record in records:
                if record.request_id != lease.request_id or record.generation_id != lease.generation_id:
                    raise APRStateError("PCM ownership does not match the APR lease")
            next_version = entry.state_version + 1
            entry.state = copy.deepcopy(dict(next_state))
            entry.state_version = next_version
            entry.pcm_records += len(records)
            return AcousticCommitResult(lease.request_id, lease.generation_id, next_version, records)

    def cancel(self, request_id: str, generation_id: int) -> None:
        request_id = _require_nonempty_text("request_id", request_id)
        generation_id = _require_nonnegative_int("generation_id", generation_id)
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                raise APRStateError("cannot cancel an unknown APR request")
            entry.cancelled_generations.add(generation_id)

    def release(self, request_id: str) -> None:
        request_id = _require_nonempty_text("request_id", request_id)
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None:
                entry.active_lease = None

    def remove(self, request_id: str) -> None:
        """Remove logical state after terminal request cleanup."""
        request_id = _require_nonempty_text("request_id", request_id)
        with self._lock:
            self._entries.pop(request_id, None)

    def snapshot(self, request_id: str) -> dict[str, Any] | None:
        request_id = _require_nonempty_text("request_id", request_id)
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return None
            return {
                "request_id": request_id,
                "generation_id": entry.generation_id,
                "state_version": entry.state_version,
                "state": copy.deepcopy(entry.state),
                "active_lease": entry.active_lease is not None,
                "cancelled_generations": sorted(entry.cancelled_generations),
                "pcm_records": entry.pcm_records,
            }
