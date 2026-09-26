"""Immutable, request-owned messages exchanged by the APR prototype.

The contract deliberately contains no model or Token2Wav implementation.  It
is the fail-closed boundary between row-aware token extraction and acoustic
progression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class APRContractError(ValueError):
    """Raised when an APR message violates identity or ordering invariants."""


def _require_nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise APRContractError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise APRContractError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class AcousticTokenBatch:
    """One immutable model-to-acoustic handoff for one logical request."""

    request_id: str
    stream_id: str
    generation_id: int
    sequence_no: int
    stoken_ids: tuple[int, ...]
    source_execution_id: str
    state_version: int
    created_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_nonempty_text("request_id", self.request_id))
        object.__setattr__(self, "stream_id", _require_nonempty_text("stream_id", self.stream_id))
        object.__setattr__(self, "source_execution_id", _require_nonempty_text("source_execution_id", self.source_execution_id))
        object.__setattr__(self, "generation_id", _require_nonnegative_int("generation_id", self.generation_id))
        object.__setattr__(self, "sequence_no", _require_nonnegative_int("sequence_no", self.sequence_no))
        object.__setattr__(self, "state_version", _require_nonnegative_int("state_version", self.state_version))
        object.__setattr__(self, "created_monotonic_ns", _require_nonnegative_int("created_monotonic_ns", self.created_monotonic_ns))
        try:
            tokens = tuple(self.stoken_ids)
        except TypeError as exc:
            raise APRContractError("stoken_ids must be an iterable of integers") from exc
        if not tokens or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
            raise APRContractError("stoken_ids must contain at least one non-negative integer")
        object.__setattr__(self, "stoken_ids", tokens)


@dataclass(frozen=True)
class AcousticPcmRecord:
    """One non-empty PCM result retaining the complete logical owner key."""

    request_id: str
    stream_id: str
    generation_id: int
    sequence_no: int
    pcm_bytes: bytes
    sample_rate: int
    pcm_seq: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_nonempty_text("request_id", self.request_id))
        object.__setattr__(self, "stream_id", _require_nonempty_text("stream_id", self.stream_id))
        object.__setattr__(self, "generation_id", _require_nonnegative_int("generation_id", self.generation_id))
        object.__setattr__(self, "sequence_no", _require_nonnegative_int("sequence_no", self.sequence_no))
        if not isinstance(self.pcm_bytes, (bytes, bytearray, memoryview)) or not self.pcm_bytes:
            raise APRContractError("pcm_bytes must be non-empty bytes")
        object.__setattr__(self, "pcm_bytes", bytes(self.pcm_bytes))
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise APRContractError("sample_rate must be a positive integer")
        object.__setattr__(self, "pcm_seq", _require_nonnegative_int("pcm_seq", self.pcm_seq))


def validate_sequence_chain(previous: AcousticTokenBatch, current: AcousticTokenBatch) -> bool:
    """Validate the next token batch for one request/generation.

    Sequence validation is explicit instead of being inferred from physical
    row order.  A caller that sees a gap, duplicate, request switch, or stale
    generation must fail closed before enqueueing the current batch.
    """

    if not isinstance(previous, AcousticTokenBatch) or not isinstance(current, AcousticTokenBatch):
        raise APRContractError("sequence chain requires AcousticTokenBatch values")
    if current.request_id != previous.request_id:
        raise APRContractError("request_id changed within an APR sequence")
    if current.stream_id != previous.stream_id:
        raise APRContractError("stream_id changed within an APR sequence")
    if current.generation_id != previous.generation_id:
        raise APRContractError("generation_id changed within an APR sequence")
    if current.sequence_no != previous.sequence_no + 1:
        raise APRContractError("APR sequence numbers must increase by exactly one")
    if current.created_monotonic_ns < previous.created_monotonic_ns:
        raise APRContractError("APR creation timestamps must be monotonic")
    return True


def validate_sequence_batches(batches: Iterable[AcousticTokenBatch]) -> bool:
    """Validate an entire non-empty sequence in logical order."""

    iterator = iter(batches)
    try:
        previous = next(iterator)
    except StopIteration as exc:
        raise APRContractError("APR sequence cannot be empty") from exc
    for current in iterator:
        validate_sequence_chain(previous, current)
        previous = current
    return True
