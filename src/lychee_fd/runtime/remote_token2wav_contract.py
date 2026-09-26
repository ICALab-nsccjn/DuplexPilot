"""Backend-neutral contract for remote Token2Wav state migration.

This module describes an API boundary only.  It is deliberately not imported by
the current remote service and does not select workers or alter serving semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class RemoteToken2WavCheckpoint:
    """Serializable logical identity and continuation envelope."""

    request_id: str
    stream_id: str
    generation_id: int
    token_position: int
    pending_pcm: tuple[bytes, ...] = ()
    cancelled: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.request_id or not self.stream_id:
            raise ValueError("request_id and stream_id are required")
        if self.generation_id < 0:
            raise ValueError("generation_id must be non-negative")
        if self.token_position < 0:
            raise ValueError("token_position must be non-negative")
        if self.schema_version != 1:
            raise ValueError("unsupported checkpoint schema_version")
        if not isinstance(self.pending_pcm, tuple):
            raise TypeError("pending_pcm must be an immutable tuple")
        if not all(isinstance(chunk, bytes) for chunk in self.pending_pcm):
            raise TypeError("pending_pcm must contain bytes")


class RemoteToken2WavStateContract(ABC):
    """Required operations for a future remote Token2Wav implementation."""

    @abstractmethod
    def export_state(self) -> RemoteToken2WavCheckpoint:
        """Export a copy-safe logical checkpoint."""

    @abstractmethod
    def validate_state(self, state: RemoteToken2WavCheckpoint) -> None:
        """Validate identity, schema, and continuation metadata."""

    @abstractmethod
    def import_state(self, state: RemoteToken2WavCheckpoint) -> None:
        """Atomically restore a validated checkpoint."""

    @abstractmethod
    def resume_stream(self, tokens: Sequence[int]) -> Any:
        """Continue the logical stream after import."""

    @abstractmethod
    def cancel_stream(self) -> None:
        """Cancel the generation and reject stale output."""
