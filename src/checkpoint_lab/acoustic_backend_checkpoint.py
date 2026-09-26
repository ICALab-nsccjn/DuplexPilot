"""APR-independent checkpoint backend protocol for one acoustic stream."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AcousticBackendCheckpoint(ABC):
    """Explicit state/output contract without worker-selection semantics."""

    @abstractmethod
    def capture_state(self) -> Any:
        """Return a copy-safe logical checkpoint."""

    @abstractmethod
    def restore_state(self, checkpoint: Any) -> None:
        """Validate and restore a logical checkpoint."""

    @abstractmethod
    def process(self, tokens, *, last_chunk: bool = False) -> None:
        """Advance the current logical stream."""

    @abstractmethod
    def commit_output(self):
        """Commit only current-generation output."""

    @abstractmethod
    def cancel(self) -> None:
        """Make the current generation terminal and drop pending output."""
