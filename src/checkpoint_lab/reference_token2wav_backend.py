"""Test-only deterministic semantic oracle; never used by production serving."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Checkpoint:
    request_id: str
    stream_id: str
    generation_id: int
    position: int
    pending_pcm: bytes
    events: tuple[tuple[int, int], ...]
    cancelled: bool


class DeterministicReferenceToken2Wav:
    """Small CPU oracle for exact checkpoint/event contract tests."""

    def __init__(self, *, request_id, stream_id, generation_id):
        self.request_id = request_id
        self.stream_id = stream_id
        self.generation_id = generation_id
        self.position = 0
        self._pending_pcm = bytearray()
        self.events = []
        self._cancelled = False

    def capture_state(self):
        return _Checkpoint(
            request_id=self.request_id,
            stream_id=self.stream_id,
            generation_id=self.generation_id,
            position=self.position,
            pending_pcm=bytes(self._pending_pcm),
            events=tuple(self.events),
            cancelled=self._cancelled,
        )

    def restore_state(self, checkpoint):
        if checkpoint.request_id != self.request_id:
            raise ValueError("request owner mismatch")
        if checkpoint.stream_id != self.stream_id:
            raise ValueError("stream owner mismatch")
        if checkpoint.generation_id != self.generation_id:
            raise ValueError("generation owner mismatch")
        self.position = checkpoint.position
        self._pending_pcm = bytearray(checkpoint.pending_pcm)
        self.events = list(checkpoint.events)
        self._cancelled = checkpoint.cancelled

    def process(self, tokens):
        if self._cancelled:
            raise RuntimeError("cancelled stream cannot process")
        for token in tokens:
            self.position += 1
            sample = (int(token) * 31 + self.position * 7) % 32768
            self._pending_pcm.extend(int(sample).to_bytes(2, "little", signed=True))
            self.events.append((self.position, int(token)))

    def commit_pcm(self):
        if self._cancelled:
            return b""
        pcm = bytes(self._pending_pcm)
        self._pending_pcm.clear()
        return pcm

    def cancel(self):
        self._cancelled = True
        self._pending_pcm.clear()
