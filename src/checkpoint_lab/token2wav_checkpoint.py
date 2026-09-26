"""Explicit checkpoints for caller-owned local Token2Wav stream state."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .acoustic_backend_checkpoint import AcousticBackendCheckpoint


CHECKPOINT_SCHEMA_VERSION = 1


def clone_state_value(value: Any) -> Any:
    """Clone nested stream state without aliasing tensor or mutable leaves."""
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_value(item) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return copy.deepcopy(value)


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class Token2WavCheckpoint:
    """Copy-safe logical state for a local Token2Wav stream."""

    schema_version: int
    request_id: str
    stream_id: str
    generation_id: int
    prompt_wav: str
    stream_state: Mapping[str, Any]
    token_buffer: tuple[int, ...]
    pending_pcm: tuple[bytes, ...]
    pcm_seq: int
    cancelled: bool

    @classmethod
    def capture(
        cls,
        *,
        request_id: str,
        stream_id: str,
        generation_id: int,
        prompt_wav: str,
        stream_state: Mapping[str, Any],
        token_buffer: tuple[int, ...] | list[int] = (),
        pending_pcm: tuple[bytes, ...] | list[bytes] = (),
        pcm_seq: int = 0,
        cancelled: bool = False,
    ) -> "Token2WavCheckpoint":
        checkpoint = cls(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            request_id=_require_text("request_id", request_id),
            stream_id=_require_text("stream_id", stream_id),
            generation_id=generation_id,
            prompt_wav=_require_text("prompt_wav", prompt_wav),
            stream_state=clone_state_value(dict(stream_state)),
            token_buffer=tuple(token_buffer),
            pending_pcm=tuple(bytes(item) for item in pending_pcm),
            pcm_seq=pcm_seq,
            cancelled=bool(cancelled),
        )
        checkpoint.validate()
        return checkpoint

    def validate(self, *, request_id: str | None = None, stream_id: str | None = None) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported Token2Wav checkpoint schema: {self.schema_version}")
        _require_text("request_id", self.request_id)
        _require_text("stream_id", self.stream_id)
        _require_text("prompt_wav", self.prompt_wav)
        if isinstance(self.generation_id, bool) or not isinstance(self.generation_id, int) or self.generation_id < 0:
            raise ValueError("generation_id must be a non-negative integer")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in self.token_buffer):
            raise ValueError("token_buffer must contain integers")
        if any(not isinstance(item, bytes) or not item for item in self.pending_pcm):
            raise ValueError("pending_pcm must contain non-empty bytes")
        if isinstance(self.pcm_seq, bool) or not isinstance(self.pcm_seq, int) or self.pcm_seq < 0:
            raise ValueError("pcm_seq must be a non-negative integer")
        if not isinstance(self.cancelled, bool):
            raise ValueError("cancelled must be boolean")
        if not isinstance(self.stream_state, Mapping):
            raise ValueError("stream_state must be a mapping")
        if request_id is not None and self.request_id != request_id:
            raise ValueError("checkpoint request_id does not match target")
        if stream_id is not None and self.stream_id != stream_id:
            raise ValueError("checkpoint stream_id does not match target")

    def restore(self, *, request_id: str | None = None, stream_id: str | None = None) -> dict[str, Any]:
        """Return a detached caller-owned stream state for continuation."""
        self.validate(request_id=request_id, stream_id=stream_id)
        return clone_state_value(dict(self.stream_state))


class LocalToken2WavCheckpointAdapter(AcousticBackendCheckpoint):
    """Small explicit adapter around Token2wav's caller-owned state API.

    This adapter is intentionally disconnected from APR selection and worker
    migration. It only makes capture/restore/process/commit/cancel semantics
    testable for one logical stream.
    """

    def __init__(self, model, *, request_id: str, stream_id: str, generation_id: int, prompt_wav: str, stream_state: Mapping[str, Any]):
        self.model = model
        self.request_id = _require_text("request_id", request_id)
        self.stream_id = _require_text("stream_id", stream_id)
        self.generation_id = generation_id
        self.prompt_wav = _require_text("prompt_wav", prompt_wav)
        self.stream_state = clone_state_value(dict(stream_state))
        self.pending_pcm: list[bytes] = []
        self.pcm_seq = 0
        self.cancelled = False

    def capture_state(self) -> Token2WavCheckpoint:
        return Token2WavCheckpoint.capture(
            request_id=self.request_id,
            stream_id=self.stream_id,
            generation_id=self.generation_id,
            prompt_wav=self.prompt_wav,
            stream_state=self.stream_state,
            pending_pcm=self.pending_pcm,
            pcm_seq=self.pcm_seq,
            cancelled=self.cancelled,
        )

    def capture_acoustic_state(
        self,
        *,
        decoder_state: Mapping[str, Any],
        token_buffer: tuple[int, ...] | list[int] = (),
        flush_state: Mapping[str, Any] | None = None,
        cpu_rng_state: Any | None = None,
        cuda_rng_state: tuple[Any, ...] | list[Any] | None = None,
        explicit_generator_state: Mapping[str, Any] | None = None,
    ):
        """Capture the full local logical acoustic state without APR semantics."""
        from .acoustic_checkpoint_state import AcousticCheckpointState

        current_rng = AcousticCheckpointState.capture_current_rng_state()
        return AcousticCheckpointState.capture(
            request_id=self.request_id,
            generation_id=self.generation_id,
            stream_id=self.stream_id,
            version=1,
            decoder_state=decoder_state,
            token2wav_state={
                "prompt_wav": self.prompt_wav,
                "stream_state": self.stream_state,
            },
            token_buffer=token_buffer,
            flush_state=flush_state or {},
            pending_output=self.pending_pcm,
            cpu_rng_state=current_rng["cpu"] if cpu_rng_state is None else cpu_rng_state,
            cuda_rng_state=(
                current_rng["cuda"] if cuda_rng_state is None else cuda_rng_state
            ),
            explicit_generator_state=(
                {}
                if explicit_generator_state is None
                else explicit_generator_state
            ),
            cancelled=self.cancelled,
            pcm_seq=self.pcm_seq,
        )

    def restore_acoustic_state(self, checkpoint) -> dict[str, Any]:
        """Restore local acoustic state and return decoder/flush state to caller."""
        from .acoustic_checkpoint_state import AcousticCheckpointState

        if not isinstance(checkpoint, AcousticCheckpointState):
            raise TypeError("checkpoint must be an AcousticCheckpointState")
        checkpoint.validate(request_id=self.request_id, stream_id=self.stream_id)
        state_payload = checkpoint.token2wav_state
        if state_payload.get("prompt_wav") != self.prompt_wav:
            raise ValueError("checkpoint prompt_wav does not match target")
        stream_state = state_payload.get("stream_state")
        if not isinstance(stream_state, Mapping):
            raise ValueError("checkpoint stream_state is missing")
        self.generation_id = checkpoint.generation_id
        self.stream_state = clone_state_value(dict(stream_state))
        self.pending_pcm = list(checkpoint.pending_output)
        self.pcm_seq = checkpoint.pcm_seq
        self.cancelled = checkpoint.cancelled
        checkpoint.restore_rng_state()
        return {
            "decoder_state": clone_state_value(dict(checkpoint.decoder_state)),
            "token_buffer": tuple(checkpoint.token_buffer),
            "flush_state": clone_state_value(dict(checkpoint.flush_state)),
        }

    def restore_state(self, checkpoint: Token2WavCheckpoint) -> None:
        checkpoint.validate(request_id=self.request_id, stream_id=self.stream_id)
        self.generation_id = checkpoint.generation_id
        self.prompt_wav = checkpoint.prompt_wav
        self.stream_state = checkpoint.restore(request_id=self.request_id, stream_id=self.stream_id)
        self.pending_pcm = list(checkpoint.pending_pcm)
        self.pcm_seq = checkpoint.pcm_seq
        self.cancelled = checkpoint.cancelled

    def process(self, tokens, *, last_chunk: bool = False) -> None:
        if self.cancelled:
            raise RuntimeError("cancelled Token2Wav stream cannot process")
        pcm = self.model.stream_with_state(
            list(tokens),
            self.prompt_wav,
            self.stream_state,
            last_chunk=bool(last_chunk),
        )
        if pcm:
            self.pending_pcm.append(bytes(pcm))

    def commit_output(self) -> tuple[bytes, ...]:
        if self.cancelled:
            self.pending_pcm.clear()
            return ()
        committed = tuple(self.pending_pcm)
        self.pending_pcm.clear()
        self.pcm_seq += len(committed)
        return committed

    def cancel(self) -> None:
        self.cancelled = True
        self.pending_pcm.clear()
