"""APR-independent composite checkpoint for one local acoustic stream."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .token2wav_checkpoint import clone_state_value


CHECKPOINT_SCHEMA_VERSION = 1


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _clone_mapping(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return clone_state_value(dict(value))


@dataclass(frozen=True)
class AcousticCheckpointState:
    """Copy-safe composite logical state, without worker-selection semantics."""

    schema_version: int
    request_id: str
    generation_id: int
    stream_id: str
    version: int
    decoder_state: Mapping[str, Any]
    token2wav_state: Mapping[str, Any]
    token_buffer: tuple[int, ...]
    flush_state: Mapping[str, Any]
    pending_output: tuple[bytes, ...]
    cpu_rng_state: Any
    cuda_rng_state: tuple[Any, ...]
    explicit_generator_state: Mapping[str, Any]
    cancelled: bool
    pcm_seq: int = 0

    @classmethod
    def capture(
        cls,
        *,
        request_id: str,
        generation_id: int,
        stream_id: str,
        version: int,
        decoder_state: Mapping[str, Any],
        token2wav_state: Mapping[str, Any],
        token_buffer: tuple[int, ...] | list[int],
        flush_state: Mapping[str, Any],
        pending_output: tuple[bytes, ...] | list[bytes],
        cpu_rng_state: Any,
        cuda_rng_state: tuple[Any, ...] | list[Any],
        explicit_generator_state: Mapping[str, Any],
        cancelled: bool = False,
        pcm_seq: int = 0,
    ) -> "AcousticCheckpointState":
        checkpoint = cls(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            request_id=_require_text("request_id", request_id),
            generation_id=generation_id,
            stream_id=_require_text("stream_id", stream_id),
            version=version,
            decoder_state=_clone_mapping("decoder_state", decoder_state),
            token2wav_state=_clone_mapping("token2wav_state", token2wav_state),
            token_buffer=tuple(token_buffer),
            flush_state=_clone_mapping("flush_state", flush_state),
            pending_output=tuple(bytes(item) for item in pending_output),
            cpu_rng_state=clone_state_value(cpu_rng_state),
            cuda_rng_state=tuple(clone_state_value(item) for item in cuda_rng_state),
            explicit_generator_state=_clone_mapping(
                "explicit_generator_state", explicit_generator_state
            ),
            cancelled=cancelled,
            pcm_seq=pcm_seq,
        )
        checkpoint.validate()
        return checkpoint

    @staticmethod
    def capture_current_rng_state() -> dict[str, Any]:
        """Capture RNG state without synchronizing a model or CUDA stream."""
        return {
            "cpu": torch.get_rng_state().clone(),
            "cuda": tuple(state.clone() for state in torch.cuda.get_rng_state_all()),
        }

    def validate(
        self,
        *,
        request_id: str | None = None,
        generation_id: int | None = None,
        stream_id: str | None = None,
        version: int | None = None,
    ) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported acoustic checkpoint schema: {self.schema_version}"
            )
        _require_text("request_id", self.request_id)
        _require_text("stream_id", self.stream_id)
        if isinstance(self.generation_id, bool) or not isinstance(self.generation_id, int):
            raise ValueError("generation_id must be an integer")
        if self.generation_id < 0:
            raise ValueError("generation_id must be non-negative")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("version must be an integer")
        if self.version < 0:
            raise ValueError("version must be non-negative")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in self.token_buffer):
            raise ValueError("token_buffer must contain integers")
        if any(not isinstance(item, bytes) for item in self.pending_output):
            raise ValueError("pending_output must contain bytes")
        if not isinstance(self.decoder_state, Mapping):
            raise ValueError("decoder_state must be a mapping")
        if not isinstance(self.token2wav_state, Mapping):
            raise ValueError("token2wav_state must be a mapping")
        if not isinstance(self.flush_state, Mapping):
            raise ValueError("flush_state must be a mapping")
        if self.cpu_rng_state is None:
            raise ValueError("cpu_rng_state is required")
        if not isinstance(self.cuda_rng_state, tuple):
            raise ValueError("cuda_rng_state must be a tuple")
        if not isinstance(self.explicit_generator_state, Mapping):
            raise ValueError("explicit_generator_state must be a mapping")
        if not isinstance(self.cancelled, bool):
            raise ValueError("cancelled must be boolean")
        if isinstance(self.pcm_seq, bool) or not isinstance(self.pcm_seq, int):
            raise ValueError("pcm_seq must be an integer")
        if self.pcm_seq < 0:
            raise ValueError("pcm_seq must be non-negative")
        if request_id is not None and self.request_id != request_id:
            raise ValueError("checkpoint request_id does not match target")
        if generation_id is not None and self.generation_id != generation_id:
            raise ValueError("checkpoint generation_id does not match target")
        if stream_id is not None and self.stream_id != stream_id:
            raise ValueError("checkpoint stream_id does not match target")
        if version is not None and self.version != version:
            raise ValueError("checkpoint version does not match target")

    def restore_rng_state(self) -> None:
        """Restore captured RNG streams without synchronizing model execution."""
        torch.set_rng_state(clone_state_value(self.cpu_rng_state))
        if self.cuda_rng_state:
            torch.cuda.set_rng_state_all(
                [clone_state_value(state) for state in self.cuda_rng_state]
            )

    def serialize(self) -> bytes:
        """Serialize a tensor-safe local envelope; remote transport is separate."""
        self.validate()
        payload = {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "generation_id": self.generation_id,
            "stream_id": self.stream_id,
            "version": self.version,
            "decoder_state": clone_state_value(dict(self.decoder_state)),
            "token2wav_state": clone_state_value(dict(self.token2wav_state)),
            "token_buffer": tuple(self.token_buffer),
            "flush_state": clone_state_value(dict(self.flush_state)),
            "pending_output": tuple(self.pending_output),
            "cpu_rng_state": clone_state_value(self.cpu_rng_state),
            "cuda_rng_state": tuple(clone_state_value(item) for item in self.cuda_rng_state),
            "explicit_generator_state": clone_state_value(
                dict(self.explicit_generator_state)
            ),
            "cancelled": self.cancelled,
            "pcm_seq": self.pcm_seq,
        }
        stream = io.BytesIO()
        torch.save(payload, stream)
        return stream.getvalue()

    @classmethod
    def restore(
        cls,
        payload: bytes | bytearray | memoryview,
        *,
        map_location: object | None = None,
    ) -> "AcousticCheckpointState":
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("acoustic checkpoint payload must be bytes-like")
        try:
            data = torch.load(
                io.BytesIO(bytes(payload)),
                map_location=map_location,
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError("invalid acoustic checkpoint payload") from exc
        if not isinstance(data, dict):
            raise ValueError("acoustic checkpoint payload must contain a mapping")
        try:
            checkpoint = cls(
                schema_version=data["schema_version"],
                request_id=data["request_id"],
                generation_id=data["generation_id"],
                stream_id=data["stream_id"],
                version=data["version"],
                decoder_state=data["decoder_state"],
                token2wav_state=data["token2wav_state"],
                token_buffer=tuple(data["token_buffer"]),
                flush_state=data["flush_state"],
                pending_output=tuple(data["pending_output"]),
                cpu_rng_state=data["cpu_rng_state"],
                cuda_rng_state=tuple(data["cuda_rng_state"]),
                explicit_generator_state=data["explicit_generator_state"],
                cancelled=data["cancelled"],
                pcm_seq=data["pcm_seq"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("acoustic checkpoint is missing required fields") from exc
        checkpoint.validate()
        return checkpoint
