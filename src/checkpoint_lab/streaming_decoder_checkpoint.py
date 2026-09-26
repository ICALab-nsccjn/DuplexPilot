"""Explicit, serializable checkpoints for StreamingDecoder logical state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


CHECKPOINT_SCHEMA_VERSION = 1
_VALID_STATES = frozenset({"l", "s", "b"})


def _int_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a tuple/list of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must contain only integers")
    return tuple(int(item) for item in value)


def _nullable_text(name: str, value: object) -> str | None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be null or a non-empty string")
    return value


@dataclass(frozen=True)
class StreamingDecoderCheckpoint:
    """Copy-safe logical state for one decoder stream."""

    schema_version: int
    tts_chunk_size: int
    end_event_on_generation_complete: bool
    text_buf: tuple[int, ...]
    stoken_buf: tuple[int, ...]
    stoken_flush_buf: tuple[int, ...]
    state: str
    event_kind: str | None
    event_id: str | None
    event_counter: int
    text_seq: int
    prev_decoded_len: int
    acoustic_trace_context: Mapping[str, Any]

    @classmethod
    def capture(cls, decoder: object) -> "StreamingDecoderCheckpoint":
        checkpoint = cls(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            tts_chunk_size=int(decoder.tts_chunk_size),
            end_event_on_generation_complete=bool(
                decoder._end_event_on_generation_complete
            ),
            text_buf=tuple(decoder._text_buf),
            stoken_buf=tuple(decoder._stoken_buf),
            stoken_flush_buf=tuple(decoder._stoken_flush_buf),
            state=str(decoder._state),
            event_kind=decoder._event_kind,
            event_id=decoder._event_id,
            event_counter=int(decoder._event_counter),
            text_seq=int(decoder._text_seq),
            prev_decoded_len=int(decoder._prev_decoded_len),
            acoustic_trace_context=dict(decoder._acoustic_trace_context),
        )
        checkpoint.validate(decoder)
        return checkpoint

    @classmethod
    def restore(
        cls, decoder: object, payload: "StreamingDecoderCheckpoint | bytes | bytearray | memoryview"
    ) -> "StreamingDecoderCheckpoint":
        checkpoint = payload if isinstance(payload, cls) else cls.deserialize(payload)
        checkpoint.validate(decoder)
        decoder._apply_checkpoint(checkpoint)
        return checkpoint

    @classmethod
    def deserialize(cls, payload: bytes | bytearray | memoryview) -> "StreamingDecoderCheckpoint":
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("checkpoint payload must be bytes-like")
        try:
            data = json.loads(bytes(payload).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid StreamingDecoder checkpoint JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("checkpoint JSON must contain an object")
        try:
            checkpoint = cls(
                schema_version=data["schema_version"],
                tts_chunk_size=data["tts_chunk_size"],
                end_event_on_generation_complete=data["end_event_on_generation_complete"],
                text_buf=tuple(data["text_buf"]),
                stoken_buf=tuple(data["stoken_buf"]),
                stoken_flush_buf=tuple(data["stoken_flush_buf"]),
                state=data["state"],
                event_kind=data["event_kind"],
                event_id=data["event_id"],
                event_counter=data["event_counter"],
                text_seq=data["text_seq"],
                prev_decoded_len=data["prev_decoded_len"],
                acoustic_trace_context=data["acoustic_trace_context"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint JSON is missing required fields") from exc
        checkpoint.validate()
        return checkpoint

    def validate(self, decoder: object | None = None) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema: {self.schema_version}")
        if isinstance(self.tts_chunk_size, bool) or not isinstance(self.tts_chunk_size, int):
            raise ValueError("tts_chunk_size must be an integer")
        if self.tts_chunk_size <= 0:
            raise ValueError("tts_chunk_size must be positive")
        if not isinstance(self.end_event_on_generation_complete, bool):
            raise ValueError("end_event_on_generation_complete must be boolean")
        _int_tuple("text_buf", self.text_buf)
        _int_tuple("stoken_buf", self.stoken_buf)
        _int_tuple("stoken_flush_buf", self.stoken_flush_buf)
        if self.state not in _VALID_STATES:
            raise ValueError(f"invalid decoder state: {self.state!r}")
        event_kind = _nullable_text("event_kind", self.event_kind)
        event_id = _nullable_text("event_id", self.event_id)
        if (event_kind is None) != (event_id is None):
            raise ValueError("event_kind and event_id must be both null or both set")
        for name, value in (
            ("event_counter", self.event_counter),
            ("text_seq", self.text_seq),
            ("prev_decoded_len", self.prev_decoded_len),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.acoustic_trace_context, Mapping):
            raise ValueError("acoustic_trace_context must be a mapping")
        try:
            json.dumps(dict(self.acoustic_trace_context), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("acoustic_trace_context must be JSON-serializable") from exc
        if decoder is not None:
            if int(decoder.tts_chunk_size) != self.tts_chunk_size:
                raise ValueError("checkpoint tts_chunk_size does not match decoder")
            if bool(decoder._end_event_on_generation_complete) != self.end_event_on_generation_complete:
                raise ValueError("checkpoint generation-complete policy does not match decoder")

    def serialize(self) -> bytes:
        self.validate()
        data = {
            "schema_version": self.schema_version,
            "tts_chunk_size": self.tts_chunk_size,
            "end_event_on_generation_complete": self.end_event_on_generation_complete,
            "text_buf": list(self.text_buf),
            "stoken_buf": list(self.stoken_buf),
            "stoken_flush_buf": list(self.stoken_flush_buf),
            "state": self.state,
            "event_kind": self.event_kind,
            "event_id": self.event_id,
            "event_counter": self.event_counter,
            "text_seq": self.text_seq,
            "prev_decoded_len": self.prev_decoded_len,
            "acoustic_trace_context": dict(self.acoustic_trace_context),
        }
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
