# StreamingDecoder Checkpoint Specification

## Scope

This contract makes the mutable logical state of one `StreamingDecoder` stream explicit. It is a prerequisite for a real APR adapter; it is not APR runtime integration and it does not change serving, scheduling, sampler, RSV, DSV, or Token2Wav semantics.

The checkpoint boundary is after row-aware token extraction and before the next call to `StreamingDecoder.feed`. The tokenizer and framework are immutable dependencies of the decoder and are not serialized as part of logical state.

## Public API

The implementation must expose a public value object and decoder methods:

```python
class StreamingDecoderCheckpoint:
    @classmethod
    def capture(cls, decoder) -> "StreamingDecoderCheckpoint": ...

    def validate(self, decoder=None) -> None: ...
    def serialize(self) -> bytes: ...

    @classmethod
    def restore(cls, decoder, payload) -> "StreamingDecoderCheckpoint": ...

class StreamingDecoder:
    def capture_checkpoint(self) -> StreamingDecoderCheckpoint: ...
    def restore_checkpoint(self, checkpoint_or_bytes) -> None: ...
```

`capture()` copies state. `restore()` validates the complete snapshot before mutating the target decoder and then restores by value. APR and other callers must never use `decoder.__dict__` or access `_xxx` fields directly.

## Required captured state

The snapshot contains the following logical fields:

| Field | Type | Ownership rule |
|---|---|---|
| `_text_buf` | tuple of int | copied in capture and copied again on restore |
| `_stoken_buf` | tuple of int | copied in capture and copied again on restore |
| `_stoken_flush_buf` | tuple of int | copied in capture and copied again on restore |
| `_state` | `l`, `s`, or `b` | validated enum |
| `_event_kind` | nullable string | restored with event identity |
| `_event_id` | nullable string | restored exactly, not regenerated |
| `_event_counter` | non-negative int | preserves future event IDs |
| `_text_seq` | non-negative int | preserves text delta sequence |
| `_prev_decoded_len` | non-negative int | preserves delta boundary |
| `_acoustic_trace_context` | JSON object | copied and JSON-validated |

The checkpoint also carries a schema version and decoder configuration needed to reject incompatible targets: `tts_chunk_size` and `end_event_on_generation_complete`. Tokenizer/framework objects, model weights, threads, locks, open sockets, and external service state are not checkpoint payloads.

## Serialization and validation

Serialization is deterministic JSON bytes with sorted keys and no executable or pickle content. Tuples are represented as JSON arrays and restored as tuples. Validation rejects unknown schema versions, invalid token types, invalid state values, negative counters, inconsistent nullable event identity, non-JSON trace context, and decoder configuration mismatches.

Restore is fail-closed and atomic from the caller's perspective: all checks run before any target field is changed. A failed restore leaves the target decoder unchanged.

## Equivalence gate

For the same event sequence, continuous feeding and partial feed followed by capture, decoder destruction, restore, and continuation must produce identical:

- event type and order;
- event IDs and event counter progression;
- text snapshots and deltas;
- audio-token flush boundaries and values;
- final event state;
- acoustic trace request/generation identity.

No APR performance or N4/N8 claim is permitted from this unit-level gate.
