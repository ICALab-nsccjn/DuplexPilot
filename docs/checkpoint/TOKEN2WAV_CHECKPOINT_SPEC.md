# Token2Wav Checkpoint Specification

## Scope and boundary

The local Token2Wav implementation already has a caller-owned continuation state through `create_stream_state()` and `stream_with_state()`. This document defines the explicit checkpoint wrapper around that state. It does not enable APR, change normal stream behavior, or claim that remote state is migratable.

The logical checkpoint covers one `(request_id, stream_id, generation_id)` and the continuation state required to resume the same acoustic stream.

## Required logical state

A checkpoint must represent:

- request, stream, generation, and prompt identity;
- generated-token buffer and flush state owned by the caller;
- local Flow continuation cache;
- local HiFT cache (`mel`, `source`, and `speech` tensors);
- pending PCM records not yet committed to playback;
- queue/commit sequence and cancellation status.

The current `Token2wav` model weights and prompt preprocessing cache are shared immutable/model-local resources. They are not logical stream state. A remote HTTP `stream_id` is an external owner key, not proof that remote state can be exported.

## API shape

The backend-facing wrapper must provide:

```python
class Token2WavCheckpoint:
    @classmethod
    def capture(cls, *, request_id, stream_id, generation_id,
                prompt_wav, stream_state, token_buffer=(),
                pending_pcm=(), cancelled=False): ...
    def validate(self) -> None: ...
    def serialize(self) -> bytes: ...
    @classmethod
    def restore(cls, payload) -> "Token2WavCheckpoint": ...

class Token2WavBackend:
    def capture_state(self, request_id): ...
    def restore_state(self, checkpoint): ...
    def process(self, tokens, *, last_chunk=False): ...
    def commit_output(self): ...
    def cancel(self): ...
```

`process()` must operate on caller-owned state and return/commit output under the same logical identity. Cancellation advances generation or marks the old generation terminal so stale PCM cannot be committed.

## Tensor and serialization rules

For in-process migration, tensor leaves are detached and cloned; restore must not alias the source state. Device and dtype are retained and validated. A portable durable serialization format is intentionally not promised for CUDA tensor caches in this phase; a future implementation may add an explicit device-aware tensor codec.

The remote service may only be marked migratable after it exposes an authenticated export/import operation that transfers the complete flow and HiFT state with identity and cancellation semantics. The existing `stream_id` endpoints alone do not satisfy that contract.

## Equivalence gate

Continuous local chunk processing must match partial processing followed by capture, independent state restore, and continuation in PCM byte sequence, chunk boundaries, stream/generation identity, and cancellation behavior.
