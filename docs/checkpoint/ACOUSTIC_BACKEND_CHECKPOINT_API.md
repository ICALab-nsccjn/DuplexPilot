# Acoustic Backend Checkpoint API

## Purpose

This is the APR-independent contract for checkpointing one logical acoustic stream. It separates logical state from physical decoder/vocoder resources without selecting workers or changing request scheduling.

```python
class AcousticBackendCheckpoint:
    def capture_state(self): ...
    def restore_state(self, state): ...
    def process(self, tokens): ...
    def commit_output(self): ...
    def cancel(self): ...
```

The concrete implementation may use a typed checkpoint object rather than a dictionary, but it must preserve identity, ordering, generation, cancellation, and output ownership. Capture and restore are explicit operations; no caller may inspect private decoder or Token2Wav fields.

## Atomicity rules

1. Capture returns a copy or immutable snapshot; later processing cannot mutate the captured object.
2. Restore validates schema, identity, configuration, and state shape before modifying a backend.
3. Commit exposes only output for the current logical generation.
4. Cancel makes queued/late output from the cancelled generation uncommittable.
5. Queue, worker, socket, and lock objects are resources, not checkpoint data.

## Gate

The checkpoint gate is `PASS` only when both the real StreamingDecoder and the real local Token2Wav state pass continuous-versus-migrated equivalence. A remote backend without complete export/import remains `BLOCKED` and is reported as a backend limitation rather than emulated.
