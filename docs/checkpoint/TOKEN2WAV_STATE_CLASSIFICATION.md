# Token2Wav State Classification

This classification is based on the current `third_party/Step-Audio2/token2wav.py`, `lychee_fd/token2wav_server.py`, and `lychee_fd/app.py` state ownership.

| State | Classification | Reason |
|---|---|---|
| Local Flow cache | MIGRATABLE | `stream_with_state()` already accepts and returns caller-owned cache; capture requires detached tensor copies. |
| Local HiFT `mel/source/speech` cache | MIGRATABLE | It is held in caller-owned `hift_cache` and can be cloned with tensor metadata. |
| Prompt preprocessing cache | RECONSTRUCTABLE | It is keyed by prompt and recomputed/reused from immutable model resources; it is not continuation state. |
| Token buffer | MIGRATABLE | Caller-owned buffer can be copied with generation identity. |
| Flush state | MIGRATABLE | Explicit wrapper state can preserve pending flush/last-chunk semantics. |
| Pending PCM | MIGRATABLE | Only if held before commit and tagged with request/stream/generation/sequence. |
| Local queue state | RECONSTRUCTABLE | Queue contents can be drained into an explicit ordered pending-output list; live thread/lock objects are not migrated. |
| Local model weights | NOT_MIGRATABLE | Shared runtime resource, not logical request state. |
| Remote `stream_id` | NOT_MIGRATABLE | It identifies server-side state but the current API has no complete export/import operation. |
| Remote Flow/HiFT cache | NOT_MIGRATABLE | It is held inside the remote process and is not serializable through current endpoints. |
| Remote pending output | NOT_MIGRATABLE | Current service does not expose an atomic pending-output snapshot. |
| Worker thread/lock | NOT_MIGRATABLE | Execution primitives cannot be transferred as logical state. |

## Gate consequence

The local backend can proceed to a real checkpoint prototype after equivalence tests pass. The remote backend remains blocked unless an explicit state transfer API is added and independently validated; normal stream endpoints must not be treated as migration support.
