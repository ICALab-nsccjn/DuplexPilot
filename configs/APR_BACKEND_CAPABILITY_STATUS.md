# APR Backend Capability Status

## Frozen status

This document records the capability closure before any APR runtime
integration. No APR selector, worker migration, scheduler, RSV/DSV, or N4/N8
path is enabled by this work.

| Capability | Status | Evidence |
|---|---|---|
| Checkpoint API shape | VALID | Explicit local/fake and backend-neutral contracts exist. |
| StreamingDecoder migration | PASS | Capture/restore preserves event identity, ordering, flush, and trace state. |
| Local Token2Wav cache restoration | PASS | Recursive Flow/HiFT state values, keys, shapes, devices, and dtypes restore correctly. |
| Real Token2Wav numerical equivalence | UNRESOLVED | Strict deterministic mode is unavailable; current checkpoint omits stochastic RNG continuation state. |
| Deterministic reference semantics | PASS (test-only) | CPU reference backend passes exact checkpoint/output tests; it is not the real acoustic backend. |
| Remote production migration | BLOCKED | The service has no export/import/validate/resume/cancel state API. |

## Current interpretation

The local logical-state envelope is valid for the state currently exposed by
the model. It is not yet a complete continuation envelope because the real
acoustic path consumes global CPU/CUDA RNG state. A diagnostic that restored the
same RNG snapshot obtained bitwise-identical PCM twice; a diagnostic that did
not restore RNG produced different PCM. RNG stream position is therefore part
of the capability requirement, not an optional reproducibility detail.

The CUDA backend also contains a `torch.cumsum` operation whose CUDA kernel is
not registered as deterministic. This prevents a strict deterministic reference
claim, but it does not by itself prove that state migration is semantically
wrong when stochastic state is controlled.

## Remote boundary

`lychee_fd/token2wav_server.py` stores Flow/HiFT state in process-local
`_stream_states`, protects all synthesis with a global `_model_lock`, and
exposes only start/stream/close/health operations. There is no atomic state
export/import or generation-aware cancellation protocol. The mock contract is
useful for API design, but it is not production capability.

## Gate consequence

APR integration remains disallowed. The next closure must provide and validate:

1. an RNG-aware real local checkpoint contract, and
2. a production remote state migration contract, or an explicit decision to
   exclude the remote backend from APR scope.
