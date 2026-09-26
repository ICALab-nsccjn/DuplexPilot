# APR Gate v3 Summary

## Frozen conclusions

State migration is semantically feasible for the local backend when the logical
checkpoint includes the stochastic continuation state. The A100 real local
Token2Wav gate passed capture → serialize → restore → continue with bitwise
equal PCM under the same captured CPU/CUDA RNG state.

The key conclusions are:

```text
RNG state is part of Acoustic Logical State.
Local Flow/HiFT cache restoration is valid.
StreamingDecoder migration is validated independently.
The current remote service has no production migration capability.
```

## Evidence

`tests/test_local_full_acoustic_checkpoint.py` ran in the pinned model
environment with four tests passing, including the real Token2Wav test. The
strict deterministic warning for `cumsum_cuda_kernel` remains, so the local
pass is based on the declared RNG-aware contract and is stronger than the
minimum signal-level result, not a claim that every CUDA kernel is
deterministic.

The remote service still exposes only start/stream/close/health and retains
continuation data in process-local state. No remote production change was made.

## Scope consequence

Local explicit-state checkpointing is a real capability. Full remote APR serving
is not a capability of the current system. The project should therefore scope
the contribution as an explicit-state virtualization framework with a validated
local backend contract, while treating remote state migration as a separate
future backend extension.
