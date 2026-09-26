# APR Backend Limitation Report

## Status

```text
APR: BLOCKED
LOCAL_STATE_INTEGRITY: PASS
REAL_NUMERICAL_EQUIVALENCE: BLOCKED/FAIL
REMOTE_STATE_MIGRATION: BLOCKED
```

The APR abstraction and fake/local checkpoint contracts are testable, but the
current real acoustic backend does not expose a complete migration-equivalence
capability.

## Limitation 1: CUDA numerical reference

The strict deterministic reference is unavailable because the real HiFiGAN path
uses `cumsum_cuda_kernel`, which has no deterministic implementation in the
frozen CUDA/PyTorch environment. Under the pre-registered signal-level
criterion, the 50-token continuation also failed (`normalized RMSE=0.135442`,
correlation `0.012183`, SNR `-3.081 dB`).

This is not fixed by changing tolerances or by adding synchronization. It
requires a backend-level deterministic/reference mode or a new, explicitly
validated equivalence semantics.

## Limitation 2: Remote state API

The current remote service keeps continuation state in process-local storage and
does not provide atomic `export_state`, `import_state`, `validate_state`,
`resume_stream`, or `cancel_stream` operations. A mock contract exists and is
tested, but production support is absent.

## Scope consequence

Do not enable APR selector/runtime, modify RSV/DSV, alter scheduler or sampler
semantics, or run N4/N8. The safe next decision is whether to implement a real
backend checkpoint/equivalence capability, narrow APR to a backend with an
explicit state contract, or revise the research claim. Until then the APR
checkpoint gate remains `BLOCKED`.
