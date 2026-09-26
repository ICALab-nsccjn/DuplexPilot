# Acoustic Equivalence Contract v2

## Scope

This contract defines correctness for moving one logical acoustic stream across
physical resources. It does not enable APR and does not change current serving
semantics.

## Level 0 — State integrity (required)

Capture/restore must preserve and validate:

```text
request_id
stream_id
generation_id
schema/version
prompt and model identity
token position and ordering metadata
StreamingDecoder state
Flow continuation cache
HiFT/vocoder cache
pending PCM records and sequence numbers
flush/cancel state
CPU RNG state
CUDA RNG state for every participating device
```

If a backend uses an explicit `torch.Generator`, its state must be included.
Otherwise the backend must declare that it has no stochastic continuation
state. The snapshot must be copy-safe, device/dtype-aware, and validated before
mutating the target. A missing or mismatched field is a Level 0 failure.

## Level 1 — Streaming semantic equivalence (required)

For identical logical input and owner identity, capture/restore must preserve
exactly:

```text
event ordering and event IDs
flush boundaries
PCM chunk ordering and ownership
generation identity
cancel behavior
stale-output rejection
```

Level 1 is not allowed to use numerical tolerance to hide a missing, duplicated,
reordered, or stale event. StreamingDecoder and local/fake adapter tests already
cover this level; production remote event state remains unexposed.

## Level 2 — Audio equivalence

### Option A: bitwise PCM equality

This is the strongest result and remains required whenever a backend provides a
deterministic reference mode. It is not assumed for the current CUDA HiFiGAN
path because its `cumsum_cuda_kernel` has no deterministic implementation and
the path contains explicit random draws.

### Option B: deterministic reference mode

A backend may provide a deterministic mode with fixed RNG state and supported
deterministic kernels. The reference test must compare continuous and restored
continuations bitwise. The current real backend does not provide this mode.

### Option C: signal-level equivalence for stochastic production backends

This is the selected v2 candidate for a real stochastic backend, but only when
Level 0 includes the complete RNG continuation state and Level 1 passes exactly:

```text
same sample rate
same non-empty PCM sample count
all PCM finite
normalized RMSE <= 0.02
Pearson correlation >= 0.99
SNR >= 34 dB
```

The thresholds are fixed migration-stability criteria, not audio-quality
claims. They must not be retuned to accommodate an observed result. A signal
pass is reported as `REAL_CUDA_MIGRATION_EQUIVALENCE`; bitwise equality remains a
separate diagnostic field.

## Gate interpretation

```text
Level 0 PASS + Level 1 PASS + Level 2 A/B PASS = FULL_EQUIVALENCE
Level 0 PASS + Level 1 PASS + Level 2 C PASS   = STOCHASTIC_EQUIVALENCE
Level 0/1 PASS + missing RNG or failed Level 2 = BLOCKED
```

The remote production API is an additional hard gate. A local or mock pass does
not imply remote migration capability.
