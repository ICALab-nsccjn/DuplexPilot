# APR Checkpoint Gate v3 Verdict

## Final Verdict

```text
VERDICT: BLOCKED

LEVEL0_STATE_INTEGRITY: PARTIAL/PASS FOR EXPOSED LOCAL STATE
LEVEL1_STREAMING_SEMANTICS: PASS FOR DECODER AND LOCAL/FAKE ADAPTER
LEVEL2_AUDIO_EQUIVALENCE: CANDIDATE CONTRACT DEFINED; REAL API NOT CLOSED
LOCAL_TOKEN2WAV: RNG-AWARE CONTINUATION DIAGNOSTIC PASS
REMOTE_TOKEN2WAV: BLOCKED

APR_INTEGRATION_ALLOWED: NO
APR_RUNTIME: NOT ENABLED
N4/N8: NOT RUN
FORMAL_BENCHMARK: NOT STARTED
```

## Q1 — Is acoustic state migration semantically valid?

Partially. StreamingDecoder and the exposed local Flow/HiFT cache state pass
state/event tests. The new RNG isolation experiment produced bitwise-equal
continuous and restored PCM when the same CPU/CUDA RNG snapshot was restored.
However, the current real checkpoint API does not capture that RNG state, so the
production local contract is incomplete.

## Q2 — What equivalence level is achievable?

The v2 design selects exact Level 0 and Level 1 semantics plus a fixed
signal-level Level 2 contract for stochastic production backends. A deterministic
reference backend is available only as a test-only CPU semantic oracle; the real
CUDA backend cannot claim strict deterministic mode because
`cumsum_cuda_kernel` has no deterministic implementation.

## Q3 — Can local Token2Wav satisfy the contract?

Potentially, with one missing capability: the logical checkpoint must include
CPU/CUDA/explicit-generator RNG continuation state. The controlled diagnostic
passed bitwise equality twice, while an uncontrolled continuation consumed a
different RNG stream. No production state API was changed in this task, so the
current local implementation is not promoted to full PASS.

## Q4 — Can remote Token2Wav satisfy the contract?

Not under the current production boundary. Flow/HiFT caches, pending PCM,
flush/cancel state, and RNG state are not exportable. The service exposes only
process-local stream IDs and start/stream/close/health operations. See
`REMOTE_TOKEN2WAV_GAP_ANALYSIS.md` and
`APR_REMOTE_BACKEND_LIMITATION_REPORT.md`.

## Q5 — Is APR integration allowed?

No. The remote production migration hard gate is not satisfied, and the local
RNG-aware checkpoint capability is diagnostic evidence rather than an implemented
production contract. APR selector/runtime, worker migration, and N4/N8 remain
disabled.

## Decision and next action

Keep APR `BLOCKED`. The next valid work is to implement and validate the
RNG-aware local checkpoint API and a production remote migration API, or to
formally narrow the APR research scope to a backend that exposes those
capabilities. Do not claim cross-stage acoustic virtualization or start APR
performance experiments before a fresh gate passes.
