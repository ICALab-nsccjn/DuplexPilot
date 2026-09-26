# APR Scope Decision

## Decision

```text
SELECTED: OPTION 2 — EXPLICIT-STATE APR FRAMEWORK
```

## Evidence

```text
Local full acoustic checkpoint: PASS
StreamingDecoder checkpoint: PASS
RNG-aware local Token2Wav continuation: PASS
Remote production state migration: NOT AVAILABLE
```

The local backend now has an explicit logical state contract that can preserve
cache state, streaming state, pending output, identity, and stochastic RNG
continuation. The current remote service has none of the required production
export/import/validate/resume/cancel semantics. Supporting it would require a
new remote serving mechanism owned by the service.

## Research scope

The contribution should be framed as:

```text
state-contract-driven virtualization for stateful neural serving
```

The validated scope includes explicit-state local backends and the abstraction
boundary needed to virtualize logical request/acoustic state. It must not claim
that the current remote Token2Wav service supports acoustic worker migration.

## Why not Option 1

`FULL APR` requires both local and remote migration capability. Remote migration
is not present and was not implemented in this task. Starting APR runtime or N4/N8
would turn an unimplemented production boundary into an unsupported claim.

## Why not Option 3

State migration is not impossible in general: the local real backend passed the
RNG-aware full gate, and the remote service could in principle add a service-
owned checkpoint API. Therefore APR is narrowed, not dropped.

## Next allowed phase

The next phase may refine the explicit-state backend interface and evaluate its
local mechanism in a separately pre-registered experiment. It must not enable
APR selector/worker migration or start N4/N8 until a new scope-specific protocol
and gate are approved.
