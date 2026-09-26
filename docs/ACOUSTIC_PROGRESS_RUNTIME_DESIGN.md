# Acoustic Progression Runtime Design

## Status

This document is an architecture proposal. It does not modify the frozen
RSV/DSV implementation and does not authorize performance experiments.

The proposed component is the **Acoustic Progression Runtime (APR)**. Its
purpose is to separate model execution progress from acoustic output progress
after row-aware token extraction.

## 1. Motivation and Evidence

The current B5 runtime is stable at N=2 but not practically servable at N>=3.
The diagnostic boundary is:

```text
row-aware token extraction (P2)
        -> missing for some requests
StreamingDecoder.feed (P3)
```

Physical batch rotation and ownership remain correct. The limitation is in the
shared downstream progression path, not in RSV/DSV state ownership.

## 2. Frozen Invariants

APR must not change:

- logical request state abstraction;
- physical execution row mapping;
- RSV or DSV semantics;
- `can_pack` and batch selection;
- Bmax policy;
- request ownership and request-ID lookup;
- model logits, sampling, or token semantics.

APR adds a new boundary after token extraction. It must preserve the request ID
on every token batch, acoustic state transition, PCM chunk, cancellation, and
error.

## 3. Current Pipeline

```text
Scheduler
  -> ModelRunner
  -> row-aware forward
  -> row-aware token extraction / sampler result
  -> StreamingDecoder.feed
  -> Token2Wav
  -> PCM event queue
  -> logical playback buffer
```

The current synchronous/serialized characteristics are:

1. model execution and generator progression share the vLLM adapter runtime;
2. the adapter protects the non-thread-safe engine/forward context with a
   shared `_stream_lock`;
3. acoustic decoding is entered directly from the inference generator;
4. Token2Wav state and PCM emission are downstream of that direct call path;
5. one request's acoustic progress can therefore occupy the same progression
   path needed by other live requests.

The ownership requirement is that a token batch extracted for request `R` may
only be consumed by `R`'s acoustic state and may only produce PCM owned by `R`.

## 4. Proposed Boundary

```text
Model execution / sampler
        |
        | AcousticTokenBatch(request_id, sequence, tokens, state_version)
        v
Per-request acoustic ingress queue
        |
        v
APR scheduler / bounded worker pool
        |
        v
StreamingDecoder + Token2Wav adapter
        |
        v
PCM commit -> measurement-side playback buffer
```

The model path performs a bounded enqueue and returns to the serving scheduler.
The acoustic path consumes independently and commits output asynchronously.
The enqueue is not allowed to silently drop audio tokens. If admission or
backpressure fails, APR emits an explicit fail-closed error for the owning
request.

## 5. Logical/Physical State Model

APR extends the existing virtualization principle:

```text
Logical Request State
        != Physical Execution Row

Logical Acoustic State
        != Physical Acoustic Progression Slot
```

Each logical request owns an immutable identity and a mutable versioned
acoustic state. A physical worker slot temporarily materializes that state,
advances it, and commits the next version. A worker must never retain a live
request state after the commit or cancellation boundary.

The minimum state record is:

```yaml
request_id: string
stream_id: string
state_version: integer
decoder_state: opaque serialized decoder state
token2wav_state: opaque serialized or checkpointable state
pending_token_count: integer
last_committed_pcm_seq: integer
generation_id: integer
cancelled: boolean
```

Opaque state is an interface contract. The first implementation must prove
that the existing decoder/Token2Wav state can be captured and restored without
changing token interpretation or PCM bytes.

## 6. Recommended Approach

The recommended design is a hybrid of the requested Options A, B, and C:

- **Option A:** per-request ingress queues provide isolation and bounded
  backpressure;
- **Option B:** a bounded acoustic worker pool provides independent progression
  from model execution;
- **Option C:** versioned logical acoustic state permits a request to move
  between worker slots without transferring ownership.

The first prototype uses two physical acoustic slots because the frozen
baseline already uses Bmax=2 and the diagnostic environment has one dedicated
Token2Wav GPU. The pool size is a configuration parameter, but the initial
experiment must compare only the registered values `P=1` and `P=2`.

`P=1` is a decoupling control: it proves that model progress no longer waits
on direct acoustic execution but is not expected to remove the capacity limit.
`P=2` is the first scalability candidate. It must fail closed if the underlying
Token2Wav implementation cannot provide isolated or checkpointable state.

## 7. Non-Goals

APR does not initially:

- change model batching or scheduler selection;
- increase Bmax;
- add a new admission policy;
- change sampling or interruption semantics;
- hide acoustic backpressure by dropping tokens;
- claim N=4/N=8 performance before clean diagnostic completion;
- replace the official native baseline.

## 8. Research Hypotheses

```yaml
H1:
  RSV restores model execution batching opportunity.
H2:
  Acoustic progression virtualization removes the downstream concurrency
  ceiling caused by direct serialized acoustic progression.
H3:
  RSV/DSV plus APR enable scalable full-duplex serving without cross-request
  state contamination or unacceptable realtime tail degradation.
```

H2 is falsified if APR produces token queues but N=4/N=8 still has the same
P2-to-P3 completion ceiling and no measurable reduction in acoustic scheduling
delay.

## 9. Correctness Contract

For every `AcousticTokenBatch`:

1. `request_id` exists in the live request registry;
2. `state_version` is exactly the next accepted version;
3. tokens are consumed once and in sequence order;
4. cancellation invalidates all older generations;
5. PCM carries the same request ID, stream ID, generation, and sequence;
6. a stale worker result is discarded and reported, never committed to another
   request;
7. queue overflow, worker failure, and state-restore failure are explicit
   request-scoped errors;
8. cleanup waits for the request queue and worker lease to reach a terminal
   state.

## 10. Success Boundary

APR is not considered a success because P2 events become asynchronous. The
minimum diagnostic success is:

```text
B5 + APR, W1, N=4: 5/5 clean attempts
B5 + APR, W1, N=8: at least 3/5 clean attempts
ownership mismatch: 0
state restore/commit errors: 0
measurement contract: unchanged
```

Only after this gate may APR be evaluated for E2E throughput or realtime SLO.

## 11. Architecture Decision

Implement the APR prototype as a post-extraction runtime boundary with
versioned logical acoustic state and a bounded worker pool. Keep all changes on
an isolated prototype branch. The frozen B5 baseline and all prior reports
remain immutable reference evidence.
