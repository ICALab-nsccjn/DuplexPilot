# APR Workload Suite Design

## Scope

This suite characterizes where explicit-state acoustic virtualization can help. It does not change APR, RSV/DSV, the model runner, Token2Wav, or the existing `APR_E2E_PERFORMANCE_REPORT.md`. Existing balanced W1 results remain historical evidence; this suite adds workload strata with explicit arrival and lifetime structure.

The suite has three workload families and one frozen comparison matrix:

| Workload | Purpose | Main stressor |
|---|---|---|
| A: public full-duplex interaction trace | realism validation | overlapping turns, interruption, resume, multi-turn state |
| B: production-like voice serving | main evaluation | heavy-tailed lifetimes and Poisson/burst arrivals |
| C: resource-fragmentation stress | mechanism validation | long resident sessions followed by late short sessions |

The supported matrix is `A/B/C × N=4/8/16 × 5 repeats` for `original_affinity`, `apr`, `apr_no_migration`, and `no_rsv_dsv_apr`. Workload generation is deterministic from `(workload, concurrency, seed, arrival_process)`; random seeds are recorded in every trace manifest.

## Canonical event contract

Every session is represented by a monotonic sequence of these event types:

```text
arrival -> input_phase -> output_phase -> [barge_in -> resume] -> finish
```

An event contains `session_id`, `timestamp_s`, `event_type`, `input_phase`, `output_phase`, and a small JSON-compatible payload. Timestamps are relative to the experiment start. Events from different sessions are globally sorted by `(timestamp_s, event_priority, session_id)` so the same trace produces the same admission and lifecycle order on every system.

The contract separates workload description from runtime implementation. A runtime may consume the trace online, but it must preserve session identity, event order, phase labels, barge-in/resume markers, and finish boundaries. The generator never creates PCM or model tokens.

## Workload A: public full-duplex interaction trace

### Data source

The public-facing input is a normalized JSON/JSONL record rather than a bundled copyrighted audio file. A record supplies a session identifier and timestamped interaction labels (`input`, `output`, `barge_in`, `resume`, `finish`). The repository includes a deterministic public-inspired fixture generated from this normalized schema. It is labeled public-inspired until a specific public corpus and license are registered; it must not be described as a reproduction of an unnamed dataset.

### Conversion

The conversion adapter maps source labels to the canonical event contract, normalizes timestamps to seconds from session arrival, preserves turn boundaries, and inserts an explicit `output_phase` event when a source record has an output span but no separate start marker. It rejects duplicate event ids, negative relative timestamps, and a `resume` without a preceding `barge_in`.

### Trace generation

The fixture cycles through interaction templates so that one trace contains turn overlap, at least one active interruption/resume pair, and multiple turns per session. Increasing N repeats templates with deterministic session ids while retaining overlap. No arrival or duration is hand-tuned per baseline.

## Workload B: production-like voice serving

Session durations use three classes:

| Class | Relative duration | Role |
|---|---:|---|
| short | 1.0x | brief request |
| medium | 3.0x | normal conversation |
| long | 8.0x | extended duplex session |

The class assignment deliberately forms a heavy tail: 25% short, 50% medium, and 25% long. The generator supports:

* `poisson`: deterministic pseudo-random exponential inter-arrivals using the supplied seed;
* `burst`: groups of sessions arrive in short bursts separated by an idle gap.

The default is `poisson`; the arrival process is stored in trace metadata and never inferred from results.

## Workload C: resource-fragmentation stress

The first resident group consists of long sessions that arrive at time zero and remain active for most of the trace. The remaining sessions are short and arrive only after the long group is resident. This creates a controlled opportunity for a scheduler that can move logical acoustic state between physical workers. The generated trace records `long_session_ids` and `late_short_session_ids` so the condition is auditable rather than inferred from aggregate duration.

## Baselines and ablations

The frozen system names are:

```text
original_affinity
apr
apr_no_migration
no_rsv_dsv_apr
```

`original_affinity` is the fixed-worker control. `apr` enables the existing explicit-state APR. `apr_no_migration` keeps the APR queue/scheduler shape but forbids checkpoint migration. `no_rsv_dsv_apr` is a diagnostic ablation that removes RSV/DSV while retaining the acoustic layer; it is not a replacement for the official baseline and must be reported separately.

## Metrics

Primary metrics remain throughput, TTFA, and completion latency. The suite adds:

```text
worker_utilization_variance
idle_worker_time_s
waiting_sessions
migration_count
checkpoint_overhead_s
```

If a metric is not instrumented by a system, its value is `NA`, never zero. A zero is valid only when the runtime explicitly reports that no migration or checkpoint occurred. This prevents an uninstrumented control from appearing better.

## Matrix and decision questions

For each workload family and each N in `{4, 8, 16}`, run five independent repetitions for each system. A run is counted only if all sessions complete, PCM is non-empty and owned, timestamps are monotonic, cleanup succeeds, and the workload trace hash in the manifest matches the trace used by the runner.

The report must answer:

1. Does APR improve the balanced/public workload, or is the prior no-gain result reproduced?
2. Does APR improve heterogeneous lifetimes and burst arrivals?
3. Which combination of duration variance, burstiness, and worker fragmentation predicts a benefit?

The final claim is restricted to the measured advantage region. A positive result in Workload C alone does not justify a general full-duplex speedup claim.
