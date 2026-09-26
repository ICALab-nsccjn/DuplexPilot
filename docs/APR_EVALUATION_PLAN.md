# APR Evaluation Plan

## 1. Purpose

This is a diagnostic validation plan for the Acoustic Progression Runtime. It
does not reopen the frozen full-paper protocol and does not permit a throughput,
speedup, latency, or utilization claim.

The question is whether APR removes the observed post-extraction acoustic
progression ceiling while preserving request ownership, PCM semantics, and the
existing measurement boundary.

## 2. Registered Configurations

Baseline and prototype:

```text
B5-disabled: existing Dynamic-Virtualized runtime
B5+APR-P1: APR with one physical acoustic worker slot
B5+APR-P2: APR with two physical acoustic worker slots
```

Workload and concurrency:

```text
W1 Natural Full-Duplex
N=2, N=4, N=8
```

The initial diagnostic count is:

```text
3 runtime variants × 3 concurrency levels × 3 valid repeats
= 27 counted diagnostic runs
```

The frozen B5-disabled N=2 control is rerun in the same session setup. Prior
B5 N=3/N=4/N=8 failures remain historical boundary evidence and are not merged
into APR aggregates.

## 3. Entry Gates

Before N=4 or N=8:

1. B5-disabled W1 N=2 is 3/3 clean;
2. B5+APR-P1 W1 N=2 is 3/3 clean;
3. B5+APR-P2 W1 N=2 is 3/3 clean;
4. ownership mismatch is zero in all controls;
5. event schema and `FIRST_PLAYABLE_AUDIO` semantics are unchanged;
6. APR queue and state-store unit tests pass inside `duplexpilot`.

If an entry gate fails, the higher-concurrency cells do not run.

## 4. Success Gates

The minimum prototype gate is:

```text
B5 + APR-P1, W1, N=4: 5/5 clean diagnostic attempts
B5 + APR-P2, W1, N=4: 5/5 clean diagnostic attempts
B5 + APR-P2, W1, N=8: 3/3 clean diagnostic attempts
```

P1 is a decoupling control. P2 is the first capacity candidate. A clean run
requires all sessions to produce valid PCM, observe done, complete cleanup,
and pass ownership/measurement validation.

The N=8 gate is diagnostic only. Passing it does not establish an N=8 paper
performance claim.

## 5. Frozen Conditions

The following must match the B5 boundary diagnosis:

- W1 input asset and chunk cadence;
- Bmax=2;
- model checkpoint and runtime commit;
- `LYCHEEFD_VLLM_MAX_NUM_SEQS=2`;
- Token2Wav configuration and GPU mapping;
- warmup and timeout rules;
- canonical event schema;
- monotonic clock;
- `INPUT_ARRIVAL`, `FIRST_PLAYABLE_AUDIO`, and `SESSION_FINISH` definitions;
- fail-closed invalid-run handling.

APR worker count is the only registered independent variable.

## 6. Required Metrics

Existing metrics remain available:

```text
session completion
useful playable PCM duration
INPUT_ARRIVAL -> FIRST_PLAYABLE_AUDIO
inter-audio gap P50/P95/P99/Max
GPU0/GPU1 utilization and memory
physical batch size and effective batch size
ownership and cleanup errors
```

APR metrics:

```text
acoustic queue depth: mean/P50/P95/P99/max
token waiting time: enqueue -> dequeue
acoustic scheduling delay: ready -> worker start
acoustic processing time: worker start -> worker end
PCM generation delay: token enqueue -> PCM commit
state restore time
state commit time
backpressure count and duration
stale-generation output count
worker-slot occupancy
```

Each metric is keyed by request ID and generation ID. Pooled distributions are
supplementary; primary run-level aggregation remains the registered harness
aggregation.

## 7. Invalid Runs

An attempt is invalid and excluded if any of the following occurs:

- missing done or cleanup;
- missing `FIRST_PLAYABLE_AUDIO` for a required completed session;
- non-monotonic timestamp;
- missing APR queue/state/PCM event;
- request-ID, generation-ID, sequence, or state-version mismatch;
- stale PCM committed to playback;
- silent queue drop;
- worker crash or unhandled state-restore exception;
- ownership regression;
- invalid PCM;
- GPU telemetry or measurement schema failure.

Invalid attempts are retained with failure signature and retried only under the
registered protocol. Repeated systematic failure pauses the cell.

## 8. Causal Analysis

The analysis extends the original chain:

```text
state divergence
  -> exact-compatible opportunity loss
  -> RSV batch restoration
  -> effective model batch
  -> APR acoustic scheduling
  -> PCM availability
  -> completed full-duplex sessions
```

For each N and worker count, report:

1. B5-disabled versus APR physical batch and effective batch;
2. APR queue depth and token waiting time;
3. whether P2 token extraction events reach P3 for every request;
4. whether P3-to-PCM delay decreases or merely moves into the queue;
5. whether clean completion improves without ownership errors;
6. whether the existing realtime tail remains interpretable.

APR is not credited with a causal gain if it only increases queue depth or
changes completion accounting without increasing valid PCM availability.

## 9. Interpretation Matrix

| Observation | Interpretation |
|---|---|
| P1 separates model from acoustic work but N=4 still fails | Decoupling alone is insufficient; physical acoustic capacity remains limiting |
| P2 reaches clean N=4/N=8 with zero ownership errors | H2 receives diagnostic support |
| queue grows without PCM improvement | Acoustic backend is the remaining capacity boundary |
| P3 coverage improves but P99 audio gap worsens sharply | APR has a realtime tradeoff; do not claim success |
| state-version or ownership mismatch | Core correctness regression; stop immediately |
| N=2 APR control changes PCM/measurement semantics | Formal measurement regression; stop immediately |

## 10. Deliverables

```text
apr_raw_runs.csv
apr_primary_metrics.csv
apr_realtime_metrics.csv
apr_queue_metrics.csv
apr_state_metrics.csv
apr_gpu_metrics.csv
apr_invalid_attempts.csv
APR_DIAGNOSTIC_REPORT.md
```

Raw evidence for each attempt includes manifest, canonical events, APR trace,
GPU samples, server/client logs, and sanity output.

## 11. Stop Rule

Do not tune APR based on an unattractive graph. Stop only for correctness,
measurement, systematic runtime, hardware, or environment comparability
failure. If the target gates fail, report the exact remaining boundary and do
not enter the full paper benchmark.
