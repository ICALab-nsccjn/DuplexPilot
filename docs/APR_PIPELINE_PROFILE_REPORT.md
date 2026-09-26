# APR Pipeline Profile Report

This report is populated only from the isolated APR profiling pilot. It is not a
performance aggregate and does not authorize a new workload matrix.

Required evidence per accepted run:

- `pipeline_spans.jsonl` with finite monotonic spans;
- matching profile manifest and workload trace hash;
- canonical PCM/event/cleanup evidence;
- GPU collector status when the run is performance-enabled.

The analyzer keeps unavailable metrics as `NOT_MEASURED`/`null`. It never
converts a missing span or telemetry sample into zero.

Optimization selection is fail-closed: a candidate requires both a wall-time
signal and an independent supporting signal.
