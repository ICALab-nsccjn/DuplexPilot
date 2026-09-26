# APR Scheduler Analysis

The profiling path measures `APR_SCHEDULE_WAIT` around the existing
`next_ready()` boundary and records queue depth, request identity, and worker
slot. It does not change ready-request ordering, admission, preemption,
request-yield, or worker allocation semantics.

Scheduler optimization remains unselected until queue-wait/residency imbalance
and an independent idle-worker/GPU signal are both present in fresh profile
runs.
