# APR Optimization Baseline

This document freezes the pre-optimization evidence. It is a diagnostic reference and does not establish an APR speedup claim.

## Scope

- Source report: `reports/apr_workload_suite/APR_WORKLOAD_ADVANTAGE_REPORT.md`
- Systems: `apr, original_affinity`
- Concurrency: `4, 8, 16`
- Valid counted runs: `135 / 135`
- Invalid counted runs: `0`
- Existing performance advantage status: `NOT_ESTABLISHED`.

## Frozen environment

- source_commit: `"be0dadcd217a756d8e0cf1ca8ca1fd2b8869f8dc"`
- model_checkpoint: `"/mnt/DuplexPilot/data/models/lychee_full_duplex"`
- token2wav_checkpoint: `"/mnt/DuplexPilot/data/models/token2wav"`
- gpu_mapping: `{"local_token2wav": "GPU1", "model_execution": "GPU0", "visible_devices": "0,1"}`
- worker_count: `2`
- measurement_schema_version: `"apr-real-e2e-v1"`

## Baseline handling

The historical workload-suite rows remain separate from all future profiling and optimization rows. Missing measurements remain `NOT_MEASURED`; they are never converted to zero.

No code or serving semantics are changed by this baseline artifact.
