# DuplexPilot

Public code release accompanying the DuplexPilot project: **Request-Owned Joint State Checkpointing for Planned Cold Resumption in Full-Duplex Speech Serving**.

Repository: <https://github.com/ICALab-nsccjn/DuplexPilot>

This release is organized for reviewer inspection and code-level reproduction. It includes the implementation surface across the model and acoustic execution planes, rather than only the small paper-facing contract layer:

- `src/lychee_fd/`: service entry points, model integration, request ownership, streaming decoder, acoustic checkpointing, APR queues/workers, and runtime state handling;
- `src/model_plane/`: row-aware model execution, model execution tracing, and batch-capacity logic;
- `src/joint_benchmark/`: joint model/acoustic execution contracts and fixed-work orchestration;
- `src/benchmark_harness/`: acoustic lanes, Flow/Token2Wav helpers, online routing, and public benchmark runners;
- `src/checkpoint_lab/`: checkpoint adapters, state equivalence checks, and deterministic reference backends;
- `src/third_party/`: the small set of project-specific vLLM and Step-Audio integration adapters;
- `tests/`: checkpoint, runtime, ownership, model-plane, joint-execution, and integration contract tests;
- `tools/`: analysis, profiling, and validation utilities that operate on user-supplied records;
- `workloads/`: deterministic workload generators;
- `configs/` and `docs/`: execution-envelope manifests and design/checkpoint contracts.

The repository intentionally excludes generated logs, intermediate experiment outputs, archival snapshots, raw run traces, model weights/checkpoints, and machine-specific environment directories. Public datasets are also kept out of Git because the local files exceed GitHub's normal per-file limit; obtain them from their upstream distribution under the applicable license.

## Quick start

For the pure-Python contract checks, use Python 3.10+:

```bash
python -m venv .venv
. .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-public.txt
PYTHONPATH=src pytest tests/checkpoint tests/joint tests/ownership tests/runtime tests/model_plane
```

The integration and model-plane tests that touch PyTorch, audio backends, vLLM, or a GPU require the matching pinned backend environment. They remain in the tree so the execution contracts and expected interfaces are reviewable; they are not replaced by fake results.

The source-only validation used for this release is:

```bash
python scripts/verify_public_tree.py
python -m compileall -q src tests tools workloads scripts
```

## Reproduction scope

The public release exposes the recovery transaction boundaries, request/row ownership invariants, checkpoint serialization and validation code, joint batch contracts, workload generators, and the runners needed to connect those components to a compatible backend. Reported measurements still depend on the pinned model/backend environment and private run records; those records and weights are intentionally not redistributed.
