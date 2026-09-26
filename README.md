# DuplexPilot

Public code release accompanying the DuplexPilot paper: **Request-Owned Joint State Checkpointing for Planned Cold Resumption in Full-Duplex Speech Serving**.

Repository: <https://github.com/ICALab-nsccjn/DuplexPilot>

This repository contains the reviewable implementation surface rather than the private experiment archive. It includes:

- `src/lychee_fd/`: the public service/runtime integration, request ownership, acoustic checkpoint, token-to-wave checkpoint, streaming decoder, APR queue/state/worker contracts, and joint execution tracing;
- `src/duplexpilot/`: the public joint execution and serving orchestration modules used by the controlled recovery path;
- `src/tools/`: analysis and workload utilities that operate on user-supplied records;
- `tests/`: checkpoint, ownership, APR, and joint-execution contract tests;
- `workloads/`: deterministic workload generators;
- `paper/`: the English and Chinese LaTeX sources, figure sources/assets, bibliography, and built PDFs;
- `docs/`: design and checkpoint-contract documentation.

The repository intentionally excludes generated logs, intermediate experiment outputs, archival snapshots, raw run traces, model weights/checkpoints, and machine-specific environment directories. Public datasets are also kept out of the Git history because the source files are larger than GitHub's normal per-file limit; obtain them from their upstream distribution and follow the corresponding dataset license.

## Quick start

The contract and pure-Python tests can be run from the repository root with Python 3.10+:

```bash
python -m venv .venv
. .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -U pytest numpy
PYTHONPATH=src pytest tests/checkpoint tests/joint tests/ownership tests/runtime
```

Some tests exercise optional PyTorch, audio, or GPU integrations. Those tests require the matching backend environment and are not silently replaced by mocks in the release.

The paper source can be built from `paper/` with the local TeX Live installation:

```bash
cd paper
bash build.sh
```

The figures and tables in `paper/` are the publication assets. The scripts do not download private checkpoints or publish experiment artifacts.

## Reproducibility scope

The public release documents the state contracts, ownership invariants, recovery transaction boundaries, and the commands needed to exercise the deterministic contract tests. The reported measurements depend on the pinned model/backend environment and private run records; those records and weights are intentionally not redistributed.
