#!/usr/bin/env python3
"""Generate auditable source/environment manifests for the capacity phase."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any


ROOT = Path(os.environ.get(
    "DUPLEXPILOT_RESULT_ROOT",
    "/mnt/DuplexPilot/data/DuplexPilot/results/apr_contention_zero_copy_slo_20260830",
))
WORKTREE = Path(os.environ.get(
    "DUPLEXPILOT_WORKTREE",
    "/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-contention-zero-copy-slo",
))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def source_manifest() -> dict[str, Any]:
    relevant = [
        "lychee_fd/runtime/apr/worker_assignment_policy.py",
        "lychee_fd/runtime/apr/elastic_worker_pool.py",
        "lychee_fd/runtime/apr/worker_handoff.py",
        "lychee_fd/runtime/apr/online_router.py",
        "lychee_fd/runtime/apr/paper_systems.py",
        "tools/apr/capacity_slo_lanes.py",
        "tools/apr/build_corrected_capacity_calibration.py",
        "tools/apr/analyze_corrected_capacity_calibration.py",
        "tools/apr/run_zero_copy_handoff_benchmark.py",
        "tools/apr/run_capacity_slo_online_case.sh",
        "tools/benchmarks/paper_backend_selector.py",
    ]
    files: dict[str, Any] = {}
    for relative in relevant:
        path = WORKTREE / relative
        if path.is_file():
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        else:
            files[relative] = {"missing": True}
    untracked = command(["git", "-C", str(WORKTREE), "status", "--short"])
    payload = {
        "manifest_version": 2,
        "generated_epoch_s": time.time(),
        "base_tag": "apr-capacity-slo-state-v2-final",
        "base_commit": "a6750d9c4556e38d46862c8195e2518854c984d7",
        "branch": command(["git", "-C", str(WORKTREE), "branch", "--show-current"]),
        "commit": command(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"]),
        "source_files": files,
        "worktree_status": untracked,
        "ignored_runtime_artifacts": {
            "vllm_C_sha256": "35566116acd4ba3d6ef8e0fbb711cb34be1b4a845e6291985e5b89e99bab58eb",
            "vllm_flash_attn_sha256": "ce054b692341acea33f1aef258640c43419c6c3a14f9043c2c4f3c776f1cdf92",
            "note": "Copied from the previously working audited worktree; not tracked by git.",
        },
        "frozen_contracts": [
            "APR state/checkpoint schema v2",
            "RSV/DSV semantics",
            "Flow numerical formula",
            "request ownership/generation fencing",
            "PCM/cancel/flush/client protocol",
            "GPU placement",
        ],
        "reused_evidence": [
            "Phase 1-4 correctness",
            "B=2/B=4 mechanism and online evidence",
            "mixed-step/variable-length/chunk-padding",
            "CUDA Graph fixed-work",
            "preemption",
            "capacity/SLO v2 mechanism gate",
        ],
        "benchmark_commands": [
            "tools/apr/run_capacity_slo_online_case.sh",
            "tools/apr/run_zero_copy_handoff_benchmark.py",
        ],
        "result_directory": str(ROOT),
    }
    return payload


def environment_manifest() -> dict[str, Any]:
    torch_version = "UNAVAILABLE"
    cuda_version = "UNAVAILABLE"
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception as exc:
        torch_version = f"UNAVAILABLE: {exc}"
    return {
        "manifest_version": 2,
        "generated_epoch_s": time.time(),
        "container": "duplexpilot",
        "image": "triepilot-a100:20260720-ubuntu2404-lightweight",
        "image_digest": "sha256:464c407c55b7cc95a13e1b81fc85aec52d35a5ccdf614de97f24c54119956808",
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "torch": torch_version,
        "cuda_runtime": cuda_version,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        "device_order": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        "gpu0_role": "model/vLLM",
        "gpu1_role": "Local Token2Wav/Flow",
        "token2wav_device": os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE", "1"),
        "require_dual_gpu_placement": os.environ.get("LYCHEEFD_REQUIRE_DUAL_GPU_PLACEMENT", "1"),
        "physical_acoustic_workers": 2,
        "flow_attention_cache_capacity": 2048,
        "gpu1_conservative_envelope_gib": 32,
        "model_path": "/mnt/DuplexPilot/data/models/lychee_full_duplex",
        "token2wav_path": "/mnt/DuplexPilot/data/models/token2wav",
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
        "disk": command(["df", "-h", "/mnt/DuplexPilot"]),
        "notes": [
            "All tests and model execution run inside duplexpilot.",
            "GPU0 model KV is not migrated.",
            "Zero-copy transfers same-GPU logical state lease; no tensor/CPU copy.",
            "Shared execution lock means this phase does not claim parallel acoustic throughput.",
        ],
    }


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "APR_CAPACITY_SLO_CORRECTED_SOURCE_MANIFEST.json").write_text(
        json.dumps(source_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "APR_CAPACITY_SLO_CORRECTED_ENVIRONMENT_MANIFEST.json").write_text(
        json.dumps(environment_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"root": str(ROOT), "commit": source_manifest()["commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
