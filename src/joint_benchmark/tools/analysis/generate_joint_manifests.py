"""Generate immutable source, environment, and dataset manifests for J11--J22.

This utility is intentionally metadata-only.  It is run inside the approved
``duplexpilot`` container after the target worktree and its environment have
been fixed.  It refuses to overwrite an existing artifact so an experiment
attempt can never silently change its provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_once(path: Path, payload: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _source_entries(worktree: Path) -> list[dict[str, Any]]:
    patterns = (
        "lychee_fd/runtime/apr/joint_execution.py",
        "lychee_fd/runtime/apr/joint_execution_trace.py",
        "lychee_fd/runtime/apr/paper_systems.py",
        "lychee_fd/runtime/apr/online_router.py",
        "lychee_fd/runtime/apr/online_step_coordinator.py",
        "lychee_fd/runtime/model_batch_config.py",
        "lychee_fd/runtime/row_aware_model_execution_plane.py",
        "lychee_fd/vllm_integration/engine.py",
        "lychee_fd/app.py",
        "tools/benchmarks/realtime_public_client.py",
        "tools/benchmarks/apr_public_online_e2e.py",
        "tools/apr/flow_batch_acoustic_lane.py",
    )
    entries: list[dict[str, Any]] = []
    for relative in patterns:
        path = worktree / relative
        entries.append(
            {
                "relative_path": relative,
                "absolute_path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() and path.is_file() else None,
                "sha256": _sha256(path),
            }
        )
    return entries


def _trace_entries(paths: list[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(paths):
        result.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else None,
                "sha256": _sha256(path),
            }
        )
    return result


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--image", default="unknown")
    parser.add_argument("--image-digest", default="unknown")
    parser.add_argument("--disk", default="unknown")
    parser.add_argument("--gpu", default="unknown")
    parser.add_argument("--python-torch-cuda", default="unknown")
    parser.add_argument("--model-path", default="/mnt/DuplexPilot/data/models/lychee_full_duplex")
    parser.add_argument("--token2wav-path", default="/mnt/DuplexPilot/data/models/token2wav")
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    tag_object = _git(worktree, "rev-parse", "safe-variable-length-flow-b2-mixed-step-v1")
    peeled = _git(worktree, "rev-parse", "safe-variable-length-flow-b2-mixed-step-v1^{}").lower()
    head = _git(worktree, "rev-parse", "HEAD").lower()
    branch = _git(worktree, "branch", "--show-current")
    status = _git(worktree, "status", "--short")

    trace_root = Path("/mnt/DuplexPilot/data/DuplexPilot/results/online_n_batch_formation_closure/traces_v2")
    trace_paths = [
        path
        for dataset in ("HD-Balanced", "HD-LongTail", "HD-Burst")
        for n in (8, 16)
        for path in (trace_root / dataset / f"N{n}" / "window-0.jsonl",)
    ]
    multidataset_path = Path(
        "/mnt/DuplexPilot/data/DuplexPilot/results/published_baseline_multidataset_20260829/traces_v4/MULTI_DATASET_TRACE_MANIFEST.json"
    )
    multidataset = _load_json(multidataset_path) if multidataset_path.exists() else None
    if isinstance(multidataset, dict):
        for entry in multidataset.get("traces", []):
            for key in ("canonical_path", "public_path"):
                value = entry.get(key)
                if value:
                    trace_paths.append(Path(str(value)))

    source_manifest = {
        "artifact": "APR_JOINT_B22_SOURCE_MANIFEST",
        "schema": "apr-joint-b22-source-v1",
        "created_epoch_s": time.time(),
        "base_tag": "safe-variable-length-flow-b2-mixed-step-v1",
        "tag_object": tag_object,
        "base_commit_peeled": peeled,
        "expected_base_commit": "fef6f2be025b0a76338255d0f009a7fe8f978d03",
        "branch": branch,
        "head": head,
        "worktree": str(worktree),
        "worktree_status": status or "clean",
        "audited_migrations": [
            "a1e58be",
            "92fe03a",
            "735a8ad",
            "cb9b7b6",
        ],
        "audited_mixed_padding_source": {
            "commit": "0a15ea2",
            "reason": "frozen mixed_step_chunk_padding public path was absent from peeled base",
        },
        "source_sha256": _source_entries(worktree),
        "environment_only_artifacts": [
            {
                "path": str(worktree / "third_party/vllm/vllm/vllm_flash_attn"),
                "reason": "approved ABI extension symlink; not source and not committed",
            },
            {"path": "**/__pycache__", "reason": "generated cache"},
        ],
        "fixed_joint_cells": {
            "J11": [1, 1, "legacy_serialized", "apr_step_b1"],
            "J21": [2, 1, "row_aware_cap2", "apr_step_b1"],
            "J12": [1, 2, "legacy_serialized", "mixed_chunk_padding_b2"],
            "J22": [2, 2, "row_aware_cap2", "mixed_chunk_padding_b2"],
        },
        "no_repeat_scope": [
            "historical N=1/2/4/8/16 correctness matrices",
            "historical B=2/B=4 and mixed-step/chunk-padding mechanisms",
            "CUDA Graph/Inductor/operator/capacity/staging experiments",
            "external baseline and dataset acquisition audit",
        ],
    }
    environment_manifest = {
        "artifact": "APR_JOINT_B22_ENVIRONMENT_MANIFEST",
        "schema": "apr-joint-b22-environment-v1",
        "created_epoch_s": time.time(),
        "container": {"name": "duplexpilot", "image": args.image, "digest": args.image_digest},
        "disk": args.disk,
        "gpu": args.gpu,
        "python_torch_cuda": args.python_torch_cuda,
        "fixed_environment": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "LYCHEEFD_TOKEN2WAV_DEVICE": "1",
            "LYCHEEFD_REQUIRE_DUAL_GPU_PLACEMENT": "1",
            "LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY": "2048",
            "LYCHEEFD_MAX_FLOW_BATCH_SIZE": "2",
            "LYCHEEFD_ROW_AWARE_MODEL_MAX_BATCH_SIZE": "1 or 2, per joint cell",
            "flow_dtype": "float32",
            "flow_steps": 10,
            "physical_acoustic_workers": 2,
            "gpu1_primary_envelope_bytes": 34359738368,
            "gpu1_diagnostic_hard_stop_bytes": 40802189312,
        },
        "placement": "GPU0=model/vLLM; GPU1=Local Token2Wav/Flow",
        "model_path": args.model_path,
        "token2wav_path": args.token2wav_path,
        "target_worktree": str(worktree),
    }
    dataset_manifest = {
        "artifact": "APR_JOINT_B22_DATASET_MANIFEST",
        "schema": "apr-joint-b22-dataset-v1",
        "created_epoch_s": time.time(),
        "primary_online_datasets": [
            "HumDial-FDBench/HD-Balanced",
            "HumDial-FDBench/HD-LongTail",
            "HumDial-FDBench/HD-Burst",
        ],
        "secondary_online_dataset": "FD-Bench-Audio-Input",
        "limited_behavior_dataset": "Full-Duplex-Bench-v1.5",
        "burstgpt_role": "arrival-pattern source only; not an independent audio dataset",
        "canonical_trace_manifest": str(multidataset_path),
        "canonical_trace_manifest_sha256": _sha256(multidataset_path),
        "canonical_trace_manifest_snapshot": multidataset,
        "trace_inputs": _trace_entries(trace_paths),
        "rules": [
            "preserve original arrival/audio/pause/overlap/interruption timing",
            "no artificial barrier, synchronized injection, or per-system retiming",
            "formal runs use real sleep; --no-sleep is debug-only",
            "v1.5 remains limited behavior evidence if PCM lifecycle is incomplete",
            "discovery and held-out paths are fixed before inspecting results",
        ],
    }
    no_repeat = "# APR Joint B22 No-Repeat Ledger\n\n"
    no_repeat += "This ledger records evidence reused without rerunning the historical full matrices.\n\n"
    no_repeat += "| Evidence | Status | Reused from |\n|---|---|---|\n"
    for item in source_manifest["no_repeat_scope"]:
        no_repeat += f"| {item} | REUSED_VALIDATED_RESULT | prior frozen reports/tags |\n"
    no_repeat += "\nNew delta work in this branch:\n\n"
    no_repeat += "- JointExecutionSpec and J11/J21/J12/J22 payload contracts.\n"
    no_repeat += "- Unified model/acoustic event correlation and work fingerprints.\n"
    no_repeat += "- Four-cell fixed-work and live multi-dataset online measurements.\n"
    no_repeat += "- Conditional one-boundary mixed-padding operator only if measured overhead warrants it.\n"
    no_repeat += "\nA failed or invalid attempt is retained and is never silently replaced.\n"
    freeze = "# APR B>1 Mechanism Freeze V3\n\n"
    freeze += "The B>1 acoustic mechanism is frozen at `safe-variable-length-flow-b2-mixed-step-v1` / `fef6f2b`.\n\n"
    freeze += "The joint branch may select the audited `mixed_step_chunk_padding` public path for B_acoustic=2, but it does not alter step-index relaxation, state identity, cache semantics, checkpoint semantics, ownership, PCM, cancel, flush, or client protocol.\n\n"
    freeze += "J11/J21/J12/J22 are benchmark-only combinations; local model and acoustic speedups are measured separately and are not multiplied.\n"

    _write_once(result_root / "APR_JOINT_B22_SOURCE_MANIFEST.json", json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n")
    _write_once(result_root / "APR_JOINT_B22_ENVIRONMENT_MANIFEST.json", json.dumps(environment_manifest, indent=2, ensure_ascii=False) + "\n")
    _write_once(result_root / "APR_JOINT_B22_DATASET_MANIFEST.json", json.dumps(dataset_manifest, indent=2, ensure_ascii=False) + "\n")
    _write_once(result_root / "APR_JOINT_B22_NO_REPEAT_LEDGER.md", no_repeat)
    _write_once(result_root / "APR_B2_MECHANISM_FREEZE_V3.md", freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
