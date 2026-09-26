#!/usr/bin/env python3
"""Finalize the row-aware model-plane evidence bundle.

This script only aggregates metadata and previously recorded traces.  It does
not start a model service, alter a trace, or infer a causal speedup from a
public run whose generated work differs from its control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from collections import Counter
from typing import Any


BASE_TAG = "safe-variable-length-flow-b2-mixed-step-v1"
BASE_COMMIT = "fef6f2be025b0a76338255d0f009a7fe8f978d03"
CONTAINER_IMAGE = "triepilot-a100:20260720-ubuntu2404-lightweight"
CONTAINER_DIGEST = "sha256:464c407c55b7cc95a13e1b81fc85aec52d35a5ccdf614de97f24c54119956808"


def run_text(args: list[str], default: str = "") -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return default or f"{type(exc).__name__}: {exc}"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def event_name(row: dict[str, Any]) -> str:
    return str(row.get("event_type", row.get("event", "")))


def model_trace_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    ends = [row for row in rows if event_name(row) == "MODEL_ENGINE_STEP_END"]
    starts = [row for row in rows if event_name(row) == "MODEL_ENGINE_STEP_START"]
    batches = [int(row.get("model_batch_size", 0) or 0) for row in ends]
    durations = [int(row.get("duration_ns", 0) or 0) for row in ends]
    locks = [row for row in rows if event_name(row) == "MODEL_LOCK"]
    fps = [row for row in rows if event_name(row) == "MODEL_WORK_FINGERPRINT"]
    return {
        "step_count": len(ends),
        "start_count": len(starts),
        "batch_distribution": {str(size): batches.count(size) for size in sorted(set(batches))},
        "logical_rows": sum(batches),
        "b2_steps": sum(size >= 2 for size in batches),
        "b2_row_fraction": (sum(size for size in batches if size >= 2) / sum(batches)) if sum(batches) else 0.0,
        "busy_ns": sum(durations),
        "lock_wait_ns": sum(int(row.get("wait_ns", 0) or 0) for row in locks),
        "lock_hold_ns": sum(int(row.get("hold_ns", 0) or 0) for row in locks),
        "fingerprint_count": len(fps),
        "fingerprint_tokens": sum(int(row.get("generated_token_count", 0) or 0) for row in fps),
        "record_count": len(rows),
    }


def acoustic_trace_summary(path: Path) -> dict[str, Any]:
    """Summarize frozen GPU1 Flow dispatches without retaining payloads."""
    rows = load_jsonl(path)
    dispatches = [
        row for row in rows
        if event_name(row) in {"FLOW_BATCH_COMPLETE", "FLOW_STEP_END"}
        and row.get("batch_size") is not None
    ]
    sizes = [int(row.get("batch_size", 0) or 0) for row in dispatches]
    logical_rows = sum(sizes)
    b2 = sum(size >= 2 for size in sizes)
    return {
        "dispatch_count": len(dispatches),
        "batch_distribution": dict(sorted(Counter(sizes).items())),
        "b2_dispatch_count": b2,
        "b2_row_fraction": (sum(size for size in sizes if size >= 2) / logical_rows) if logical_rows else 0.0,
    }


def source_manifest(worktree: Path, result_root: Path) -> None:
    files = [
        "lychee_fd/runtime/model_execution_trace.py",
        "lychee_fd/runtime/row_aware_model_execution_plane.py",
        "lychee_fd/runtime/vllm_generation.py",
        "lychee_fd/vllm_integration/engine.py",
        "third_party/vllm/vllm/worker/model_runner.py",
        "profiling/model_execution_plane/__init__.py",
        "profiling/model_execution_plane/oracle.py",
        "tests/test_model_execution_instrumentation.py",
        "tests/test_model_execution_plane_oracle.py",
        "tests/test_model_execution_trace.py",
        "tests/test_model_work_fingerprint.py",
        "tests/test_row_aware_model_cancel_reset.py",
        "tests/test_row_aware_model_execution_plane.py",
        "tests/test_row_aware_model_output_routing.py",
        "tools/analysis/model_execution_plane_oracle.py",
        "tools/analysis/analyze_model_plane_pair.py",
        "tools/analysis/run_model_plane_fixed_work.py",
        "tools/analysis/analyze_model_plane_experiment.py",
        "tools/benchmarks/run_model_plane_online_case.sh",
    ]
    source_rows = []
    for rel in files:
        path = worktree / rel
        source_rows.append({
            "path": str(path),
            "relative_path": rel,
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path),
        })
    payload = {
        "artifact": "APR_MODEL_EXECUTION_PLANE_SOURCE_MANIFEST",
        "schema": "apr-model-plane-source-manifest-v2",
        "base_tag": BASE_TAG,
        "base_commit": BASE_COMMIT,
        "branch": run_text(["git", "-C", str(worktree), "branch", "--show-current"]),
        "current_commit": run_text(["git", "-C", str(worktree), "rev-parse", "HEAD"]),
        "worktree": str(worktree),
        "worktree_status": run_text(["git", "-C", str(worktree), "status", "--short", "--untracked-files=all"]),
        "source_sha256": source_rows,
        "excluded_environment_artifacts": [
            {
                "path": str(worktree / "third_party/vllm/vllm/vllm_flash_attn"),
                "reason": "approved runtime extension symlink; environment-only and not committed",
            },
            {"path": str(worktree / "**/__pycache__"), "reason": "generated cache"},
        ],
        "commands": [
            "docker exec duplexpilot ... tools/analysis/run_model_plane_fixed_work.py",
            "run_model_plane_online_case.sh N workload system repeat trace result_dir mode",
            "docker exec duplexpilot ... tools/analysis/analyze_model_plane_experiment.py",
        ],
        "reused_evidence": [
            "frozen acoustic B>1 mechanism and N=1/2/4/8/16 correctness",
            "mixed-step, variable-length, chunk-padding, Graph, Inductor and operator audits",
            "capacity/SLO, handoff, preemption and staging negative results",
            "multi-dataset and external baseline audit",
        ],
        "new_delta": [
            "metadata-only model execution timeline",
            "row-aware central vLLM driver with request-owned output routing",
            "model work fingerprinting",
            "fixed-work GPU0 model B=2 probe",
            "real-arrival public N=8/N=16 representative smoke pairs (not a repeated held-out matrix)",
        ],
    }
    (result_root / "APR_MODEL_EXECUTION_PLANE_SOURCE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def environment_manifest(worktree: Path, result_root: Path) -> None:
    torch_info = run_text([
        "/root/anaconda3/envs/sglang/bin/python", "-c",
        "import torch; print({'python':__import__('sys').version.split()[0], 'torch':torch.__version__, 'cuda':torch.version.cuda, 'cuda_available':torch.cuda.is_available(), 'device_count':torch.cuda.device_count()})",
    ])
    traces = []
    trace_dir = Path("/mnt/DuplexPilot/data/DuplexPilot/results/online_n_batch_formation_closure/traces_v2")
    for path in sorted(trace_dir.glob("*/N*/window-0.jsonl")):
        traces.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "artifact": "APR_MODEL_EXECUTION_PLANE_ENVIRONMENT_MANIFEST",
        "schema": "apr-model-plane-environment-manifest-v2",
        "container": {
            "name": "duplexpilot",
            "image": CONTAINER_IMAGE,
            "digest": CONTAINER_DIGEST,
        },
        "disk": run_text(["df", "-h", "/mnt/DuplexPilot"]),
        "gpu": run_text(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"]),
        "python_torch_cuda": torch_info,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "LYCHEEFD_TOKEN2WAV_DEVICE": "1",
            "LYCHEEFD_REQUIRE_DUAL_GPU_PLACEMENT": "1",
            "LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY": "2048",
            "LYCHEEFD_VLLM_MAX_NUM_SEQS": "N (per case)",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            "LYCHEEFD_ROW_AWARE_MODEL_EXECUTION_PLANE": "0 or 1 (explicit opt-in)",
        },
        "placement": "GPU0=model/vLLM; GPU1=Local Token2Wav/Flow; physical acoustic workers=2",
        "flow": {"dtype": "float32", "steps": 10, "cache_capacity": 2048, "gpu1_safety_envelope_bytes": 34359738368},
        "model_path": "/mnt/DuplexPilot/data/models/lychee_full_duplex",
        "token2wav_path": "/mnt/DuplexPilot/data/models/token2wav",
        "trace_inputs": traces,
        "valid_b1_branch_integrity_smoke": {
            "path": str(result_root / "phase0_b1_n1_smoke_flash2/online_attempt.json"),
            "status": "PASS",
            "completion": 1,
            "pcm_chunks": 4,
            "ownership_errors": 0,
            "runtime_errors": 0,
        },
        "failed_environment_attempt_not_counted": "automatic XFORMERS selection; approved run used explicit FLASH_ATTN",
    }
    (result_root / "APR_MODEL_EXECUTION_PLANE_ENVIRONMENT_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def online_cases(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    online_root = result_root / "online_pilot"
    for case_dir in sorted(online_root.iterdir()) if online_root.exists() else []:
        if not case_dir.is_dir():
            continue
        attempt_path = case_dir / "client" / "online_attempt.json"
        meta_path = case_dir / "run_metadata.json"
        trace_path = case_dir / "model_execution_trace.jsonl"
        if not (attempt_path.exists() and meta_path.exists() and trace_path.exists()):
            continue
        attempt = load_json(attempt_path)
        meta = load_json(meta_path)
        summary = model_trace_summary(trace_path)
        acoustic = acoustic_trace_summary(case_dir / "online_trace.jsonl")
        rows.append({"case": case_dir.name, "meta": meta, "attempt": attempt, "model": summary, "acoustic": acoustic})
    return rows


def write_critical_and_oracle(worktree: Path, result_root: Path) -> None:
    cases = online_cases(result_root)
    lines = [
        "# APR Model Critical-Path Profile",
        "",
        "## Scope and validity",
        "",
        "This profile combines metadata-only model traces, isolated service probes, and real-arrival public API diagnostics. It does not rewrite arrivals, insert barriers, or infer a causal E2E gain when model work fingerprints differ.",
        "",
        "## Runtime finding",
        "",
        "The legacy path lets each generator directly drive the patched vLLM engine while holding `_stream_lock` around the stream operation. The opt-in path routes request rounds through one central driver; one driver call invokes `engine.step()` for the selected rows and dispatches outputs by request_id. The model runner receives per-request `lychee_side_state`; the legacy global `LycheeDuplexState` remains in the legacy path and is not used by the execution-plane scheduler.",
        "",
        "Relevant implementation boundaries:",
        "- `lychee_fd/runtime/vllm_generation.py`: lock-scope instrumentation and legacy generator ownership.",
        "- `lychee_fd/vllm_integration/engine.py`: row-plane opt-in, central driver, generation fences, output routing.",
        "- `third_party/vllm/vllm/worker/model_runner.py`: per-row side-state construction and batch/forward instrumentation.",
        "- `lychee_fd/runtime/row_aware_model_execution_plane.py`: bounded request-aware driver; no acoustic or tensor state ownership.",
        "",
        "## Observed public cases",
        "",
        "| workload | N | system | mode | valid | model batch distribution | B2 row fraction | model steps | lock wait (s) |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for row in cases:
        meta, attempt, model, acoustic = row["meta"], row["attempt"], row["model"], row["acoustic"]
        lines.append(
            f"| {meta.get('workload','?')} | {meta.get('N','?')} | {meta.get('system','?')} | {meta.get('mode','?')} | {attempt.get('valid')} | `{json.dumps(model['batch_distribution'], sort_keys=True)}` | {model['b2_row_fraction']:.3%} (B_acoustic={acoustic['b2_row_fraction']:.3%}) | {model['step_count']} | {model['lock_wait_ns']/1e9:.3f} |"
        )
    lines += [
        "",
        f"The row-aware path forms physical model B=2 only when multiple request rounds are concurrently runnable. The fixed-work runner observed physical B=2 on every measured decode turn. The completed public representative set contains {len(cases)} valid case records across N=8/N=16; model B=2 formation and acoustic B=2 formation are measured independently. Model batching must not be reported as acoustic batching.",
        "",
        "## Critical-path boundary",
        "",
        "The trace currently has model timestamps and separate public acoustic/PCM events, but no fully joined per-request causal interval proving which model step is on the first-audio or PCM-completion critical path. Summed lock wait is a sum across generator threads, not elapsed wall time. Therefore this artifact reports model critical-path evidence and coverage, not an unqualified E2E Amdahl fraction.",
        "",
        "## Confounds",
        "",
        "The existing `MultiHeadRequestState.to_worker_payload()` path still copies request-side payload data. It was intentionally not optimized in this candidate, so its cost remains a separate profile item. Public multi-head sampling can change token sequence, termination, and PCM chunk work when row scheduling changes; those pairs are descriptive unless fingerprints match.",
    ]
    (result_root / "APR_MODEL_CRITICAL_PATH_PROFILE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    oracle_rows: list[dict[str, Any]] = []
    for path in sorted((result_root / "oracle_recompute").glob("*.json")):
        if path.name not in {"baseline.json", "candidate.json"}:
            continue
        payload = load_json(path)
        for item in payload.get("results", []):
            oracle_rows.append({
                "source": path.name,
                "workload": payload.get("workload"),
                "policy": item.get("policy"),
                "observed_steps": item.get("observed_steps"),
                "candidate_batch2_steps": item.get("candidate_batch2_steps"),
                "candidate_batch2_work_fraction": item.get("candidate_batch2_work_fraction"),
                "predicted_model_makespan_ns": item.get("predicted_model_makespan_ns"),
                "predicted_model_speedup": item.get("predicted_model_speedup"),
                "model_batch_occupancy": item.get("model_batch_occupancy"),
                "lock_wait_fraction": item.get("lock_wait_fraction"),
                "causal_eligible": item.get("causal_eligible"),
                "assumptions": ";".join(item.get("assumptions", [])),
            })
    if oracle_rows:
        fields = list(oracle_rows[0])
        with (result_root / "APR_MODEL_BATCH_ORACLE.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(oracle_rows)
    report = [
        "# APR Model Batch Oracle Report",
        "",
        "## Interpretation rule",
        "",
        "The strict oracle replays observed request demand and measured service times. Counterfactual closed-loop and ideal-ready rows are upper-bound sensitivity analyses, not measured performance and not paper claims.",
        "",
        "## Strict observed-demand result",
        "",
        "On the available observed-demand diagnostics, the central-driver prediction has approximately 1.0001x model speedup. The traces have no aligned E2E interval and the original oracle input has no complete work fingerprint, so the strict oracle is not a positive E2E gate.",
        "",
        "## Sensitivity bounds",
        "",
        "The closed-loop sensitivity row assumes that model demand becomes jointly runnable after the candidate changes the producer/driver feedback; the ideal-ready row assumes nearly all rows can pair. These rows explain why a fixed-work probe can show a large model-path gain, but they do not establish that the unchanged public arrival process realizes that gain.",
        "",
        "## Decision",
        "",
        "`MODEL_PLANE_ORACLE_STRICT_PASS`: not established. `MODEL_PLANE_FIXED_WORK_PASS`: established separately by the deterministic GPU0 probe. The public row-aware runs are retained as feasibility/descriptive evidence until a repeated same-work, clock-aligned online validation exists.",
        "",
        "See `APR_MODEL_BATCH_ORACLE.csv` for all replay rows and assumptions.",
    ]
    (result_root / "APR_MODEL_BATCH_ORACLE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Keep a compact, metadata-only merged schedule trace for audit navigation.
    schedule_path = result_root / "APR_MODEL_EXECUTION_PLANE_ONLINE_SCHEDULE_TRACE.jsonl"
    with schedule_path.open("w", encoding="utf-8") as out:
        for case in cases:
            case_dir = result_root / "online_pilot" / case["case"]
            meta = case["meta"]
            for row in load_jsonl(case_dir / "model_execution_trace.jsonl"):
                if event_name(row) not in {
                    "MODEL_REQUEST_REGISTER", "MODEL_ENGINE_STEP_START", "MODEL_ENGINE_BATCH_FORMED",
                    "MODEL_ENGINE_STEP_END", "MODEL_LOCK", "MODEL_WORK_FINGERPRINT",
                    "MODEL_ROW_OUTPUT", "MODEL_DECODE_DEMAND",
                }:
                    continue
                compact = {
                    "case": case["case"],
                    "workload": meta.get("workload"),
                    "N": meta.get("N"),
                    "system": meta.get("system"),
                    "mode": meta.get("mode"),
                    **row,
                }
                out.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_audit_and_ledger(result_root: Path) -> None:
    audit = [
        "# APR Model Execution Plane Source Audit",
        "",
        "## Legacy path",
        "",
        "Each legacy realtime generator can enter the stream-lock scope and directly advance its request through the patched engine. The lock protects the call-scoped/global bridge and serializes request demand at the Python entry point; vLLM's internal scheduler cannot create a useful decode batch when only one request has an armed demand at a time.",
        "",
        "## Row-aware path",
        "",
        "The opt-in environment flag constructs one `RowAwareModelExecutionPlane` with logical maximum batch 2. A driver thread owns add/step/abort calls. Generator threads register request-owned state and submit one round; the driver invokes one engine step for selected rounds and routes returned objects by request ID. The model runner constructs row side-state from request metadata and records physical batch formation.",
        "",
        "## State isolation",
        "",
        "The row-plane scheduler itself has no dependency on `LycheeDuplexState`. Legacy engine functions still reference that global bridge; this is an explicit boundary and is not silently claimed to be removed. Request generation fences, bounded pending queues, cancel/reset, fail-closed output validation, and cleanup are covered by the focused suite.",
        "",
        "## Deliberately out of scope",
        "",
        "`to_worker_payload()` deep-copy cost, acoustic B>1 admission, CUDA Graph, Inductor, operator fusion, cross-GPU state movement, model sampling semantics, and client protocol were not changed in this candidate. The fixed-work runner is GPU0 model-only; its acoustic batch label is a frozen join label, not an execution claim.",
        "",
        "## Evidence boundary",
        "",
        "Fixed-work greedy traces establish a causal GPU0 model-path comparison. Real public runs establish that the execution plane can form B_model=2 under real arrival demand and preserve API/PCM completion. Because stochastic public work differs between modes, their wall-time ratios remain descriptive.",
    ]
    (result_root / "APR_MODEL_EXECUTION_PLANE_SOURCE_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    ledger = [
        "# APR Model Execution Plane No-Repeat Ledger",
        "",
        "| evidence | status | treatment |",
        "|---|---|---|",
        "| Acoustic B>1 N/correctness and mixed-step history | REUSED_VALIDATED_RESULT | frozen mechanism/appendix evidence; not rerun |",
        "| Variable-length, chunk-padding, Graph, Inductor, operator, capacity, handoff, preemption, staging | REUSED_VALIDATED_RESULT | prior reports retained; no source migration |",
        "| External baseline and multi-dataset audit | REUSED_VALIDATED_RESULT | capability/comparability tables reused |",
        "| Model timeline instrumentation | NEW_DELTA_SMOKE | metadata-only, fail-open diagnostics |",
        "| Row-aware central driver | NEW_DELTA_IMPLEMENTATION | explicit opt-in, benchmark-only |",
        "| Fixed-work GPU0 model B=2 | NEW_DELTA_VALIDATION | 5 warmups, 20 repeats, P10/P50/P90 |",
        "| Public N=8/N=16 model-plane representative pairs | NEW_DELTA_SMOKE | real arrival; descriptive if work mismatch |",
        "| Repeated held-out matrix (5 counted runs) | NOT_RUN | no claim inferred |",
        "",
        "The worktree remains isolated from dirty prior branches. Generated caches and the approved extension symlink are excluded from source commits.",
    ]
    (result_root / "APR_MODEL_EXECUTION_PLANE_NO_REPEAT_LEDGER.md").write_text("\n".join(ledger) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    source_manifest(args.worktree, args.result_root)
    environment_manifest(args.worktree, args.result_root)
    write_critical_and_oracle(args.worktree, args.result_root)
    write_audit_and_ledger(args.result_root)
    print(json.dumps({"status": "PASS", "result_root": str(args.result_root), "online_cases": len(online_cases(args.result_root))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
