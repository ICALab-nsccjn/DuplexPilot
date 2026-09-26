#!/usr/bin/env python3
"""Generate auditable reports for the APR capacity/SLO state-v2 phase.

The script intentionally treats the 30-session calibration and the online
pilot as separate evidence classes.  It never turns an incomplete run into a
zero-valued measurement and never invents a confidence interval when a cell
has fewer than the registered paired repeats.

It is an analysis utility; it does not start a model, mutate a trace, or
change any serving code.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ROOT = Path(
    "/mnt/DuplexPilot/data/DuplexPilot/results/"
    "apr_capacity_slo_state_v2_20260829"
)
DEFAULT_WORKTREE = Path(
    "/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-capacity-slo-state-v2"
)

PILOT_COLUMNS = [
    "run_path", "system", "workload", "N", "repeat", "valid",
    "completed_sessions", "session_count", "pcm_chunks",
    "useful_audio_seconds", "completed_sessions_per_second",
    "useful_audio_throughput_sps", "e2e_makespan_s",
    "ttfa_p50_s", "ttfa_p95_s", "ttfa_p99_s",
    "completion_latency_p50_s", "completion_latency_p95_s",
    "completion_latency_p99_s", "inter_audio_gap_p95_s",
    "short_session_completion_p95_s", "slo_goodput_sps",
    "worker_switch_count", "handoff_count", "checkpoint_count",
    "restore_count", "flow_chunk_count", "flow_work_fraction",
    "worker_occupancy_fraction", "queue_wait_p95_s",
    "handoff_latency_p50_ms", "handoff_latency_p95_ms",
    "checkpoint_bytes_median", "checkpoint_bytes_max",
    "gpu1_peak_memory_gib", "ownership_errors", "runtime_errors",
    "state_contamination", "stale_output", "cleanup_ok", "oom",
    "slo_status", "evidence_class", "failure_reason",
]

SESSION_COLUMNS = [
    "run_path", "system", "workload", "N", "repeat", "session_id",
    "accepted", "done", "cancelled", "pcm_chunks", "pcm_seconds",
    "sent_chunks", "ttfa_s", "completion_latency_s", "audio_gap_p95_s",
    "event_errors", "runtime_errors", "ownership_errors",
]


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: Iterable[float], q: float) -> float | None:
    clean = sorted(v for v in values if _finite(v) is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)


def _mean(values: Iterable[float]) -> float | None:
    clean = [v for v in values if _finite(v) is not None]
    return statistics.fmean(clean) if clean else None


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("event_type") or event.get("event") or "")


def _event_ms(event: Mapping[str, Any]) -> float | None:
    for key in ("server_sse_send_epoch_ms", "server_audio_emit_epoch_ms"):
        value = _finite(event.get(key))
        if value is not None:
            return value / 1000.0
    value = _finite(event.get("timestamp_monotonic_ns"))
    return value / 1_000_000_000.0 if value is not None else None


def _pcm_bytes(event: Mapping[str, Any]) -> int:
    frame = event.get("frame_audio")
    if isinstance(frame, Mapping):
        samples = frame.get("num_samples")
        if isinstance(samples, (int, float)) and samples >= 0:
            channels = frame.get("num_channels", 1)
            return int(samples) * max(1, int(channels)) * 2
    encoded = event.get("pcm_b64")
    if not isinstance(encoded, str) or not encoded:
        return 0
    try:
        return len(base64.b64decode(encoded, validate=False))
    except Exception:
        return 0


def _session_metrics(session: Mapping[str, Any]) -> dict[str, Any]:
    events = [e for e in (session.get("events") or []) if isinstance(e, Mapping)]
    times = [t for t in (_event_ms(e) for e in events) if t is not None]
    start = min(times) if times else None
    done_times = [
        t for e in events if _event_type(e) == "done"
        for t in [_event_ms(e)] if t is not None
    ]
    finish = max(done_times) if done_times else None
    pcm = [
        e for e in events
        if _event_type(e) == "audio_chunk_pcm" or e.get("type") == "audio_chunk_pcm"
    ]
    pcm_times = sorted(t for e in pcm for t in [_event_ms(e)] if t is not None)
    ttfa = (pcm_times[0] - start) if pcm_times and start is not None else None
    completion = (finish - start) if finish is not None and start is not None else _finite(session.get("elapsed_s"))
    gaps = [b - a for a, b in zip(pcm_times, pcm_times[1:]) if b >= a]
    sample_count = 0
    for event in pcm:
        frame = event.get("frame_audio")
        if isinstance(frame, Mapping) and isinstance(frame.get("num_samples"), (int, float)):
            sample_count += int(frame["num_samples"])
        else:
            sample_count += _pcm_bytes(event) // 2
    return {
        "session_id": str(session.get("session_id", "")),
        "accepted": bool(session.get("accepted")),
        "done": bool(session.get("done")),
        "cancelled": bool(session.get("cancelled")),
        "pcm_chunks": len(pcm),
        "pcm_seconds": sample_count / 24000.0,
        "sent_chunks": int(session.get("sent_chunks", 0) or 0),
        "ttfa_s": ttfa,
        "completion_latency_s": completion,
        "audio_gap_p95_s": _percentile(gaps, 0.95),
        "event_errors": len(session.get("event_errors") or []),
        "runtime_errors": len(session.get("runtime_errors") or []),
        "ownership_errors": int(session.get("ownership_errors", 0) or 0),
    }


def _trace_stats(path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    starts: dict[tuple[str, int], float] = {}
    ready: dict[tuple[str, int], float] = {}
    commit: dict[tuple[str, int], float] = {}
    exec_intervals: list[tuple[int, float, float]] = []
    switches = 0
    handoff_latencies: list[float] = []
    checkpoint_bytes: list[float] = []
    contexts: set[str] = set()
    queue_wait: list[float] = []
    flow_chunks = 0
    checkpoint_count = 0
    restore_count = 0
    try:
        handle = path.open(encoding="utf-8")
    except OSError:
        return {}
    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, Mapping):
                continue
            typ = _event_type(event)
            counts[typ] = counts.get(typ, 0) + 1
            request = str(event.get("request_id") or "")
            seq_value = event.get("sequence_no")
            try:
                seq = int(seq_value)
            except (TypeError, ValueError):
                seq = -1
            key = (request, seq)
            ts = _finite(event.get("timestamp_monotonic_ns"))
            ts_s = ts / 1_000_000_000.0 if ts is not None else None
            if ts_s is None:
                continue
            if typ in {"FLOW_STEP_READY", "APR_CAPACITY_TOKEN_READY"}:
                ready.setdefault(key, ts_s)
            elif typ == "APR_CAPACITY_EXECUTION_START":
                starts[key] = ts_s
                if key in ready:
                    queue_wait.append(max(0.0, ts_s - ready[key]))
            elif typ == "APR_CAPACITY_EXECUTION_END":
                if key in starts:
                    worker = int(event.get("worker_id", -1))
                    exec_intervals.append((worker, starts[key], ts_s))
            elif typ == "APR_CAPACITY_CHUNK_COMMIT":
                flow_chunks += 1
                commit[key] = ts_s
                if bool(event.get("worker_switch")):
                    switches += 1
                value = _finite(event.get("checkpoint_bytes"))
                if value is not None and value > 0:
                    checkpoint_bytes.append(value)
            elif typ == "APR_HANDOFF_CAPTURE_V2":
                checkpoint_count += 1
                value = _finite(event.get("state_bytes"))
                if value is not None:
                    checkpoint_bytes.append(value)
            elif typ == "APR_WORKER_HANDOFF_COMMIT":
                handoff_latencies.append(
                    (_finite(event.get("total_latency_ns")) or 0.0) / 1_000_000.0
                )
                restore_count += 1
            context = event.get("execution_context_id")
            if isinstance(context, str) and context:
                contexts.add(context)
    # Union busy intervals by worker; the lane uses a shared model lock, so
    # this is an observed occupancy lower bound, not a promise of overlap.
    busy = 0.0
    by_worker: dict[int, list[tuple[float, float]]] = {}
    for worker, start, end in exec_intervals:
        by_worker.setdefault(worker, []).append((start, end))
    for intervals in by_worker.values():
        current_start = current_end = None
        for start, end in sorted(intervals):
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                busy += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None:
            busy += current_end - current_start
    return {
        "counts": counts,
        "worker_switch_count": switches,
        "handoff_count": counts.get("APR_WORKER_HANDOFF_COMMIT", 0),
        "checkpoint_count": checkpoint_count,
        "restore_count": restore_count,
        "flow_chunk_count": flow_chunks,
        "contexts": sorted(contexts),
        "handoff_latency_p50_ms": _percentile(handoff_latencies, 0.50),
        "handoff_latency_p95_ms": _percentile(handoff_latencies, 0.95),
        "checkpoint_bytes_median": _percentile(checkpoint_bytes, 0.50),
        "checkpoint_bytes_max": max(checkpoint_bytes) if checkpoint_bytes else None,
        "queue_wait_p95_s": _percentile(queue_wait, 0.95),
        "busy_seconds": busy,
        "execution_intervals": len(exec_intervals),
    }


def _peak_memory_gib(server_log: Path) -> float | None:
    values: list[float] = []
    try:
        text = server_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for match in re.finditer(r"(?:max_allocated_mib|max_allocated)['\"]?\s*[:=]\s*([0-9.]+)", text):
        values.append(float(match.group(1)) / 1024.0)
    return max(values) if values else None


def _run_row(run_dir: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = _read_json(run_dir / "run_metadata.json", {}) or {}
    if not metadata:
        # A hard-stopped attempt can legitimately end before the runner writes
        # its final metadata.  Infer only immutable labels from the directory
        # name and keep all measurements absent rather than fabricating them.
        name = run_dir.name
        match = re.match(r"(cap\d+)_(burst|longtail|balanced)_n(\d+)_r(\d+)", name, re.I)
        if match:
            cap, workload, n_value, repeat = match.groups()
            metadata = {
                "system": {
                    "cap0": "rsv_dsv_affinity_physical_v2",
                    "cap1": "rsv_dsv_apr_step_b1",
                    "cap2": "rsv_dsv_apr_elastic_v2",
                }.get(cap.lower()),
                "workload": {
                    "burst": "HD-Burst",
                    "longtail": "HD-LongTail",
                    "balanced": "HD-Balanced",
                }[workload.lower()],
                "N": int(n_value),
                "repeat": int(repeat),
                "expected_session_count": int(n_value),
            }
    attempt = _read_json(run_dir / "client" / "online_attempt.json", {}) or {}
    strict = _read_json(run_dir / "strict_summary.json", {}) or {}
    sessions = [_session_metrics(s) for s in (attempt.get("sessions") or []) if isinstance(s, Mapping)]
    for session in sessions:
        session.update({
            "run_path": str(run_dir.relative_to(root)),
            "system": metadata.get("system") or attempt.get("system"),
            "workload": metadata.get("workload"),
            "N": metadata.get("N"),
            "repeat": metadata.get("repeat"),
        })
    trace = _trace_stats(run_dir / "router_trace.jsonl")
    completed = int(attempt.get("completed_sessions", strict.get("completed_sessions", 0)) or 0)
    count = int(metadata.get("expected_session_count", metadata.get("N", attempt.get("session_count", 0))) or 0)
    makespan = max((s["completion_latency_s"] for s in sessions if s.get("completion_latency_s") is not None), default=None)
    audio_seconds = sum(float(s.get("pcm_seconds") or 0.0) for s in sessions)
    row: dict[str, Any] = {key: None for key in PILOT_COLUMNS}
    short_sessions = sorted(
        (s for s in sessions if s.get("completion_latency_s") is not None),
        key=lambda item: (int(item.get("sent_chunks", 0) or 0), str(item.get("session_id", ""))),
    )
    short_sessions = short_sessions[: max(1, math.ceil(len(short_sessions) / 4))]
    row.update({
        "run_path": str(run_dir.relative_to(root)),
        "system": metadata.get("system") or attempt.get("system"),
        "workload": metadata.get("workload"),
        "N": metadata.get("N"),
        "repeat": metadata.get("repeat"),
        "valid": bool(attempt.get("valid", False)),
        "completed_sessions": completed,
        "session_count": count,
        "pcm_chunks": int(attempt.get("pcm_chunks", 0) or 0),
        "useful_audio_seconds": audio_seconds,
        "e2e_makespan_s": makespan,
        "completed_sessions_per_second": (completed / makespan if makespan and makespan > 0 else None),
        "useful_audio_throughput_sps": (audio_seconds / makespan if makespan and makespan > 0 else None),
        "ttfa_p50_s": _percentile((s["ttfa_s"] for s in sessions), 0.50),
        "ttfa_p95_s": _percentile((s["ttfa_s"] for s in sessions), 0.95),
        "ttfa_p99_s": _percentile((s["ttfa_s"] for s in sessions), 0.99),
        "completion_latency_p50_s": _percentile((s["completion_latency_s"] for s in sessions), 0.50),
        "completion_latency_p95_s": _percentile((s["completion_latency_s"] for s in sessions), 0.95),
        "completion_latency_p99_s": _percentile((s["completion_latency_s"] for s in sessions), 0.99),
        "inter_audio_gap_p95_s": _percentile((s["audio_gap_p95_s"] for s in sessions), 0.95),
        "short_session_completion_p95_s": _percentile((s["completion_latency_s"] for s in short_sessions), 0.95),
        "worker_switch_count": trace.get("worker_switch_count", 0),
        "handoff_count": trace.get("handoff_count", 0),
        "checkpoint_count": trace.get("checkpoint_count", 0),
        "restore_count": trace.get("restore_count", 0),
        "flow_chunk_count": trace.get("flow_chunk_count", 0),
        # The current online router does not emit a request-level critical
        # path denominator.  Keep this unknown instead of presenting a
        # fabricated zero as a measured fraction.
        "flow_work_fraction": None,
        "queue_wait_p95_s": trace.get("queue_wait_p95_s"),
        "handoff_latency_p50_ms": trace.get("handoff_latency_p50_ms"),
        "handoff_latency_p95_ms": trace.get("handoff_latency_p95_ms"),
        "checkpoint_bytes_median": trace.get("checkpoint_bytes_median"),
        "checkpoint_bytes_max": trace.get("checkpoint_bytes_max"),
        "gpu1_peak_memory_gib": _peak_memory_gib(run_dir / "server.log"),
        "ownership_errors": int(strict.get("ownership_errors", attempt.get("ownership_errors", 0)) or 0),
        "runtime_errors": int(strict.get("runtime_errors", attempt.get("runtime_errors", 0)) or 0),
        "cleanup_ok": None,
        "oom": False,
        "slo_status": "UNFROZEN_CALIBRATION_BLOCKED",
        "evidence_class": "NEW_DELTA_SMOKE",
        "failure_reason": None if bool(attempt.get("valid", False)) else (
            "gpu1_envelope_exceeded_and_run_stopped"
            if run_dir.name == "cap2_burst_n16_r1"
            else "incomplete_or_invalid_attempt"
        ),
    })
    # The runner's shutdown diagnostic reports the CUDA device selected by the
    # server process, which is not sufficient to reconstruct GPU1's peak when
    # a run is hard-stopped.  Preserve the independently sampled nvidia-smi
    # observation for the one stopped exploratory run instead of understating
    # the envelope.
    if run_dir.name == "cap2_burst_n16_r1" and not attempt:
        row["gpu1_peak_memory_gib"] = 34239.0 / 1024.0
    try:
        server_text = (run_dir / "server.log").read_text(encoding="utf-8", errors="replace")
        row["oom"] = bool(re.search(r"outofmemory|out of memory|CUDA out of memory", server_text, re.I))
    except OSError:
        pass
    row["cleanup_ok"] = not bool(row["oom"]) and bool(row["valid"])
    return row, sessions


def _write_csv(path: Path, columns: list[str], rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _git(worktree: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=worktree, text=True).strip()
    except Exception:
        return "unknown"


def _source_manifest(root: Path, worktree: Path) -> dict[str, Any]:
    paths = [
        "lychee_fd/runtime/acoustic_checkpoint_state.py",
        "lychee_fd/runtime/token2wav_checkpoint.py",
        "lychee_fd/runtime/acoustic_checkpoint_state_v2.py",
        "lychee_fd/runtime/apr/acoustic_backend.py",
        "lychee_fd/runtime/apr/backend_provider.py",
        "lychee_fd/runtime/apr/elastic_worker_pool.py",
        "lychee_fd/runtime/apr/online_router.py",
        "lychee_fd/runtime/apr/online_step_coordinator.py",
        "lychee_fd/runtime/apr/worker_handoff.py",
        "tools/apr/capacity_slo_lanes.py",
        "tools/apr/generate_capacity_v2_manifests.py",
        "tools/apr/run_capacity_slo_mechanism.py",
        "tools/apr/run_capacity_slo_online_case.sh",
        "tools/apr/generate_capacity_slo_v2_reports.py",
        "tests/test_acoustic_checkpoint_v2.py",
        "tests/test_acoustic_worker_handoff_v2.py",
        "tests/test_capacity_slo_lanes_v2.py",
        "tests/test_capacity_state_lifecycle.py",
        "tests/test_elastic_acoustic_session.py",
        "tests/test_physical_affinity_v2.py",
    ]
    source: dict[str, Any] = {}
    for rel in paths:
        path = worktree / rel
        if path.is_file():
            source[rel] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    failed = [
        {"attempt": "calibration/cap0_r1", "reason": "trace builder invocation used unsupported manifest fields; preserved", "path": str(root / "calibration/cap0_r1")},
        {"attempt": "calibration/cap0_r2", "reason": "GPU1 CUDA OOM during 30-session all-at-once calibration", "path": str(root / "calibration/cap0_r2")},
        {"attempt": "calibration/cap0_r3_memoryfix", "reason": "no textual OOM after lifecycle fix, but timed out incomplete and observed GPU1 >32 GiB", "path": str(root / "calibration/cap0_r3_memoryfix")},
        {"attempt": "pilot_after_memoryfix/cap2_burst_n16_r1", "reason": "exploratory run manually stopped after GPU1 crossed the 32 GiB envelope; client attempt remained incomplete", "path": str(root / "pilot_after_memoryfix/cap2_burst_n16_r1")},
    ]
    return {
        "manifest_version": 2,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_tag": "safe-variable-length-flow-b2-mixed-step-v1",
        "base_commit": "fef6f2be025b0a76338255d0f009a7fe8f978d03",
        "branch": _git(worktree, "branch", "--show-current"),
        "commit": _git(worktree, "rev-parse", "HEAD"),
        "worktree": str(worktree),
        "result_directory": str(root),
        "source_files": source,
        "new_scope": ["AcousticCheckpointStateV2", "physical worker contexts and leases", "safe-boundary handoff", "physical affinity-v2 and elastic-v2"],
        "frozen_contracts": ["APR state/checkpoint v1 semantics", "RSV/DSV", "request ownership/generation identity", "Token2Wav numerical formula and PCM contract", "client protocol"],
        "reused_evidence": ["Phase 1-4 N=1/2/4/8/16 correctness", "B=2/B=4 and mixed-step mechanism", "variable-length/CUDA Graph evidence", "published baseline/public trace audit"],
        "benchmark_commands": [
            "pytest (container): 101 passed, 1 skipped, relevant capacity/Flow/PCM suite",
            "tools/apr/run_capacity_slo_mechanism.py --repeats 3 (container, A100)",
            "tools/apr/run_capacity_slo_online_case.sh (container server/client, real paced trace)",
        ],
        "failed_attempts": failed,
    }


def _write_markdown_reports(root: Path, worktree: Path, rows: list[dict[str, Any]], session_rows: list[dict[str, Any]]) -> None:
    mechanism = root / "mechanism_gate_v4_memoryfix"
    source = _source_manifest(root, worktree)
    (root / "APR_CAPACITY_SLO_STATE_V2_SOURCE_MANIFEST.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    environment = {
        "manifest_version": 2,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "container": "duplexpilot", "image": "triepilot-a100:20260720-ubuntu2404-lightweight",
        "image_id": "9021559e72f1e1eec139ba25ab6b1788595a91a79875b3bdbedc07bc48f9f17e",
        "cuda_visible_devices": "0,1", "gpu0_role": "model/vLLM", "gpu1_role": "Local Token2Wav/Flow",
        "physical_acoustic_workers": 2, "flow_attention_cache_capacity": 2048,
        "gpu1_peak_memory_envelope_gib": 32, "python": "Python 3.10.19",
        "torch": "2.9.1+cu128", "cuda_runtime": "12.8",
        "model_path": "/mnt/DuplexPilot/data/models/lychee_full_duplex",
        "token2wav_path": "/mnt/DuplexPilot/data/models/token2wav",
        "placement": {"LYCHEEFD_TOKEN2WAV_DEVICE": "1", "LYCHEEFD_REQUIRE_DUAL_GPU_PLACEMENT": "1"},
        "notes": ["All model/tests run inside duplexpilot.", "GPU0 model KV is not migrated.", "streams/events/workspaces/queues are excluded from v2 checkpoint."],
    }
    (root / "APR_CAPACITY_SLO_STATE_V2_ENVIRONMENT_MANIFEST.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger = """# APR Capacity/SLO v2 no-repeat ledger

Base: `safe-variable-length-flow-b2-mixed-step-v1` (`fef6f2be025b0a76338255d0f009a7fe8f978d03`).
Execution branch: `apr-capacity-slo-state-v2`; final analyzed commit is recorded in the source manifest.

## Reused without rerun

- Phase 1–4 N=1/2/4/8/16 correctness and previously sealed B>1/Graph evidence.
- Existing public-trace, baseline and external-system audits.

## New delta work

- Checkpoint v2, physical worker context/lease and safe-boundary handoff implementation.
- Lifecycle correction removing redundant full CUDA state snapshots; fixed-affinity chunks now checkpoint zero times, and elastic counts only real handoffs.
- Focused regression: 101 passed, 1 skipped.
- A100 mechanism gate v4: 3/3 PASS.
- Bounded online exploratory runs after the lifecycle correction.

## Preserved failed attempts

- `calibration/cap0_r1`: calibration builder invocation/schema failure.
- `calibration/cap0_r2`: CUDA OOM during 30-session all-at-once calibration.
- `calibration/cap0_r3_memoryfix`: incomplete timeout; observed GPU1 exceeded 32 GiB envelope.

These attempts remain on disk and are excluded from performance claims. No held-out SLO matrix was run because the SLO threshold could not be frozen validly.
"""
    (root / "APR_CAPACITY_SLO_NO_REPEAT_LEDGER.md").write_text(ledger, encoding="utf-8")

    # Copy the source-level contract documents, then add execution-specific
    # status rather than silently changing their original wording.
    for name in ("APR_CHECKPOINT_V2_SCHEMA.md", "APR_PHYSICAL_WORKER_FEASIBILITY_REPORT.md"):
        source_path = worktree / name
        if source_path.is_file():
            shutil.copyfile(source_path, root / name)
    mechanism_csv = mechanism / "APR_CAPACITY_SLO_MECHANISM_METRICS.csv"
    if mechanism_csv.is_file():
        shutil.copyfile(mechanism_csv, root / "APR_CAPACITY_SLO_MECHANISM_METRICS.csv")
    mechanism_report = mechanism / "APR_WORKER_HANDOFF_CORRECTNESS_REPORT.md"
    if mechanism_report.is_file():
        shutil.copyfile(mechanism_report, root / "APR_WORKER_HANDOFF_CORRECTNESS_REPORT.md")
    mechanism_trace = mechanism / "APR_CAPACITY_SLO_SCHEDULE_TRACE.jsonl"
    with (root / "APR_CAPACITY_SLO_SCHEDULE_TRACE.jsonl").open("w", encoding="utf-8") as out:
        if mechanism_trace.is_file():
            for line in mechanism_trace.read_text(encoding="utf-8", errors="replace").splitlines():
                out.write(line + "\n")
        for row in rows:
            run_dir = root / str(row["run_path"])
            trace = run_dir / "router_trace.jsonl"
            if not trace.is_file():
                continue
            for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    event["run_path"] = row["run_path"]
                    event["evidence_class"] = row["evidence_class"]
                    out.write(json.dumps(event, sort_keys=True) + "\n")

    _write_csv(root / "APR_CAPACITY_SLO_PILOT_METRICS.csv", PILOT_COLUMNS, rows)
    _write_csv(root / "APR_CAPACITY_SLO_HELDOUT_METRICS.csv", ["status", "reason", "planned_cells"], [{"status": "NOT_RUN_CALIBRATION_BLOCKED", "reason": "30-session CAP0 calibration did not complete within the fixed 32 GiB envelope; no SLO was frozen.", "planned_cells": "CAP0/CAP1/CAP2 × LongTail/Burst × N8/N16 × 5"}])
    try:
        import pandas as pd
        pd.DataFrame(session_rows, columns=SESSION_COLUMNS).to_parquet(root / "APR_CAPACITY_SLO_SESSION_METRICS.parquet", index=False)
    except Exception as exc:
        (root / "APR_CAPACITY_SLO_SESSION_METRICS.parquet.blocked.txt").write_text(f"PARQUET_BLOCKED: {type(exc).__name__}: {exc}\n", encoding="utf-8")

    valid = [r for r in rows if r.get("valid")]
    switches = [r for r in valid if int(r.get("worker_switch_count") or 0) > 0]
    cap0 = [r for r in valid if r.get("system") == "rsv_dsv_affinity_physical_v2"]
    cap2 = [r for r in valid if r.get("system") == "rsv_dsv_apr_elastic_v2"]
    def med(rows_: list[Mapping[str, Any]], key: str) -> float | None:
        return _percentile((_finite(r.get(key)) for r in rows_), 0.5)
    def fmt(value: Any, digits: int = 3) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    (root / "APR_CHECKPOINT_V2_CORRECTNESS_REPORT.md").write_text("""# Checkpoint v2 correctness report

The focused container suite passed 101 tests with one explicitly skipped test. It covers v2 round-trip, strict fields/checksum, v1 read-only compatibility, stale generation fences, pending PCM identity, rollback, cancel/reset, and exclusion of physical worker resources from logical state accounting.

The lifecycle regression added after the first calibration attempts also passes: fixed-affinity execution does not retain a per-chunk checkpoint, and the logical view aliases one request-owned stream state. A v2 snapshot is transient and counted only for an actual handoff.

No checkpoint-format or restore-semantic change was made after this gate.
""", encoding="utf-8")
    (root / "APR_PHYSICAL_WORKER_FEASIBILITY_REPORT.md").write_text("""# Physical worker feasibility report

## Result: PASS for the bounded mechanism lane

The A100 mechanism gate (`mechanism_gate_v4_memoryfix`) completed 3/3 repeats. Each repeat used two distinct GPU1 CUDA streams and stable, distinct `worker_instance_id` and `execution_context_id` values. The elastic lane performed three safe-boundary handoffs per repeat; source and target context identities differed. PCM was non-empty and numerically identical in the registered mechanism comparison (RMSE 0, correlation 1), with zero ownership errors and successful cleanup. GPU1 peak allocation was 6.245–6.316 GiB in the bounded gate.

The lane shares model weights and uses a shared execution lock because the Token2Wav module is not proven safe for concurrent model calls. Thus this result proves physical context/lease/handoff feasibility, not parallel acoustic throughput.

The subsequent 30-session online calibration did not complete within the 32 GiB envelope, so capacity/SLO improvement remains unestablished.
""", encoding="utf-8")
    (root / "APR_WORKER_HANDOFF_CORRECTNESS_REPORT.md").write_text("""# Worker handoff correctness report

The latest A100 gate is the authoritative report. It records 3/3 valid repeats, real source/target context changes, 3 handoffs and 3 checkpoint/restore pairs per repeat, exact PCM comparisons, zero ownership errors, cleanup PASS, and GPU1 peak below 32 GiB.

The earlier v3 report over-counted checkpoint snapshots because the lane retained redundant per-chunk snapshots. That implementation accounting issue was fixed and is superseded by v4; the v4 count is the semantically correct handoff count.
""", encoding="utf-8")

    (root / "APR_CAPACITY_SLO_STATISTICAL_ANALYSIS.md").write_text(f"""# Capacity/SLO statistical analysis

## Status

No registered paired statistical test is authorized. The 30-session CAP0 calibration did not yield a completed, <=32 GiB baseline from which TTFA/audio-gap SLOs could be frozen. The available post-fix online rows are one-run exploratory observations, not five-repeat paired cells.

## Available descriptive rows

- valid post-fix rows: `{len(valid)}`
- rows with at least one observed worker switch: `{len(switches)}`
- physical-affinity rows: `{len(cap0)}`; elastic rows: `{len(cap2)}`
- CAP0 median useful-audio throughput (descriptive): `{fmt(med(cap0, 'useful_audio_throughput_sps'))}`
- CAP2 median useful-audio throughput (descriptive): `{fmt(med(cap2, 'useful_audio_throughput_sps'))}`

Because output PCM work differs across these exploratory runs and the SLO threshold is unfrozen, no ratio, bootstrap interval, p-value, or causal performance claim is reported. Held-out and formal matrices are explicitly `NOT_RUN`.
""", encoding="utf-8")

    (root / "APR_CAPACITY_SLO_E2E_REPORT.md").write_text(f"""# APR Capacity/SLO E2E report

## Executive result

The state-v2 implementation passes the bounded mechanism gate and the post-fix N=8/N=16 online correctness smoke. It does **not** yet establish a fixed-hardware capacity or SLO advantage.

### Mechanism evidence

- 3/3 dual-A100 handoff repeats PASS.
- Source and target physical execution contexts are distinct.
- PCM contract, identity, cleanup and ownership checks PASS.
- The corrected lifecycle removes redundant full CUDA state snapshots; this reduced observed memory materially, but is an implementation correctness/lifecycle result rather than a speedup claim.

### Online evidence

- Post-fix LongTail CAP0/CAP2 N=8 and N=16 runs completed with non-empty PCM and zero ownership/runtime errors.
- Post-fix Burst CAP0 N=8 and CAP2 N=8 runs completed with non-empty PCM and zero ownership/runtime errors.
- These are one-repeat exploratory rows; PCM chunk counts and generated audio work are not identical across systems, so they cannot support a paired throughput comparison.
- The 30-session calibration attempted three times. The first had a trace-builder/schema failure; the second hit CUDA OOM; the third avoided textual OOM after the lifecycle fix but timed out incomplete while observed GPU1 usage exceeded the conservative 32 GiB envelope.
- A separate N=16 Burst elastic exploratory run was stopped at the hard envelope after an independent `nvidia-smi` sample of 34,239 MiB (33.44 GiB) on GPU1; its incomplete trace is retained and excluded.

## Capacity/SLO gate

`BLOCKED`: CAP0 calibration did not complete, so TTFA and audio-gap SLOs were not frozen. Consequently no held-out five-repeat run or formal matrix was executed. This is an evidence boundary, not a claim that elasticity has no value.

## Architecture caveat

The current benchmark lane uses two distinct streams/leases but a shared model execution lock. It therefore validates safe physical handoff and request-owned state, while limiting any parallel execution interpretation. All 30 logical acoustic histories remain GPU-resident; no CPU offload or GPU0 KV migration was introduced.
""", encoding="utf-8")

    (root / "APR_CAPACITY_SLO_PILOT_BLOCKED_REPORT.md").write_text("""# APR Capacity/SLO pilot: BLOCKED

The registered capacity/SLO pilot cannot be promoted to a paper result because the independent 30-session CAP0 calibration did not complete inside the fixed 32 GiB GPU1 envelope. The preserved attempts are:

1. `calibration/cap0_r1`: builder/schema invocation failure.
2. `calibration/cap0_r2`: CUDA OOM during the all-at-once 30-session run.
3. `calibration/cap0_r3_memoryfix`: no textual OOM after the lifecycle correction, but incomplete at timeout and observed GPU1 usage above 32 GiB.
4. `pilot_after_memoryfix/cap2_burst_n16_r1`: independently sampled GPU1 at 34,239 MiB (33.44 GiB); run stopped before completion and its trace is retained.

The bounded post-fix N=8/N=16 online smoke is retained as `NEW_DELTA_SMOKE` only. No threshold was tuned from candidate results, no held-out rows were substituted, and no failed attempt was deleted.

The correct next action is to treat this phase as mechanism PASS / capacity-SLO BLOCKED, then decide separately whether to redesign the calibration/load envelope or stop the capacity claim. This report does not authorize additional uncontrolled scheduler or checkpoint changes.
""", encoding="utf-8")
    (root / "APR_CAPACITY_SLO_FINAL_VERDICT.md").write_text("""# APR Capacity/SLO state-v2 final verdict

## Classification

**MECHANISM_PASS / CAPACITY_SLO_BLOCKED**

### Established

1. Request-owned acoustic continuation state can be captured in explicit checkpoint schema v2 and restored at a safe boundary.
2. The dual-A100 benchmark lane creates physically distinct worker contexts and leases; real worker handoff occurs and preserves PCM/identity semantics.
3. Removing redundant state snapshots fixes a real memory-lifecycle defect. The focused suite and post-fix N=8/N=16 correctness smokes pass.

### Not established

1. A stable maximum-concurrency improvement under a frozen SLO.
2. A reduction in TTFA/audio-gap violations or a SLO-goodput gain.
3. Any statistically supported CAP2 advantage over physical affinity.

The calibration failure is itself part of the result: retaining every long-lived GPU acoustic history at once exceeds the conservative 32 GiB capacity/time envelope even after eliminating redundant clones. No held-out/formal SLO matrix was run, so there is no defensible performance pass or fail beyond this blocked boundary.

## Paper decision

Use the handoff and state-ownership results as mechanism evidence and state the capacity/SLO claim as unestablished. Do not convert the controlled handoff result or one-run exploratory rows into a throughput/capacity claim. Re-open a capacity experiment only with a preregistered, valid calibration/load envelope; otherwise keep the paper claim to explicit acoustic state virtualization, safe handoff, and its measured limitations.
""", encoding="utf-8")
    (root / "APR_PAPER_CLAIM_BOUNDARY_V2.md").write_text("""# APR paper claim boundary v2

## Permitted claims

- APR makes acoustic continuation state explicit and request-owned.
- Checkpoint v2 and safe-boundary fencing support auditable handoff between distinct physical execution contexts.
- B>1 remains a separately validated mechanism capability, not a universal online speedup.
- The current fixed-hardware capacity/SLO experiment is blocked by the calibration envelope; this limitation is reported transparently.

## Prohibited claims from this phase

- “APR improves capacity/SLO goodput by X%.”
- “A handoff mechanism implies parallel acoustic execution.”
- “One exploratory CAP0/CAP2 run proves an E2E gain.”
- Treating the 3× controlled handoff count or prior B>1 controlled speedups as public online results.

The current implementation shares model weights and serializes model execution with a lock; this must be disclosed whenever physical contexts are discussed.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    args = parser.parse_args()
    root = args.root.resolve()
    worktree = args.worktree.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    # Only post-lifecycle-fix rows are eligible for the generated pilot CSV.
    pilot_root = root / "pilot_after_memoryfix"
    for run_dir in sorted(pilot_root.iterdir()) if pilot_root.is_dir() else []:
        if not run_dir.is_dir() or not (
            (run_dir / "run_metadata.json").is_file()
            or (run_dir / "router_trace.jsonl").is_file()
        ):
            continue
        row, sessions = _run_row(run_dir, root)
        rows.append(row)
        for session in sessions:
            session_rows.append(session)
    _write_markdown_reports(root, worktree, rows, session_rows)
    print(json.dumps({
        "root": str(root), "pilot_rows": len(rows),
        "valid_rows": sum(bool(r.get("valid")) for r in rows),
        "session_rows": len(session_rows),
        "commit": _git(worktree, "rev-parse", "HEAD"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
