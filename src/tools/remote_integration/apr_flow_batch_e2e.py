"""Paired real-model E2E evaluation for CFG-aware Flow batching.

The runner keeps the original-affinity path frozen and labels the two new
step paths explicitly.  It writes one append-only schedule trace and one CSV
row per attempt; invalid attempts are retained instead of being dropped.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
STEP_AUDIO_ROOT = REPO_ROOT / "third_party" / "Step-Audio2"
if str(STEP_AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP_AUDIO_ROOT))

SYSTEM_BATCH_SIZE = {
    "original_affinity": 1,
    "APR_step_B1": 1,
    "APR_step_batch_B2": 2,
    "APR_step_batch_B4": 4,
}
PRIMARY_FIELDS = (
    "run_id",
    "system",
    "workload",
    "concurrency",
    "repeat",
    "valid",
    "failure_signature",
    "elapsed_s",
    "completed_sessions",
    "pcm_chunks",
    "pcm_bytes",
    "session_throughput_sps",
    "useful_audio_seconds",
    "useful_audio_throughput_sps",
    "ttfa_p50_s",
    "ttfa_p95_s",
    "completion_p50_s",
    "completion_p95_s",
    "inter_audio_gap_p95_s",
    "effective_flow_batch_size",
    "flow_b_gt1_fraction",
    "flow_batch_executions",
    "batch_wait_ms_mean",
    "fallback_count",
    "flow_cuda_ms_total",
    "checkpoint_count",
    "restore_count",
    "worker_switch_count",
    "gpu0_mean_utilization",
    "gpu0_p95_utilization",
    "gpu0_memory_max",
    "gpu1_mean_utilization",
    "gpu1_p95_utilization",
    "gpu1_memory_max",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_flow_events(events: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    executions = [
        event
        for event in events
        if event.get("event") in {"FLOW_BATCH_FORMED", "FLOW_BATCH_FALLBACK"}
    ]
    if not executions:
        return {
            "flow_batch_executions": 0,
            "effective_flow_batch_size": 0.0,
            "flow_b_gt1_fraction": 0.0,
            "batch_wait_ms_mean": 0.0,
            "fallback_count": 0,
            "flow_cuda_ms_total": 0.0,
        }
    sizes = [int(event.get("batch_size", 0)) for event in executions]
    waits = [float(event.get("wait_ms", 0.0)) for event in executions]
    cuda_times = [
        float(event.get("cuda_time_ms", 0.0))
        for event in events
        if event.get("event") == "FLOW_BATCH_TIMING"
    ]
    return {
        "flow_batch_executions": len(executions),
        "effective_flow_batch_size": sum(sizes) / len(sizes),
        "flow_b_gt1_fraction": sum(size > 1 for size in sizes) / len(sizes),
        "batch_wait_ms_mean": sum(waits) / len(waits),
        "fallback_count": sum(
            event.get("event") == "FLOW_BATCH_FALLBACK" for event in executions
        ),
        "flow_cuda_ms_total": sum(cuda_times),
    }


def _session_values(result: Mapping[str, Any], field: str) -> list[float]:
    values: list[float] = []
    for session in result.get("sessions", ()):
        value = session.get(field)
        if value is None:
            continue
        values.append(float(value) / 1e9)
    return values


def _inter_audio_gaps(result: Mapping[str, Any]) -> list[float]:
    by_session: dict[str, list[int]] = {}
    for event in result.get("canonical_events", ()):
        if event.get("event_type") != "PCM_EGRESS":
            continue
        session_id = str(event.get("session_id", ""))
        timestamp = int(event.get("timestamp_monotonic_ns", 0))
        by_session.setdefault(session_id, []).append(timestamp)
    gaps: list[float] = []
    for timestamps in by_session.values():
        ordered = sorted(timestamps)
        gaps.extend(
            (right - left) / 1e9
            for left, right in zip(ordered, ordered[1:])
            if right >= left
        )
    return gaps


def build_primary_row(
    result: Mapping[str, Any],
    *,
    label: str,
    flow_summary: Mapping[str, Any],
) -> dict[str, Any]:
    elapsed_s = float(result.get("elapsed_s", 0.0))
    if elapsed_s <= 0:
        raise ValueError("result elapsed_s must be positive")
    pcm_bytes = int(result.get("pcm_bytes", 0))
    completed = int(result.get("completed_sessions", 0))
    useful_audio_seconds = pcm_bytes / (24000.0 * 2.0)
    gpu = result.get("gpu_metrics") or {}
    row = {
        "run_id": str(result.get("run_id", "")),
        "system": label,
        "workload": str(result.get("workload", "")),
        "concurrency": int(result.get("concurrency", 0)),
        "repeat": int(result.get("repeat", 0)),
        "valid": bool(result.get("valid", False)),
        "failure_signature": str(result.get("failure_signature", "")),
        "elapsed_s": elapsed_s,
        "completed_sessions": completed,
        "pcm_chunks": int(result.get("pcm_chunks", 0)),
        "pcm_bytes": pcm_bytes,
        "session_throughput_sps": completed / elapsed_s,
        "useful_audio_seconds": useful_audio_seconds,
        "useful_audio_throughput_sps": useful_audio_seconds / elapsed_s,
        "ttfa_p50_s": _percentile(_session_values(result, "first_playable_audio_ns"), 0.50),
        "ttfa_p95_s": _percentile(_session_values(result, "first_playable_audio_ns"), 0.95),
        "completion_p50_s": _percentile(_session_values(result, "completion_ns"), 0.50),
        "completion_p95_s": _percentile(_session_values(result, "completion_ns"), 0.95),
        "inter_audio_gap_p95_s": _percentile(_inter_audio_gaps(result), 0.95),
        "effective_flow_batch_size": float(flow_summary.get("effective_flow_batch_size", 0.0)),
        "flow_b_gt1_fraction": float(flow_summary.get("flow_b_gt1_fraction", 0.0)),
        "flow_batch_executions": int(flow_summary.get("flow_batch_executions", 0)),
        "batch_wait_ms_mean": float(flow_summary.get("batch_wait_ms_mean", 0.0)),
        "fallback_count": int(flow_summary.get("fallback_count", 0)),
        "flow_cuda_ms_total": float(flow_summary.get("flow_cuda_ms_total", 0.0)),
        "checkpoint_count": sum(int(session.get("checkpoint_count", 0)) for session in result.get("sessions", ())),
        "restore_count": sum(int(session.get("restore_count", 0)) for session in result.get("sessions", ())),
        "worker_switch_count": sum(int(session.get("worker_switch_count", 0)) for session in result.get("sessions", ())),
    }
    for field in (
        "gpu0_mean_utilization",
        "gpu0_p95_utilization",
        "gpu0_memory_max",
        "gpu1_mean_utilization",
        "gpu1_p95_utilization",
        "gpu1_memory_max",
    ):
        row[field] = float(gpu.get(field, 0.0))
    return row


def append_primary_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in PRIMARY_FIELDS})


def write_schedule_events(
    path: Path,
    events: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    system: str,
    workload: str,
    concurrency: int,
    repeat: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            payload = {
                "run_id": run_id,
                "system": system,
                "workload": workload,
                "concurrency": concurrency,
                "repeat": repeat,
                **dict(event),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _build_real_components(config: Any, label: str):
    import torch
    from real_runner import RealLycheeBatchRunner
    from token2wav import Token2wav
    from tools.apr.real_acoustic_lanes import FixedAffinityAcousticLane
    from tools.apr.flow_batch_acoustic_lane import APRFlowBatchAcousticLane

    torch.cuda.set_device(0)
    model_runner = RealLycheeBatchRunner(
        str(config.model_path),
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
    )
    torch.cuda.set_device(1)
    acoustic_model = Token2wav(str(config.token2wav_path), float16=False)
    if label == "original_affinity":
        lane = FixedAffinityAcousticLane(
            worker_count=config.worker_count,
            model=acoustic_model,
            prompt_wav=str(config.prompt_wav),
        )
    else:
        lane = APRFlowBatchAcousticLane(
            worker_count=config.worker_count,
            model=acoustic_model,
            prompt_wav=str(config.prompt_wav),
            max_batch_size=SYSTEM_BATCH_SIZE[label],
        )
    return model_runner, lane


def run_one(
    *,
    config: Any,
    label: str,
    workload: str,
    concurrency: int,
    repeat: int,
    warmup: bool,
    seed: int,
    rounds: int,
    out_root: Path,
    performance: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.apr.real_e2e_runner import RealModelAcousticHandoff
    from tools.apr.run_real_apr_e2e import run_real_correctness_attempt
    from workloads.apr_benchmark import generate_workload

    trace = generate_workload(
        workload,
        concurrency=concurrency,
        seed=seed,
        arrival_process="burst" if workload == "C" else "poisson",
    )
    model_runner, lane = _build_real_components(config, label)
    session_ids = tuple(f"session-{index}" for index in range(concurrency))
    handoff = RealModelAcousticHandoff(
        model_runner,
        stream_ids={request_id: f"stream-{request_id}" for request_id in session_ids},
        generation_ids={request_id: 0 for request_id in session_ids},
        acoustic_chunk_size=config.acoustic_chunk_size,
    )
    run_dir = out_root / "runs" / label / workload / f"N{concurrency}" / (
        f"warmup-r{repeat}" if warmup else f"r{repeat}"
    )
    result = run_real_correctness_attempt(
        config=config,
        system="original_affinity" if label == "original_affinity" else "apr",
        concurrency=concurrency,
        repeat=repeat,
        warmup=warmup,
        performance=performance,
        out_root=run_dir,
        rounds=rounds,
        model_runner=model_runner,
        lane=lane,
        handoff=handoff,
        workload_trace=trace,
        run_id=f"flow-batch-{label}-{workload}-n{concurrency}-r{repeat}-"
        f"{'warmup' if warmup else 'counted'}",
    )
    events = list(getattr(lane, "events", ()))
    schedule_path = out_root / "APR_FLOW_BATCH_SCHEDULE_TRACE.jsonl"
    write_schedule_events(
        schedule_path,
        events,
        run_id=str(result.get("run_id", "")),
        system=label,
        workload=workload,
        concurrency=concurrency,
        repeat=repeat,
    )
    return result, summarize_flow_events(events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "reports" / "apr_flow_batch_e2e")
    parser.add_argument("--systems", nargs="+", choices=tuple(SYSTEM_BATCH_SIZE), default=tuple(SYSTEM_BATCH_SIZE))
    parser.add_argument("--workloads", nargs="+", choices=("A", "B", "C"), default=("A", "B", "C"))
    parser.add_argument("--concurrencies", nargs="+", type=int, default=(4, 8, 16))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats <= 0 or any(value <= 0 for value in args.concurrencies):
        raise SystemExit("repeats and concurrencies must be positive")
    args.out_root.mkdir(parents=True, exist_ok=True)
    primary_path = args.out_root / "APR_FLOW_BATCH_PRIMARY_METRICS.csv"
    schedule_path = args.out_root / "APR_FLOW_BATCH_SCHEDULE_TRACE.jsonl"
    for path in (primary_path, schedule_path):
        if path.exists():
            path.unlink()

    from tools.apr.real_e2e_config import RealE2EConfig

    for label in args.systems:
        for workload in args.workloads:
            for concurrency in args.concurrencies:
                config = RealE2EConfig.from_env()
                config = replace(
                    config,
                    max_num_seqs=max(config.max_num_seqs, concurrency),
                    max_num_batched_tokens=max(
                        config.max_num_batched_tokens, concurrency * 128
                    ),
                )
                config.validate_paths()
                if not args.skip_warmup:
                    run_one(
                        config=config,
                        label=label,
                        workload=workload,
                        concurrency=concurrency,
                        repeat=0,
                        warmup=True,
                        seed=args.seed,
                        rounds=args.rounds,
                        out_root=args.out_root,
                        performance=False,
                    )
                for repeat in range(1, args.repeats + 1):
                    result, flow_summary = run_one(
                        config=config,
                        label=label,
                        workload=workload,
                        concurrency=concurrency,
                        repeat=repeat,
                        warmup=False,
                        seed=args.seed + repeat,
                        rounds=args.rounds,
                        out_root=args.out_root,
                        performance=args.performance,
                    )
                    row = build_primary_row(result, label=label, flow_summary=flow_summary)
                    append_primary_row(primary_path, row)
                    print(json.dumps(row, sort_keys=True))
                    if not result.get("valid", False):
                        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

