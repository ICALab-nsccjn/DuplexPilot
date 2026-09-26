#!/usr/bin/env python3
"""Compare copy-based and same-GPU zero-copy acoustic handoff.

This is a bounded A100 mechanism benchmark.  It deliberately reuses the
registered handoff schedule from ``run_capacity_slo_mechanism`` and never
claims parallel worker throughput: the lane still uses the shared execution
lock.  The only experimental variable is how the request-owned lease is
transferred between two physical GPU1 contexts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOKEN2WAV_ROOT = REPO_ROOT / "third_party" / "Step-Audio2"
if str(TOKEN2WAV_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKEN2WAV_ROOT))

from tools.apr.real_e2e_config import RealE2EConfig
from tools.apr.capacity_slo_lanes import ElasticAcousticLaneV2
from tools.apr.run_capacity_slo_mechanism import (
    _configure,
    _jsonable,
    _pcm_metrics,
    _schedule,
)


PROTOCOL = "APR-CAPACITY-SLO-ZERO-COPY-v1"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _reset_rng(torch: Any, cpu_state: Any, cuda_state: Any) -> None:
    """Restore one captured RNG snapshot before each paired mode."""
    torch.set_rng_state(cpu_state.clone())
    torch.cuda.set_rng_state(cuda_state.clone(), device=1)


def _run_mode(
    model: Any,
    config: RealE2EConfig,
    out_root: Path,
    repeat: int,
    mode: str,
    cpu_rng_state: Any,
    cuda_rng_state: Any,
) -> dict[str, Any]:
    import torch

    _reset_rng(torch, cpu_rng_state, cuda_rng_state)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(1)
    mode_dir = out_root / "attempts" / f"r{repeat}" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    lane = ElasticAcousticLaneV2(
        worker_count=2,
        model=model,
        prompt_wav=str(config.prompt_wav),
        device="cuda:1",
        assignment_policy="forced_round_robin",
        handoff_mode=mode,
    )
    start_ns = time.perf_counter_ns()
    try:
        result = _schedule(
            lane,
            schedule_name=f"elastic_{mode}",
            events_path=mode_dir / "events.jsonl",
            honor_requested_targets=True,
        )
    finally:
        lane.close()
    # _schedule returns before the lane is closed.  Read cleanup only after
    # close(), otherwise a successful run is incorrectly reported as a
    # cleanup failure and cannot pass the registered correctness gate.
    result["cleanup_ok"] = bool(lane.cleanup_ok)
    torch.cuda.synchronize(1)
    elapsed_ns = time.perf_counter_ns() - start_ns
    peak_bytes = int(torch.cuda.max_memory_allocated(1))
    events = list(lane.events)
    if mode == "zero_copy":
        commit_events = [
            event
            for event in events
            if event.get("event_type") == "APR_ZERO_COPY_HANDOFF_COMMIT"
        ]
    else:
        commit_events = [
            event
            for event in events
            if event.get("event_type") == "APR_WORKER_HANDOFF_COMMIT"
        ]
    handoff_latencies = [
        int(event.get("total_latency_ns", 0)) for event in commit_events
    ]
    copied_bytes = [
        int(event.get("copied_state_bytes", 0)) for event in commit_events
    ]
    checkpoint_bytes = [
        int(event.get("state_bytes", 0)) for event in events
        if event.get("event_type") == "APR_HANDOFF_CAPTURE_V2"
    ]
    payload = {
        "protocol": PROTOCOL,
        "repeat": repeat,
        "mode": mode,
        "source_commit": _git_commit(),
        "handoff_count": int(result["handoff_count"]),
        "checkpoint_count": int(result["checkpoint_count"]),
        "restore_count": int(result["restore_count"]),
        "zero_copy_commit_count": len(commit_events) if mode == "zero_copy" else 0,
        "handoff_latency_ns": handoff_latencies,
        "copied_state_bytes": copied_bytes,
        "checkpoint_bytes": checkpoint_bytes,
        "ownership_errors": int(result["ownership_errors"]),
        "cleanup_ok": bool(result["cleanup_ok"]),
        "elapsed_ns": elapsed_ns,
        "peak_gpu1_memory_bytes": peak_bytes,
        "progress": result["progress"],
        "events": _jsonable(events),
        "pcm_by_request": result["pcm_by_request"],
    }
    # Keep the large PCM payload out of the JSON artifact while returning it
    # to the caller for the paired numerical comparison.
    pcm_by_request = payload.pop("pcm_by_request")
    (mode_dir / "summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["pcm_by_request"] = pcm_by_request
    return payload


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    latencies = payload["handoff_latency_ns"]
    copied = payload["copied_state_bytes"]
    checkpoints = payload["checkpoint_bytes"]
    return {
        "repeat": payload["repeat"],
        "mode": payload["mode"],
        "handoff_count": payload["handoff_count"],
        "checkpoint_count": payload["checkpoint_count"],
        "restore_count": payload["restore_count"],
        "zero_copy_commit_count": payload["zero_copy_commit_count"],
        "handoff_latency_p50_ns": sorted(latencies)[len(latencies) // 2]
        if latencies
        else 0,
        "handoff_latency_p95_ns": sorted(latencies)[
            min(len(latencies) - 1, int(len(latencies) * 0.95))
        ]
        if latencies
        else 0,
        "copied_state_bytes": max(copied) if copied else 0,
        "checkpoint_bytes": max(checkpoints) if checkpoints else 0,
        "elapsed_ns": payload["elapsed_ns"],
        "peak_gpu1_memory_bytes": payload["peak_gpu1_memory_bytes"],
        "ownership_errors": payload["ownership_errors"],
        "cleanup_ok": payload["cleanup_ok"],
    }


def _write_outputs(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]], out_root: Path) -> None:
    fields = list(rows[0]) if rows else []
    with (out_root / "APR_ZERO_COPY_HANDOFF_METRICS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "APR_ZERO_COPY_HANDOFF_COMPARISONS.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_root / "APR_ZERO_COPY_HANDOFF_SCHEDULE_TRACE.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in comparisons),
        encoding="utf-8",
    )

    correctness = all(
        item["pcm_pass"]
        and item["copy_cleanup_ok"]
        and item["zero_cleanup_ok"]
        and item["copy_ownership_errors"] == 0
        and item["zero_ownership_errors"] == 0
        and item["zero_switch_count"] >= 1
        and item["zero_context_distinct"]
        and item["peak_gpu1_memory_gib"] <= 32.0
        for item in comparisons
    )
    latency_ratios = [
        item["zero_handoff_p95_ns"] / item["copy_handoff_p95_ns"]
        for item in comparisons
        if item["copy_handoff_p95_ns"] > 0
    ]
    median_ratio = sorted(latency_ratios)[len(latency_ratios) // 2] if latency_ratios else None
    gate = correctness and median_ratio is not None and median_ratio <= 0.20
    lines = [
        "# APR Zero-Copy Handoff Feasibility",
        "",
        "This is a bounded A100 mechanism comparison. It does not claim physical parallel throughput: both modes use the lane's shared execution lock.",
        "",
        f"- protocol: `{PROTOCOL}`",
        f"- source commit: `{_git_commit()}`",
        "- hardware: two NVIDIA A100-SXM4-40GB; GPU1 hosts Local Token2Wav/Flow",
        "- comparison: copy-based checkpoint/restore vs same-GPU in-memory lease transfer",
        "",
        f"## Gate: {'PASS' if gate else 'FAIL'}",
        "",
        "The gate requires PCM/identity/cleanup correctness, a real context switch, <=32 GiB GPU1 peak, and zero-copy handoff p95 no more than 20% of the copy path.",
        "",
        "| repeat | copy p95 (ms) | zero-copy p95 (ms) | ratio | copy bytes | zero-copy bytes | PCM | contexts | peak GiB | |",
        "|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['repeat']} | {item['copy_handoff_p95_ns']/1e6:.3f} | "
            f"{item['zero_handoff_p95_ns']/1e6:.3f} | "
            f"{item['zero_handoff_p95_ns']/item['copy_handoff_p95_ns']:.3f} | "
            f"{item['copy_state_bytes']} | {item['zero_state_bytes']} | "
            f"{'PASS' if item['pcm_pass'] else 'FAIL'} | "
            f"{'PASS' if item['zero_context_distinct'] else 'FAIL'} | "
            f"{item['peak_gpu1_memory_gib']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "`copied_state_bytes=0` and the zero-copy API's storage-pointer contract establish that the hot path does not clone or serialize the logical state. The A100 result is still a handoff-latency feasibility result; it must be combined with a paced public online CAP pilot before making an SLO claim.",
        "",
    ]
    (out_root / "APR_ZERO_COPY_HANDOFF_FEASIBILITY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    correctness_lines = [
        "# APR Zero-Copy Handoff Correctness Report",
        "",
        f"- overall gate: **{'PASS' if correctness else 'FAIL'}**",
        f"- repeats: {len(comparisons)}",
        "- all comparisons use the same deterministic token schedule and RNG reset.",
        "",
        "| repeat | copy switches | zero-copy switches | copy checkpoint/restore | zero-copy checkpoint/restore | ownership | cleanup | PCM contract |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in comparisons:
        correctness_lines.append(
            f"| {item['repeat']} | {item['copy_switch_count']} | {item['zero_switch_count']} | "
            f"{item['copy_checkpoint_count']}/{item['copy_restore_count']} | "
            f"{item['zero_checkpoint_count']}/{item['zero_restore_count']} | "
            f"{'PASS' if item['copy_ownership_errors'] == item['zero_ownership_errors'] == 0 else 'FAIL'} | "
            f"{'PASS' if item['copy_cleanup_ok'] and item['zero_cleanup_ok'] else 'FAIL'} | "
            f"{'PASS' if item['pcm_pass'] else 'FAIL'} |"
        )
    (out_root / "APR_ZERO_COPY_HANDOFF_CORRECTNESS_REPORT.md").write_text(
        "\n".join(correctness_lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats != 3:
        parser.error("the registered comparison requires exactly 3 repeats")
    _configure()
    config = RealE2EConfig.from_env()
    config.validate_paths()
    if config.worker_count != 2:
        parser.error("the comparison requires two acoustic workers")
    args.out_root.mkdir(parents=True, exist_ok=True)
    import torch
    from token2wav import Token2wav

    model = None
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    try:
        model = Token2wav(str(config.token2wav_path), float16=False)
        cpu_rng_state = torch.get_rng_state().clone()
        cuda_rng_state = torch.cuda.get_rng_state(device=1).clone()
        for repeat in range(1, 4):
            copy = _run_mode(
                model,
                config,
                args.out_root,
                repeat,
                "copy",
                cpu_rng_state,
                cuda_rng_state,
            )
            zero = _run_mode(
                model,
                config,
                args.out_root,
                repeat,
                "zero_copy",
                cpu_rng_state,
                cuda_rng_state,
            )
            rows.extend((_row(copy), _row(zero)))
            pcm_metrics = {
                request_id: _pcm_metrics(
                    copy["pcm_by_request"][request_id],
                    zero["pcm_by_request"][request_id],
                )
                for request_id in ("A", "B", "C")
            }
            switches = [
                item
                for item in zero["progress"]
                if bool(item.get("worker_switch"))
            ]
            context_distinct = bool(switches) and all(
                item.get("source_execution_context_id")
                != item.get("execution_context_id")
                for item in switches
            )
            pcm_pass = all(
                metric["same_length"]
                and metric["normalized_rmse"] <= 0.02
                and metric["correlation"] >= 0.99
                and metric["snr_db"] >= 34.0
                for metric in pcm_metrics.values()
            )
            copy_p95 = _row(copy)["handoff_latency_p95_ns"]
            zero_p95 = _row(zero)["handoff_latency_p95_ns"]
            comparisons.append(
                {
                    "repeat": repeat,
                    "copy_handoff_p95_ns": copy_p95,
                    "zero_handoff_p95_ns": zero_p95,
                    "copy_state_bytes": max(copy["checkpoint_bytes"] or [0]),
                    "zero_state_bytes": max(zero["copied_state_bytes"] or [0]),
                    "copy_checkpoint_count": copy["checkpoint_count"],
                    "copy_restore_count": copy["restore_count"],
                    "zero_checkpoint_count": zero["checkpoint_count"],
                    "zero_restore_count": zero["restore_count"],
                    "copy_switch_count": sum(bool(item.get("worker_switch")) for item in copy["progress"]),
                    "zero_switch_count": len(switches),
                    "zero_context_distinct": context_distinct,
                    "copy_ownership_errors": copy["ownership_errors"],
                    "zero_ownership_errors": zero["ownership_errors"],
                    "copy_cleanup_ok": copy["cleanup_ok"],
                    "zero_cleanup_ok": zero["cleanup_ok"],
                    "peak_gpu1_memory_gib": max(
                        copy["peak_gpu1_memory_bytes"],
                        zero["peak_gpu1_memory_bytes"],
                    )
                    / (1024**3),
                    "pcm_pass": pcm_pass,
                    "pcm_metrics": pcm_metrics,
                }
            )
    except Exception as exc:
        (args.out_root / "APR_ZERO_COPY_HANDOFF_FAILURE.json").write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "source_commit": _git_commit(),
                    "failure_type": type(exc).__name__,
                    "failure": str(exc),
                    "completed_repeats": len(comparisons),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(False)
    _write_outputs(rows, comparisons, args.out_root)
    gate = all(
        item["pcm_pass"]
        and item["zero_switch_count"] >= 1
        and item["zero_context_distinct"]
        and item["copy_ownership_errors"] == item["zero_ownership_errors"] == 0
        and item["copy_cleanup_ok"]
        and item["zero_cleanup_ok"]
        and item["peak_gpu1_memory_gib"] <= 32.0
        for item in comparisons
    ) and len(comparisons) == 3
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
