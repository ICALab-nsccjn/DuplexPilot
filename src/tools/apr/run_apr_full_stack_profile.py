#!/usr/bin/env python3
"""Run one profiling-only real APR/original-affinity diagnostic attempt.

The output is raw attribution evidence.  It is intentionally marked ineligible
for formal performance aggregates and does not alter any serving policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiling.apr_timeline import TimelineSink
from lychee_fd.runtime.apr.profiling import StageProfiler
from tools.apr.real_acoustic_lanes import APRLocalAcousticLane, FixedAffinityAcousticLane
from tools.apr.real_e2e_config import RealE2EConfig
from tools.apr.real_e2e_runner import RealModelAcousticHandoff
from tools.apr.run_real_apr_e2e import build_real_components, run_real_correctness_attempt
from workloads.apr_benchmark import generate_workload, trace_hash


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run_profile_attempt(
    *,
    system: str,
    workload: str,
    concurrency: int,
    repeat: int,
    warmup: bool,
    out_root: Path,
    seed: int = 0,
    arrival_process: str = "poisson",
    rounds: int = 4,
) -> dict:
    if system not in {"apr", "original_affinity"}:
        raise ValueError("system must be apr or original_affinity")
    if concurrency <= 0 or repeat <= 0 or rounds <= 0:
        raise ValueError("concurrency, repeat, and rounds must be positive")
    config = RealE2EConfig.from_env()
    config.validate_paths()
    trace = generate_workload(
        workload,
        concurrency=concurrency,
        seed=seed,
        arrival_process=arrival_process,
    )
    run_id = f"apr-fullstack-{system}-n{concurrency}-r{repeat}-{uuid.uuid4().hex[:12]}"
    run_root = Path(out_root) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    timeline = TimelineSink(
        enabled=True,
        run_context={
            "run_id": run_id,
            "system": system,
            "workload": trace.workload_id,
            "concurrency": concurrency,
            "repeat": repeat,
        },
    )
    profiler = StageProfiler(
        enabled=True,
        run_context=timeline.run_context,
        timeline=timeline,
    )
    model_runner, lane = build_real_components(config, system, profiler=profiler)
    session_ids = tuple(f"session-{index}" for index in range(concurrency))
    handoff = RealModelAcousticHandoff(
        model_runner,
        stream_ids={request_id: f"stream-{request_id}" for request_id in session_ids},
        generation_ids={request_id: 0 for request_id in session_ids},
        acoustic_chunk_size=config.acoustic_chunk_size,
        profiler=profiler,
    )
    result = run_real_correctness_attempt(
        config=config,
        system=system,
        concurrency=concurrency,
        repeat=repeat,
        warmup=warmup,
        rounds=rounds,
        performance=True,
        out_root=run_root,
        model_runner=model_runner,
        lane=lane,
        handoff=handoff,
        workload_trace=trace,
        profiler=profiler,
        run_id=run_id,
    )
    attempt_dir = run_root / "attempts" / system / f"N{concurrency}" / (
        f"warmup-r{repeat}" if warmup else f"r{repeat}"
    )
    profile_error = ""
    for event in result.get("canonical_events", ()):
        if event.get("event_type") != "CLEANUP" or not event.get("session_id"):
            continue
        try:
            timeline.emit(
                "request_finish",
                session_id=str(event["session_id"]),
                timestamp_monotonic_ns=int(event["timestamp_monotonic_ns"]),
                details={"cleanup": bool(event.get("cleanup"))},
            )
        except Exception:
            profile_error = profile_error or "TimelineContractError:request_finish"
    try:
        timeline.flush_json(attempt_dir / "apr_full_timeline.json")
    except Exception as exc:
        # Serving may have completed, but a missing/invalid profile is an
        # invalid diagnostic attempt and must not enter attribution summaries.
        profile_error = f"{type(exc).__name__}:{exc}"
    metadata = {
        "run_id": run_id,
        "system": system,
        "workload": workload,
        "concurrency": concurrency,
        "repeat": repeat,
        "warmup": warmup,
        "diagnostic_only": True,
        "formal_aggregate_eligible": False,
        "source_commit": _commit(),
        "workload_trace_hash": trace_hash(trace),
        "timeline_schema_version": "apr-full-timeline-v1",
        "profile_schema_version": "apr-profile-v1",
        "nsight_available": False,
        "profile_valid": not bool(profile_error),
        "profile_error": profile_error,
    }
    (attempt_dir / "profiling_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "valid": result["valid"], "attempt_dir": str(attempt_dir)}, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("apr", "original_affinity"), required=True)
    parser.add_argument("--workload", choices=("A", "B", "C"), default="A")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-process", choices=("poisson", "burst"), default="poisson")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "reports" / "apr_full_stack_profile" / "raw")
    args = parser.parse_args(argv)
    result = run_profile_attempt(
        system=args.system,
        workload=args.workload,
        concurrency=args.concurrency,
        repeat=args.repeat,
        warmup=args.warmup,
        out_root=args.out_root,
        seed=args.seed,
        arrival_process=args.arrival_process,
        rounds=args.rounds,
    )
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
