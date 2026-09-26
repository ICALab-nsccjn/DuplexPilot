#!/usr/bin/env python3
"""Run one isolated APR optimization profile attempt."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import uuid
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from lychee_fd.runtime.apr.profiling import PROFILE_SCHEMA_VERSION, StageProfiler
from tools.apr.real_e2e_config import RealE2EConfig, build_profile_manifest
from tools.apr.real_e2e_runner import RealModelAcousticHandoff
from tools.apr.run_real_apr_e2e import (
    build_real_components,
    run_real_correctness_attempt,
)
from workloads.apr_benchmark import generate_workload, trace_hash


PROFILE_SYSTEMS = ("original_affinity", "apr")
PROFILE_WORKLOADS = ("A", "B", "C")
SPAN_FIELDS = (
    "profile_schema_version",
    "run_id",
    "system",
    "workload",
    "concurrency",
    "repeat",
    "session_id",
    "generation_id",
    "sequence_no",
    "state_version",
    "worker_id",
    "event_type",
    "start_monotonic_ns",
    "end_monotonic_ns",
    "duration_ns",
    "queue_depth",
    "error_type",
)


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _write_span_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SPAN_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in SPAN_FIELDS})


def _split_profile_spans(attempt_dir: Path) -> None:
    spans_path = attempt_dir / "pipeline_spans.jsonl"
    records = [
        json.loads(line)
        for line in spans_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _write_span_csv(
        attempt_dir / "scheduler_spans.csv",
        (record for record in records if record.get("event_type") == "APR_SCHEDULE_WAIT"),
    )
    migration_stages = {
        "STATE_ACQUIRE",
        "STATE_RESTORE",
        "STATE_CAPTURE",
        "STATE_COMMIT",
    }
    _write_span_csv(
        attempt_dir / "apr_migration_spans.csv",
        (record for record in records if record.get("event_type") in migration_stages),
    )


def run_profile_attempt(
    *,
    config: RealE2EConfig,
    system: str,
    workload: str,
    concurrency: int,
    repeat: int,
    warmup: bool,
    out_root: Path | None = None,
    seed: int = 0,
    arrival_process: str = "poisson",
    rounds: int = 4,
) -> dict[str, Any]:
    """Execute exactly one isolated profile attempt and retain raw evidence."""
    if system not in PROFILE_SYSTEMS:
        raise ValueError(f"profile system must be one of {PROFILE_SYSTEMS}")
    if workload not in PROFILE_WORKLOADS:
        raise ValueError(f"profile workload must be one of {PROFILE_WORKLOADS}")
    if concurrency <= 0 or repeat <= 0:
        raise ValueError("concurrency and repeat must be positive")

    workload_trace = generate_workload(
        workload,
        concurrency=concurrency,
        seed=seed,
        arrival_process=arrival_process,
    )
    run_id = (
        f"apr-profile-{system}-n{concurrency}-r{repeat}-"
        f"{'warmup' if warmup else 'counted'}-{uuid.uuid4().hex[:12]}"
    )
    raw_root = Path(out_root or REPO_ROOT / "reports" / "apr_optimization" / "raw")
    run_root = raw_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    manifest = build_profile_manifest(
        config,
        run_id=run_id,
        system=system,
        workload=workload,
        concurrency=concurrency,
        repeat=repeat,
        warmup=warmup,
        source_commit=_source_commit(),
        workload_trace_hash=trace_hash(workload_trace),
        profiling_enabled=True,
        torch_profiler_enabled=False,
        nsight_enabled=False,
    )
    (run_root / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    profiler = StageProfiler(
        enabled=True,
        run_context={
            "run_id": run_id,
            "system": system,
            "workload": workload,
            "concurrency": concurrency,
            "repeat": repeat,
        },
    )
    config.validate_paths()
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
        workload_trace=workload_trace,
        profiler=profiler,
        run_id=run_id,
    )
    attempt_dir = run_root / "attempts" / system / f"N{concurrency}" / (
        f"warmup-r{repeat}" if warmup else f"r{repeat}"
    )
    (attempt_dir / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _split_profile_spans(attempt_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=PROFILE_SYSTEMS, required=True)
    parser.add_argument("--workload", choices=PROFILE_WORKLOADS, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-process", choices=("poisson", "burst"), default="poisson")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "reports" / "apr_optimization" / "raw",
    )
    args = parser.parse_args(argv)
    config = RealE2EConfig.from_env()
    result = run_profile_attempt(
        config=config,
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
    print(json.dumps({"run_id": result["run_id"], "valid": result["valid"]}, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
