#!/usr/bin/env python3
"""Analyze APR profile spans without silently fabricating unavailable metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

PROFILE_REQUIRED_FOR_APR = {"STATE_RESTORE"}
MIGRATION_STAGES = {"STATE_ACQUIRE", "STATE_RESTORE", "STATE_CAPTURE", "STATE_COMMIT"}


def _duration(record: Mapping[str, Any]) -> int | None:
    value = record.get("duration_ns")
    if value in (None, ""):
        start = record.get("start_monotonic_ns")
        end = record.get("end_monotonic_ns")
        if start in (None, "") or end in (None, ""):
            return None
        value = int(end) - int(start)
    value = int(value)
    if value < 0:
        return None
    return value


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _stage_stats(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    durations = [duration for record in records if (duration := _duration(record)) is not None]
    return {
        "count": len(durations),
        "total_ns": sum(durations) if durations else None,
        "mean_ns": float(mean(durations)) if durations else None,
        "p50_ns": _percentile(durations, 0.50),
        "p95_ns": _percentile(durations, 0.95),
    }


def _gpu_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {"GPU0": [], "GPU1": []}
    for record in records:
        raw_id = str(record.get("gpu_id", "")).strip()
        key = f"GPU{raw_id}" if raw_id in {"0", "1"} else raw_id
        if key in grouped:
            grouped[key].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        utilizations = [float(row["gpu_utilization"]) for row in rows if row.get("gpu_utilization") not in (None, "")]
        memories = [float(row["memory_used"]) for row in rows if row.get("memory_used") not in (None, "")]
        summary[key] = {
            "sample_count": len(rows),
            "mean_utilization": float(mean(utilizations)) if utilizations else None,
            "p95_utilization": _percentile([int(value) for value in utilizations], 0.95) if utilizations else None,
            "max_memory_used": max(memories) if memories else None,
        }
    return summary


def analyze_profile_records(
    records: Iterable[Mapping[str, Any]],
    *,
    gpu_records: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    records = [dict(record) for record in records]
    grouped: dict[str, list[dict[str, Any]]] = {}
    invalid_run_ids: list[str] = []
    for record in records:
        run_id = str(record.get("run_id", "")).strip()
        if not run_id:
            continue
        grouped.setdefault(run_id, []).append(record)

    valid_run_ids: list[str] = []
    valid_records: list[dict[str, Any]] = []
    for run_id, run_records in grouped.items():
        system = str(run_records[0].get("system", ""))
        stages = {str(record.get("event_type", "")) for record in run_records}
        required = PROFILE_REQUIRED_FOR_APR if system == "apr" else set()
        if required.difference(stages):
            invalid_run_ids.append(run_id)
            continue
        if any(_duration(record) is None for record in run_records):
            invalid_run_ids.append(run_id)
            continue
        valid_run_ids.append(run_id)
        valid_records.extend(run_records)

    by_stage: dict[str, list[Mapping[str, Any]]] = {}
    for record in valid_records:
        by_stage.setdefault(str(record.get("event_type", "")), []).append(record)
    stages = {stage: _stage_stats(rows) for stage, rows in sorted(by_stage.items())}
    migration_records = [record for record in valid_records if record.get("event_type") in MIGRATION_STAGES]
    migration_latency = sum(_duration(record) or 0 for record in migration_records) if migration_records else None
    queue_records = [record for record in valid_records if record.get("event_type") == "APR_SCHEDULE_WAIT"]
    queue_wait = sum(_duration(record) or 0 for record in queue_records) if queue_records else None
    residency: dict[str, int] = {}
    for record in valid_records:
        worker_id = record.get("worker_id")
        duration = _duration(record)
        if worker_id in (None, "") or duration is None:
            continue
        key = str(worker_id)
        residency[key] = residency.get(key, 0) + duration
    total_duration = sum(_duration(record) or 0 for record in valid_records) if valid_records else 0
    backend_duration = sum(
        _duration(record) or 0
        for record in valid_records
        if record.get("event_type") == "BACKEND_PROCESS"
    )
    backend_share = float(backend_duration / total_duration) if total_duration else None
    return {
        "valid_run_ids": sorted(valid_run_ids),
        "invalid_run_ids": sorted(invalid_run_ids),
        "aggregated_run_ids": sorted(valid_run_ids),
        "stages": stages,
        "migration": {
            "count": len([record for record in valid_records if record.get("event_type") == "STATE_RESTORE"]),
            "latency_ns": migration_latency,
        },
        "scheduler": {"queue_wait_ns": queue_wait},
        "worker_residency_ns": residency,
        "backend_share": backend_share,
        "gpu": _gpu_summary(gpu_records),
        "optimization_decision": {
            "selected_candidate": "none",
            "reason": "insufficient independent evidence categories to select one evidence-supported optimization",
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_profile_reports(report: Mapping[str, Any], out_root: Path) -> None:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "profile_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "valid", "invalid_reason"])
        for run_id in report.get("valid_run_ids", []):
            writer.writerow([run_id, "PASS", ""])
        for run_id in report.get("invalid_run_ids", []):
            writer.writerow([run_id, "INVALID_RUN", "missing_or_nonfinite_span"])
    migration = report.get("migration", {})
    with (out_root / "APR_MIGRATION_COST_ANALYSIS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["migration_count", "migration_latency_ns", "queue_wait_ns", "backend_share"])
        writer.writerow([
            migration.get("count", ""),
            migration.get("latency_ns", ""),
            report.get("scheduler", {}).get("queue_wait_ns", ""),
            report.get("backend_share", ""),
        ])
    (out_root / "APR_PIPELINE_PROFILE_REPORT.md").write_text(
        "# APR Pipeline Profile Report\n\n"
        f"Valid runs: {len(report.get('valid_run_ids', []))}.\n\n"
        f"Invalid runs: {len(report.get('invalid_run_ids', []))}.\n\n"
        "Missing telemetry remains unavailable; no zero-filled inference is made.\n",
        encoding="utf-8",
    )
    (out_root / "APR_SCHEDULER_ANALYSIS.md").write_text(
        "# APR Scheduler Analysis\n\n"
        f"Queue wait (ns): {report.get('scheduler', {}).get('queue_wait_ns')}.\n",
        encoding="utf-8",
    )
    (out_root / "TOKEN2WAV_OPTIMIZATION_REPORT.md").write_text(
        "# Token2Wav Optimization Report\n\n"
        "No kernel/runtime optimization is selected by this analyzer alone.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("reports/apr_optimization"))
    args = parser.parse_args(argv)
    gpu_records = []
    if args.gpu is not None and args.gpu.is_file():
        with args.gpu.open(newline="", encoding="utf-8") as handle:
            gpu_records = list(csv.DictReader(handle))
    report = analyze_profile_records(load_jsonl(args.spans), gpu_records=gpu_records)
    write_profile_reports(report, args.out_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
