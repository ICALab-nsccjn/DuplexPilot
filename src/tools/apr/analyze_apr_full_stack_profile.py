#!/usr/bin/env python3
"""Aggregate profiling-only APR evidence without promoting performance claims."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import shutil
from typing import Any, Iterable


MIGRATION_STAGES = ("STATE_ACQUIRE", "STATE_RESTORE", "STATE_CAPTURE", "STATE_COMMIT")
ACOUSTIC_STAGES = (
    "STREAMING_DECODER",
    "TOKEN2WAV_TOKEN_QUEUE",
    "TOKEN2WAV_FLOW",
    "TOKEN2WAV_HIFT",
    "TOKEN2WAV_VOCODER",
    "PCM_GENERATION",
)
REPORT_NAMES = (
    "APR_FULL_STACK_PROFILE_REPORT.md",
    "APR_MIGRATION_PROFILE.md",
    "ACOUSTIC_BACKEND_PROFILE.md",
    "CUDA_PROFILING_GAP_REPORT.md",
    "APR_BOTTLENECK_DECISION.md",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _timeline(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def _p95(values: Iterable[float]) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * 0.95)]


def _stage_table(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    state_sizes: dict[str, list[int]] = defaultdict(list)
    for row in records:
        stage = str(row.get("event_type", row.get("stage", "")))
        try:
            duration = float(row.get("duration_ns", -1))
        except (TypeError, ValueError):
            duration = -1
        if stage and duration >= 0:
            grouped[stage].append(duration)
        if row.get("state_size_bytes") not in (None, ""):
            try:
                state_sizes[stage].append(int(row["state_size_bytes"]))
            except (TypeError, ValueError):
                pass
    return {
        stage: {
            "count": len(values),
            "total_ns": sum(values),
            "mean_ns": mean(values) if values else None,
            "p95_ns": _p95(values),
            "state_size_mean": mean(state_sizes.get(stage, ())) if state_sizes.get(stage) else None,
            "state_size_max": max(state_sizes.get(stage, ())) if state_sizes.get(stage) else None,
        }
        for stage, values in sorted(grouped.items())
    }


def summarize_root(root: Path) -> dict[str, Any]:
    root = Path(root)
    attempts = sorted(root.rglob("attempt_result.json"))
    all_spans: list[dict[str, Any]] = []
    all_timeline: list[dict[str, Any]] = []
    spans_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runs: list[dict[str, Any]] = []
    for result_path in attempts:
        attempt_dir = result_path.parent
        result = json.loads(result_path.read_text(encoding="utf-8"))
        spans = _jsonl(attempt_dir / "pipeline_spans.jsonl")
        timeline = _timeline(attempt_dir / "apr_full_timeline.json")
        metadata_path = attempt_dir / "profiling_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        profile_valid = bool(metadata.get("profile_valid", (attempt_dir / "apr_full_timeline.json").is_file()))
        run_valid = bool(result.get("valid")) and profile_valid
        if run_valid:
            all_spans.extend(spans)
            all_timeline.extend(timeline)
            system_name = str(result.get("system", metadata.get("system", "unknown")))
            spans_by_system[system_name].extend(spans)
        runs.append({
            "path": str(attempt_dir),
            "system": result.get("system", metadata.get("system", "unknown")),
            "concurrency": result.get("concurrency", metadata.get("concurrency")),
            "valid": run_valid,
            "elapsed_s": result.get("elapsed_s"),
            "timeline_events": len(timeline),
            "span_events": len(spans),
        })
    stage_table = _stage_table(all_spans)
    observed = {stage for stage in ACOUSTIC_STAGES if stage in stage_table}
    migration_present = any(stage in stage_table for stage in MIGRATION_STAGES)
    acoustic_present = bool(observed)
    attribution_complete = all(stage in observed for stage in ("TOKEN2WAV_FLOW", "TOKEN2WAV_HIFT", "TOKEN2WAV_VOCODER"))
    total_elapsed_s = sum(float(run["elapsed_s"]) for run in runs if run.get("elapsed_s") is not None)
    migration_total_ns = sum(stage_table.get(stage, {}).get("total_ns", 0.0) for stage in MIGRATION_STAGES)
    decision = "NO_SAFE_OPTIMIZATION_SELECTED"
    if attribution_complete and total_elapsed_s > 0 and migration_total_ns / (total_elapsed_s * 1e9) > 0.10:
        decision = "CHECKPOINT_OPTIMIZATION_CANDIDATE"
    elif attribution_complete and stage_table.get("TOKEN2WAV_FLOW", {}).get("total_ns", 0) > stage_table.get("MODEL_FORWARD", {}).get("total_ns", 0):
        decision = "ACOUSTIC_BACKEND_CANDIDATE"
    return {
        "root": str(root),
        "runs": runs,
        "run_count": len(runs),
        "valid_runs": sum(run["valid"] for run in runs),
        "span_count": len(all_spans),
        "timeline_count": len(all_timeline),
        "timeline_events": Counter(row.get("event") for row in all_timeline),
        "stage_table": stage_table,
        "system_stage_tables": {system: _stage_table(rows) for system, rows in sorted(spans_by_system.items())},
        "migration_present": migration_present,
        "acoustic_present": acoustic_present,
        "attribution_complete": attribution_complete,
        "total_elapsed_s": total_elapsed_s,
        "migration_total_ns": migration_total_ns,
        "decision": decision,
        "nsight": bool(shutil.which("nsys") or shutil.which("nsight-sys")),
    }


def _ms(value: float | None) -> str:
    return "NOT_MEASURED" if value is None else f"{value / 1e6:.3f}"


def _stage_markdown(summary: dict[str, Any], names: Iterable[str] | None = None) -> list[str]:
    table = summary["stage_table"]
    selected = list(names) if names is not None else sorted(table)
    lines = ["| stage | count | total ms | mean ms | p95 ms |", "|---|---:|---:|---:|---:|"]
    for stage in selected:
        row = table.get(stage, {})
        lines.append(
            f"| `{stage}` | {row.get('count', 0)} | {_ms(row.get('total_ns'))} | {_ms(row.get('mean_ns'))} | {_ms(row.get('p95_ns'))} |"
        )
    return lines


def write_reports(root: Path, output_dir: Path | None = None) -> list[Path]:
    summary = summarize_root(Path(root))
    output = Path(output_dir or root)
    output.mkdir(parents=True, exist_ok=True)
    run_lines = [
        "| path | system | N | valid | timeline events | spans |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        run_lines.append(
            f"| `{run['path']}` | `{run['system']}` | {run['concurrency']} | {run['valid']} | {run['timeline_events']} | {run['span_events']} |"
        )
    common = [
        "This artifact is diagnostic-only. It is not a formal throughput, speedup, or latency aggregate.",
        "Raw attempts remain under the input root; invalid attempts are not silently promoted.",
    ]
    full = [
        "# APR Full-Stack Profile Report",
        "",
        *common,
        "",
        f"- Runs observed: `{summary['run_count']}`; valid: `{summary['valid_runs']}`",
        f"- Timeline events: `{summary['timeline_count']}`; pipeline spans: `{summary['span_count']}`",
        f"- Acoustic attribution complete: `{summary['attribution_complete']}`",
        "",
        "## Run coverage",
        "",
        *run_lines,
        "",
        "## Stage coverage",
        "",
        *_stage_markdown(summary),
        "",
        "## APR vs original-affinity diagnostic breakdown",
        "",
        "| system | model ms | acoustic PCM ms | migration ms | scheduler-wait ms | idle time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system, table in sorted(summary["system_stage_tables"].items()):
        model_ns = table.get("MODEL_FORWARD", {}).get("total_ns", 0.0)
        pcm_ns = table.get("PCM_GENERATION", {}).get("total_ns", 0.0)
        migration_ns = sum(table.get(stage, {}).get("total_ns", 0.0) for stage in MIGRATION_STAGES)
        wait_ns = table.get("APR_SCHEDULE_WAIT", {}).get("total_ns", 0.0)
        full.append(
            f"| `{system}` | {_ms(model_ns)} | {_ms(pcm_ns)} | {_ms(migration_ns)} | {_ms(wait_ns)} | NOT_MEASURED |"
        )
    full += [
        "",
        "## Interpretation boundary",
        "",
        "N=8 APR/original-affinity evidence, if present, is diagnostic side-by-side evidence only. No performance claim is selected from this report without complete acoustic attribution and valid-run review.",
    ]
    migration_fraction = (summary["migration_total_ns"] / (summary["total_elapsed_s"] * 1e9)) if summary["total_elapsed_s"] else None
    migration = [
        "# APR Migration Profile",
        "",
        *common,
        "",
        f"- Migration stages observed: `{summary['migration_present']}`",
        f"- Migration wall time: `{_ms(summary['migration_total_ns'])} ms`",
        f"- E2E migration share: `{migration_fraction * 100:.3f}%`" if migration_fraction is not None else "- E2E migration share: `NOT_MEASURED`",
        f"- Migration frequency: {summary['stage_table'].get('STATE_RESTORE', {}).get('count', 0)} restore spans / {summary['valid_runs']} valid attempts",
        "- Serialization latency: `NOT_MEASURED_AS_A_SEPARATE_RUNTIME_STAGE`; the frozen runtime exposes capture/restore spans but no independent serialization boundary.",
        f"- Captured state size mean/max: `{summary['stage_table'].get('STATE_CAPTURE', {}).get('state_size_mean', 'NOT_MEASURED')}` / `{summary['stage_table'].get('STATE_CAPTURE', {}).get('state_size_max', 'NOT_MEASURED')}` bytes.",
        "",
        "## Migration stages",
        "",
        *_stage_markdown(summary, MIGRATION_STAGES),
    ]
    acoustic = [
        "# Acoustic Backend Profile",
        "",
        *common,
        "",
        f"- Detailed acoustic stage coverage: `{summary['acoustic_present']}`",
        f"- Required Flow/HiFT/vocoder attribution complete: `{summary['attribution_complete']}`",
        "",
        "## Acoustic stages",
        "",
        *_stage_markdown(summary, ACOUSTIC_STAGES),
        "",
        "StreamingDecoder spans are only present when the real runtime constructs it with the profiling sink; absence is reported as missing attribution, not zero cost.",
    ]
    nsight = shutil.which("nsys") or shutil.which("nsight-sys")
    cuda = [
        "# CUDA Profiling Gap Report",
        "",
        *common,
        "",
        f"- Nsight Systems executable: `{nsight or 'NOT_AVAILABLE'}`",
        "- Missing attribution when Nsight is unavailable: CUDA kernel/operator boundaries, stream concurrency, GPU-side queue wait, and exact Flow/HiFT/vocoder kernel breakdown.",
        "- Available alternatives: `torch.profiler` with CUDA activities for bounded model/acoustic traces, and optional NVTX ranges for future Nsight collection.",
        "- NVTX markers added in failure-safe mode: `APR_CHECKPOINT`, `APR_RESTORE`, `STREAMING_DECODER`, `TOKEN2WAV_FLOW`, `TOKEN2WAV_HIFT`, `VOCODER`.",
        "- No Nsight-dependent optimization claim is admitted.",
    ]
    bottleneck = [
        "# APR Bottleneck Decision",
        "",
        *common,
        "",
        f"## Decision: `{summary['decision']}`",
        "",
        f"- Acoustic attribution complete: `{summary['attribution_complete']}`",
        f"- Migration evidence present: `{summary['migration_present']}`",
        f"- Nsight available: `{summary['nsight']}`",
        "",
        "The optimization gate remains closed unless migration, acoustic sub-stages, and runtime validity are all independently evidenced. Missing attribution is not classified as a bottleneck.",
    ]
    contents = [full, migration, acoustic, cuda, bottleneck]
    paths = []
    for name, lines in zip(REPORT_NAMES, contents):
        path = output / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    summary = summarize_root(args.root)
    write_reports(args.root, args.output_dir)
    print(json.dumps({"runs": summary["run_count"], "decision": summary["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
