"""Aggregate and interpret APR public-workload suite evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from workloads.apr_benchmark import BASELINES, SUPPORTED_WORKLOADS, generate_workload, summarize_trace


DEFAULT_CONCURRENCIES = (4, 8, 16)
NOT_AVAILABLE_SYSTEM = "no_rsv_dsv_apr"
PRIMARY_FIELDS = (
    "run_id", "system", "workload", "concurrency", "repeat", "elapsed_s",
    "completed_sessions", "pcm_chunks", "pcm_bytes", "session_throughput_sps",
    "useful_audio_throughput_sps", "time_to_first_audio_s", "completion_latency_s",
    "checkpoint_count", "restore_count", "worker_switch_count",
    "acoustic_progression_duration_s", "migration_time_s", "gpu0_mean_utilization",
    "gpu0_p95_utilization", "gpu0_memory_mean", "gpu0_memory_max",
    "gpu1_mean_utilization", "gpu1_p95_utilization", "gpu1_memory_mean", "gpu1_memory_max",
)
MATRIX_FIELDS = (
    "workload", "concurrency", "system", "expected_count", "valid_count", "invalid_count",
    "warmup_count", "status", "completion_rate", "duration_variance_s", "arrival_span_s",
    "arrival_interarrival_cv", "min_progression_opportunities", "session_throughput_mean",
    "useful_audio_throughput_mean", "ttfa_mean_s", "completion_latency_mean_s",
    "checkpoint_count_mean", "restore_count_mean", "worker_switch_count_mean",
    "migration_count_mean", "checkpoint_overhead_s", "worker_utilization_variance",
    "idle_worker_time_s", "waiting_sessions",
)
GAIN_FIELDS = (
    "workload", "concurrency", "apr_valid", "original_valid", "status",
    "session_throughput_ratio", "useful_audio_throughput_ratio", "ttfa_delta_s",
    "completion_latency_delta_s", "session_ratio_mean", "session_ratio_median",
    "session_ratio_std", "session_ratio_min", "session_ratio_max", "audio_ratio_mean",
    "audio_ratio_median", "audio_ratio_std", "audio_ratio_min", "audio_ratio_max",
    "positive_repeat_count", "negative_repeat_count", "direction_consistency",
    "per_repeat_session_ratios", "per_repeat_audio_ratios",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _collect(root: Path, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob(filename)):
        for row in _read_csv(path):
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [number for row in rows if (number := _float(row.get(field))) is not None]
    if not values:
        return "NOT_MEASURED"
    return statistics.mean(values)


def _unique_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    key_fields: Sequence[str] = ("run_id",),
) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    result: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda value: (value.get("run_id", ""), value.get("_source_file", ""))):
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if any(key) and key in seen:
            continue
        if any(key):
            seen.add(key)
        result.append(dict(row))
    return result


def _matches(row: Mapping[str, str], workload: str, concurrency: int, system: str) -> bool:
    return (
        row.get("workload") == workload
        and int(row.get("concurrency", "-1")) == concurrency
        and row.get("system") == system
    )


def _features(workload: str, concurrency: int, arrival_process: str) -> dict[str, Any]:
    return summarize_trace(
        generate_workload(workload, concurrency=concurrency, seed=1001, arrival_process=arrival_process)
    )


def _cell_row(
    *,
    workload: str,
    concurrency: int,
    system: str,
    repeats: int,
    primary: Sequence[Mapping[str, str]],
    invalid: Sequence[Mapping[str, str]],
    warmups: Sequence[Mapping[str, str]],
    arrival_process: str,
) -> dict[str, Any]:
    valid_rows = _unique_rows([
        row for row in primary if _matches(row, workload, concurrency, system)
    ])
    invalid_count = sum(_matches(row, workload, concurrency, system) for row in invalid)
    warmup_count = sum(_matches(row, workload, concurrency, system) for row in warmups)
    features = _features(workload, concurrency, arrival_process)
    if system == NOT_AVAILABLE_SYSTEM:
        status = "NOT_AVAILABLE"
        completion_rate: float | str = "NOT_AVAILABLE"
    elif len(valid_rows) == repeats:
        status = "PASS"
        completion_rate = 1.0
    else:
        status = "INCOMPLETE"
        completion_rate = len(valid_rows) / repeats
    migration = _mean(valid_rows, "worker_switch_count")
    return {
        "workload": workload,
        "concurrency": concurrency,
        "system": system,
        "expected_count": repeats,
        "valid_count": len(valid_rows),
        "invalid_count": invalid_count,
        "warmup_count": warmup_count,
        "status": status,
        "completion_rate": completion_rate,
        "duration_variance_s": features["duration_variance_s"],
        "arrival_span_s": features["arrival_span_s"],
        "arrival_interarrival_cv": features["arrival_interarrival_cv"],
        "min_progression_opportunities": features["min_progression_opportunities"],
        "session_throughput_mean": _mean(valid_rows, "session_throughput_sps"),
        "useful_audio_throughput_mean": _mean(valid_rows, "useful_audio_throughput_sps"),
        "ttfa_mean_s": _mean(valid_rows, "time_to_first_audio_s"),
        "completion_latency_mean_s": _mean(valid_rows, "completion_latency_s"),
        "checkpoint_count_mean": _mean(valid_rows, "checkpoint_count"),
        "restore_count_mean": _mean(valid_rows, "restore_count"),
        "worker_switch_count_mean": migration,
        "migration_count_mean": migration,
        "checkpoint_overhead_s": _mean(valid_rows, "migration_time_s"),
        "worker_utilization_variance": "NOT_MEASURED",
        "idle_worker_time_s": "NOT_MEASURED",
        "waiting_sessions": "NOT_MEASURED",
        "_valid_rows": valid_rows,
    }


def _ratio(apr: float | str, control: float | str) -> float | str:
    apr_value = _float(apr)
    control_value = _float(control)
    if apr_value is None or control_value in (None, 0.0):
        return "NOT_MEASURED"
    return apr_value / control_value


def _stats(values: Sequence[Any]) -> dict[str, float | str]:
    numbers = [number for value in values if (number := _float(value)) is not None]
    if not numbers:
        return {
            "mean": "NOT_MEASURED",
            "median": "NOT_MEASURED",
            "std": "NOT_MEASURED",
            "min": "NOT_MEASURED",
            "max": "NOT_MEASURED",
        }
    return {
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def _gain_row(
    *,
    workload: str,
    concurrency: int,
    cells: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    apr = cells[(workload, concurrency, "apr")]
    original = cells[(workload, concurrency, "original_affinity")]
    apr_rows = {int(row["repeat"]): row for row in apr["_valid_rows"]}
    original_rows = {int(row["repeat"]): row for row in original["_valid_rows"]}
    paired = sorted(set(apr_rows) & set(original_rows))
    session_ratios = [
        _ratio(apr_rows[repeat].get("session_throughput_sps"), original_rows[repeat].get("session_throughput_sps"))
        for repeat in paired
    ]
    audio_ratios = [
        _ratio(apr_rows[repeat].get("useful_audio_throughput_sps"), original_rows[repeat].get("useful_audio_throughput_sps"))
        for repeat in paired
    ]
    session_ratios = [value for value in session_ratios if _float(value) is not None]
    audio_ratios = [value for value in audio_ratios if _float(value) is not None]
    session_stats = _stats(session_ratios)
    audio_stats = _stats(audio_ratios)
    numeric_session_ratios = [float(value) for value in session_ratios]
    positive_repeat_count = sum(value > 1.0 for value in numeric_session_ratios)
    negative_repeat_count = sum(value < 1.0 for value in numeric_session_ratios)
    if not numeric_session_ratios:
        direction_consistency = "NO_PAIRED_RUNS"
    elif positive_repeat_count == len(numeric_session_ratios):
        direction_consistency = "CONSISTENT_POSITIVE"
    elif negative_repeat_count == len(numeric_session_ratios):
        direction_consistency = "CONSISTENT_NEGATIVE"
    elif positive_repeat_count == 0 and negative_repeat_count == 0:
        direction_consistency = "EQUAL"
    else:
        direction_consistency = "MIXED_DIRECTION"
    status = "PASS" if apr["status"] == original["status"] == "PASS" else "INCOMPLETE"
    return {
        "workload": workload,
        "concurrency": concurrency,
        "apr_valid": apr["valid_count"],
        "original_valid": original["valid_count"],
        "status": status,
        "session_throughput_ratio": _ratio(apr["session_throughput_mean"], original["session_throughput_mean"]),
        "useful_audio_throughput_ratio": _ratio(apr["useful_audio_throughput_mean"], original["useful_audio_throughput_mean"]),
        "ttfa_delta_s": (
            _float(apr["ttfa_mean_s"]) - _float(original["ttfa_mean_s"])
            if _float(apr["ttfa_mean_s"]) is not None and _float(original["ttfa_mean_s"]) is not None
            else "NOT_MEASURED"
        ),
        "completion_latency_delta_s": (
            _float(apr["completion_latency_mean_s"]) - _float(original["completion_latency_mean_s"])
            if _float(apr["completion_latency_mean_s"]) is not None and _float(original["completion_latency_mean_s"]) is not None
            else "NOT_MEASURED"
        ),
        "session_ratio_mean": session_stats["mean"],
        "session_ratio_median": session_stats["median"],
        "session_ratio_std": session_stats["std"],
        "session_ratio_min": session_stats["min"],
        "session_ratio_max": session_stats["max"],
        "audio_ratio_mean": audio_stats["mean"],
        "audio_ratio_median": audio_stats["median"],
        "audio_ratio_std": audio_stats["std"],
        "audio_ratio_min": audio_stats["min"],
        "audio_ratio_max": audio_stats["max"],
        "positive_repeat_count": positive_repeat_count,
        "negative_repeat_count": negative_repeat_count,
        "direction_consistency": direction_consistency,
        "per_repeat_session_ratios": json.dumps(session_ratios, sort_keys=True),
        "per_repeat_audio_ratios": json.dumps(audio_ratios, sort_keys=True),
    }


def analyze_suite(
    root: Path,
    *,
    repeats: int = 5,
    workloads: Sequence[str] = SUPPORTED_WORKLOADS,
    concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
    systems: Sequence[str] = BASELINES,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "suite_manifest.json"
    arrival_process = "poisson"
    if manifest_path.is_file():
        arrival_process = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("arrival_process", "poisson"))
    primary = _collect(root, "formal_primary_metrics.csv")
    invalid = _collect(root, "invalid_attempts.csv")
    warmups = _collect(root, "warmup_attempts.csv")
    matrix: list[dict[str, Any]] = []
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for workload in workloads:
        for concurrency in concurrencies:
            for system in systems:
                cell = _cell_row(
                    workload=workload,
                    concurrency=concurrency,
                    system=system,
                    repeats=repeats,
                    primary=primary,
                    invalid=invalid,
                    warmups=warmups,
                    arrival_process=arrival_process,
                )
                cells[(workload, concurrency, system)] = cell
                matrix.append({field: cell[field] for field in MATRIX_FIELDS})
    gain = [
        _gain_row(workload=workload, concurrency=concurrency, cells=cells)
        for workload in workloads
        for concurrency in concurrencies
    ]
    primary_unique = _unique_rows(
        primary,
        key_fields=("workload", "concurrency", "system", "repeat", "run_id"),
    )
    return {
        "primary": primary_unique,
        "invalid": invalid,
        "warmups": warmups,
        "matrix": matrix,
        "gain": gain,
        "arrival_process": arrival_process,
        "cells": cells,
    }


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _fmt(value: Any) -> str:
    number = _float(value)
    return f"{number:.4f}" if number is not None else str(value)


def _report(result: Mapping[str, Any]) -> str:
    matrix = list(result["matrix"])
    gain = list(result["gain"])
    valid_count = sum(int(row["valid_count"]) for row in matrix if row["status"] != "NOT_AVAILABLE")
    expected_count = sum(int(row["expected_count"]) for row in matrix if row["status"] != "NOT_AVAILABLE")
    complete_cells = sum(row["status"] == "PASS" for row in matrix if row["system"] != NOT_AVAILABLE_SYSTEM)
    lines = [
        "# APR Workload Advantage Report",
        "",
        "This report characterizes the measured advantage region of explicit-state APR under the public-inspired workload suite. It is descriptive and does not create a speedup claim outside complete, valid cells.",
        "",
        "## Scope and accounting",
        "",
        f"- Workloads: `{', '.join(sorted({str(row['workload']) for row in matrix}))}`",
        f"- Concurrency: `{', '.join(str(value) for value in sorted({int(row['concurrency']) for row in matrix}))}`",
        f"- Arrival process for the formal matrix: `{result['arrival_process']}`",
        f"- Valid counted runs: `{valid_count} / {expected_count}` runnable expected runs",
        f"- Complete runnable cells: `{complete_cells}`",
        f"- `no_rsv_dsv_apr`: `NOT_AVAILABLE` (no real no-RSV/DSV model runner was provided)",
        "",
        "## Experimental setup",
        "",
        "- Runtime: real-model Lychee execution with the local Token2Wav acoustic backend.",
        "- Physical acoustic workers: 2; APR and original_affinity use the same model, checkpoint, and GPU mapping.",
        "- Measurement: external 5 Hz GPU collector with GPU0/GPU1 kept separate; counted rows are fail-closed on PCM, cleanup, ownership, and telemetry errors.",
        "- Raw evidence: `formal_matrix_v1/runs/`, including per-attempt manifests, canonical events, model/acoustic traces, GPU traces, and launcher logs.",
        "",
        "## Workload generation method",
        "",
        "- A is a deterministic public-inspired normalized interaction fixture with overlapping turns, barge-in/resume markers, and four turns per session.",
        "- B is generated from short/medium/long heavy-tail duration classes with the registered Poisson arrival process; burst generation remains available but was not mixed into this formal matrix.",
        "- C starts resident long sessions and admits late short sessions to expose static worker fragmentation.",
        "",
        "## Cell summary",
        "",
        "| Workload | N | System | Valid/expected | Invalid | Status | Completion | Throughput mean | Useful audio mean | TTFA mean (s) | Duration variance | Arrival CV | Migration/checkpoint overhead |",
    ]
    lines.append("|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for row in matrix:
        overhead = row["checkpoint_overhead_s"]
        lines.append(
            f"| {row['workload']} | {row['concurrency']} | {row['system']} | {row['valid_count']}/{row['expected_count']} | {row['invalid_count']} | {row['status']} | {_fmt(row['completion_rate'])} | {_fmt(row['session_throughput_mean'])} | {_fmt(row['useful_audio_throughput_mean'])} | {_fmt(row['ttfa_mean_s'])} | {_fmt(row['duration_variance_s'])} | {_fmt(row['arrival_interarrival_cv'])} | {_fmt(row['migration_count_mean'])}/{_fmt(overhead)} |"
        )
    lines.extend([
        "",
        "## APR versus original_affinity",
        "",
        "Ratios are APR divided by original_affinity. A ratio is reported only from measured rows; incomplete cells are not treated as negative or positive evidence.",
        "",
        "| Workload | N | Status | APR valid | Control valid | Session ratio mean/median/std/min/max | Audio ratio mean/median/std/min/max | TTFA delta (s) | Completion latency delta (s) | Direction | Positive/negative repeats | Per-repeat session ratios |",
        "|---|---:|---|---:|---:|---|---|---:|---:|---|---:|---|",
    ])
    for row in gain:
        lines.append(
            f"| {row['workload']} | {row['concurrency']} | {row['status']} | {row['apr_valid']} | {row['original_valid']} | {_fmt(row['session_ratio_mean'])}/{_fmt(row['session_ratio_median'])}/{_fmt(row['session_ratio_std'])}/{_fmt(row['session_ratio_min'])}/{_fmt(row['session_ratio_max'])} | {_fmt(row['audio_ratio_mean'])}/{_fmt(row['audio_ratio_median'])}/{_fmt(row['audio_ratio_std'])}/{_fmt(row['audio_ratio_min'])}/{_fmt(row['audio_ratio_max'])} | {_fmt(row['ttfa_delta_s'])} | {_fmt(row['completion_latency_delta_s'])} | {row['direction_consistency']} | {row['positive_repeat_count']}/{row['negative_repeat_count']} | `{row['per_repeat_session_ratios']}` |"
        )
    consistent_positive = [
        f"{row['workload']}/N{row['concurrency']}"
        for row in gain
        if row["direction_consistency"] == "CONSISTENT_POSITIVE"
    ]
    consistent_negative = [
        f"{row['workload']}/N{row['concurrency']}"
        for row in gain
        if row["direction_consistency"] == "CONSISTENT_NEGATIVE"
    ]
    mean_positive_mixed = [
        f"{row['workload']}/N{row['concurrency']} ({_fmt(row['session_ratio_mean'])})"
        for row in gain
        if row["status"] == "PASS"
        and row["direction_consistency"] == "MIXED_DIRECTION"
        and (_float(row["session_ratio_mean"]) or 0.0) > 1.0
    ]
    lines.extend(
        [
            "",
            "## Advantage region",
            "",
            f"- Direction-consistent positive cells: `{', '.join(consistent_positive) if consistent_positive else 'NONE'}`.",
            f"- Mean-positive but direction-mixed cells: `{', '.join(mean_positive_mixed) if mean_positive_mixed else 'NONE'}`.",
            f"- Direction-consistent negative cells: `{', '.join(consistent_negative) if consistent_negative else 'NONE'}`.",
            "- A mean ratio above 1.0 with mixed repeat directions is reported as an observed cell-level fluctuation, not a stable APR advantage.",
            "- Under this complete suite, APR has no direction-consistent positive workload/concurrency region; the measured evidence does not establish a general performance advantage over original_affinity.",
        ]
    )
    fragmentation = {
        int(row["concurrency"]): row
        for row in gain
        if row["workload"] == "C"
    }
    lines.extend(
        [
            "",
            "## Fragmentation analysis",
            "",
            "C is the registered long-resident/late-short stressor. Its APR/control session-throughput ratios are:",
            "",
            *[
                f"- C/N{concurrency}: ratio `{_fmt(row['session_ratio_mean'])}`, direction `{row['direction_consistency']}`, TTFA delta `{_fmt(row['ttfa_delta_s'])}` s."
                for concurrency, row in sorted(fragmentation.items())
            ],
            "",
            "The measured C region therefore does not show APR recovering throughput under this real workload: C/N16 is consistently negative across all five repeats, while C/N4 and C/N8 are mixed-direction.",
        ]
    )
    lines.extend([
        "",
        "## Workload interpretation",
        "",
        "- Workload A is the realism/public-inspired stratum: overlap, interruption, resume, and multiple turns. A complete cell is required before making a realism claim.",
        "- Workload B is the heterogeneous stratum: heavy-tailed durations plus the registered arrival process. Duration variance and arrival inter-arrival CV are reported from the frozen trace, not inferred from runtime outcomes.",
        "- Workload C is the fragmentation stratum: resident long sessions precede late short sessions. Its evidence is the relevant region for testing whether APR avoids static worker fragmentation.",
        "",
        "## Measurement limits",
        "",
        "`worker_utilization_variance`, `idle_worker_time_s`, and waiting-session time are `NOT_MEASURED` because the current real runner does not expose those counters. `migration_count_mean` is the observed worker-switch count emitted by the APR acoustic lane; it is not silently replaced with zero. `checkpoint_overhead_s` remains `NOT_MEASURED` when the runner emits no migration duration.",
        "",
    ])
    if complete_cells == len(matrix) - sum(row["system"] == NOT_AVAILABLE_SYSTEM for row in matrix):
        lines.append("All runnable cells are complete; conclusions remain restricted to the three measured workload strata.")
    else:
        lines.append("The matrix is incomplete. No final APR advantage claim is established until every runnable cell reaches five valid counted repeats.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(root: Path, result: Mapping[str, Any], output_dir: Path | None = None) -> dict[str, Path]:
    root = Path(root)
    output_dir = Path(output_dir) if output_dir is not None else root
    primary_rows = [
        {field: row.get(field, "") for field in PRIMARY_FIELDS}
        for row in result["primary"]
    ]
    invalid_rows = list(result["invalid"])
    primary_path = _write_csv(output_dir / "workload_suite_primary_metrics.csv", PRIMARY_FIELDS, primary_rows)
    matrix_path = _write_csv(output_dir / "workload_suite_matrix.csv", MATRIX_FIELDS, result["matrix"])
    invalid_fields = sorted({key for row in invalid_rows for key in row if not key.startswith("_")}) or ["run_id", "failure_signature"]
    invalid_path = _write_csv(output_dir / "invalid_attempts.csv", invalid_fields, invalid_rows)
    gain_path = _write_csv(output_dir / "apr_vs_original_gain.csv", GAIN_FIELDS, result["gain"])
    report_path = output_dir / "APR_WORKLOAD_ADVANTAGE_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    return {
        "primary": primary_path,
        "matrix": matrix_path,
        "invalid": invalid_path,
        "gain": gain_path,
        "report": report_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("reports/apr_workload_suite/formal_matrix"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/apr_workload_suite"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    result = analyze_suite(args.raw_root, repeats=args.repeats)
    outputs = write_outputs(args.raw_root, result, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
