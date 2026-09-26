#!/usr/bin/env python3
"""Production-model N=2 correctness bring-up for APR local acoustic lanes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RSV_ROOT = REPO_ROOT / "tools" / "lychee_rsv_dsv"
TOKEN2WAV_ROOT = REPO_ROOT / "third_party" / "Step-Audio2"
VLLM_ROOT = REPO_ROOT / "third_party" / "vllm"
for path in (REPO_ROOT, RSV_ROOT, TOKEN2WAV_ROOT, VLLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lychee_fd.runtime.apr.contracts import AcousticTokenBatch
from lychee_fd.runtime.apr.profiling import StageProfiler
from tools.apr.real_acoustic_lanes import (
    APRLocalAcousticLane,
    FixedAffinityAcousticLane,
)
from tools.apr.real_e2e_config import RealE2EConfig
from tools.apr.real_e2e_runner import RealModelAcousticHandoff
from rsv_runtime import BatchPolicy, LogicalRequestState, RequestStateStore
from workloads.apr_benchmark import (
    BASELINES,
    WorkloadTrace,
    generate_workload,
    progression_rounds,
    summarize_trace,
    trace_hash,
    write_trace_jsonl,
)


PROTOCOL_VERSION = "APR-REAL-E2E-N2-v1"
MEASUREMENT_SCHEMA_VERSION = "apr-real-e2e-v1"
SUPPORTED_SYSTEMS = BASELINES
SUPPORTED_CONCURRENCIES = (2, 4, 8, 16)
COLLECTOR_FREQUENCY_HZ = 5.0
GPU_TELEMETRY_QUERY_VARIANT = "external_5hz_nvidia_smi"
COLLECTOR_SCRIPT = REPO_ROOT / "tools" / "final_measurement" / "collect_gpu_telemetry.py"


class PerformanceMeasurementError(ValueError):
    """Raised when a performance attempt lacks complete measurement evidence."""


def build_gpu_collector_command(
    *, out_csv: Path, run_id: str, system: str
) -> list[str]:
    """Build the frozen external collector command for one attempt."""
    return [
        sys.executable,
        str(COLLECTOR_SCRIPT),
        "--out",
        str(out_csv),
        "--run-id",
        str(run_id),
        "--system",
        str(system),
        "--runtime-variant",
        GPU_TELEMETRY_QUERY_VARIANT,
        "--duration",
        "3600.0",
        "--frequency-hz",
        str(COLLECTOR_FREQUENCY_HZ),
    ]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise PerformanceMeasurementError("GPU telemetry has no samples")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_gpu_telemetry(csv_path: Path) -> dict[str, float]:
    """Read and summarize the external collector while preserving GPU identity."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise PerformanceMeasurementError(f"missing GPU telemetry: {csv_path}")
    required = {
        "timestamp_monotonic_ns",
        "gpu_id",
        "gpu_utilization",
        "memory_used",
        "memory_total",
    }
    grouped: dict[str, dict[str, list[float]]] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(set(reader.fieldnames or ())):
                raise PerformanceMeasurementError("GPU telemetry schema is incomplete")
            for row in reader:
                gpu_id = str(row.get("gpu_id", "")).strip()
                if gpu_id not in {"0", "1"}:
                    raise PerformanceMeasurementError(
                        f"unexpected GPU id in telemetry: {gpu_id!r}"
                    )
                try:
                    utilization = float(row["gpu_utilization"])
                    memory_used = float(row["memory_used"])
                    memory_total = float(row["memory_total"])
                    timestamp = int(row["timestamp_monotonic_ns"])
                except (TypeError, ValueError) as exc:
                    raise PerformanceMeasurementError(
                        "GPU telemetry contains non-numeric fields"
                    ) from exc
                if timestamp < 0 or not all(
                    math.isfinite(value)
                    for value in (utilization, memory_used, memory_total)
                ):
                    raise PerformanceMeasurementError(
                        "GPU telemetry contains invalid numeric values"
                    )
                values = grouped.setdefault(
                    gpu_id,
                    {"utilization": [], "memory_used": [], "memory_total": []},
                )
                values["utilization"].append(utilization)
                values["memory_used"].append(memory_used)
                values["memory_total"].append(memory_total)
    except OSError as exc:
        raise PerformanceMeasurementError(
            f"cannot read GPU telemetry: {csv_path}"
        ) from exc
    if set(grouped) != {"0", "1"}:
        raise PerformanceMeasurementError(
            f"GPU telemetry must contain GPU0 and GPU1, got {sorted(grouped)}"
        )
    summary: dict[str, float] = {}
    for gpu_id, values in grouped.items():
        prefix = f"gpu{gpu_id}"
        utilization = values["utilization"]
        memory_used = values["memory_used"]
        summary[f"{prefix}_mean_utilization"] = sum(utilization) / len(utilization)
        summary[f"{prefix}_p95_utilization"] = _percentile(utilization, 0.95)
        summary[f"{prefix}_memory_mean"] = sum(memory_used) / len(memory_used)
        summary[f"{prefix}_memory_max"] = max(memory_used)
        summary[f"{prefix}_sample_count"] = float(len(utilization))
    return summary


def validate_gpu_telemetry(
    csv_path: Path, status_path: Path
) -> dict[str, float]:
    """Require collector PASS status and complete GPU0/GPU1 samples."""
    status_path = Path(status_path)
    if not status_path.is_file():
        raise PerformanceMeasurementError(f"missing collector status: {status_path}")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerformanceMeasurementError("invalid GPU collector status") from exc
    if status.get("status") != "PASS":
        raise PerformanceMeasurementError(
            f"GPU collector did not pass: {status.get('status')!r}"
        )
    return summarize_gpu_telemetry(csv_path)


class GPUCollectorSession:
    """Own one external five-Hz collector for a performance attempt."""

    def __init__(
        self,
        attempt_dir: Path,
        *,
        run_id: str,
        system: str,
        process_factory: Any = subprocess.Popen,
    ) -> None:
        self.attempt_dir = Path(attempt_dir)
        self.run_id = str(run_id)
        self.system = str(system)
        self._process_factory = process_factory
        self._process: Any | None = None
        self.csv_path = self.attempt_dir / "gpu_telemetry.csv"
        self.status_path = self.attempt_dir / "gpu_collector_status.json"

    def start(self) -> None:
        if self._process is not None:
            raise PerformanceMeasurementError("GPU collector already started")
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        self._process = self._process_factory(
            build_gpu_collector_command(
                out_csv=self.csv_path,
                run_id=self.run_id,
                system=self.system,
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self) -> dict[str, Any]:
        if self._process is None:
            raise PerformanceMeasurementError("GPU collector was not started")
        process = self._process
        try:
            process.terminate()
            try:
                exit_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=15)
            try:
                gpu_summary = summarize_gpu_telemetry(self.csv_path)
                status = "PASS" if exit_code in (0, -15) else "FAIL"
                error = "" if status == "PASS" else (
                    f"collector exited with code {exit_code}"
                )
            except PerformanceMeasurementError as exc:
                gpu_summary = {}
                status = "FAIL"
                error = str(exc)
        except Exception as exc:
            exit_code = getattr(process, "returncode", None)
            gpu_summary = {}
            status = "FAIL"
            error = f"collector lifecycle failure: {type(exc).__name__}: {exc}"
        payload = {
            "status": status,
            "collector": "external_5hz_nvidia_smi",
            "frequency_hz": COLLECTOR_FREQUENCY_HZ,
            "exit_code": exit_code,
            "error": error,
            "summary": gpu_summary,
        }
        self.status_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload


def _finite_number(value: Any, *, name: str) -> float:
    """Return a finite numeric value or fail closed for metric aggregation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def build_performance_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build primary performance metrics from one validated attempt result.

    This function is intentionally pure: it consumes the already persisted
    attempt result and does not inspect or mutate runtime state.
    """
    elapsed_s = _finite_number(result.get("elapsed_s"), name="elapsed_s")
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive")
    completed_sessions = int(result.get("completed_sessions", 0))
    if completed_sessions < 0:
        raise ValueError("completed_sessions must be non-negative")
    pcm_bytes = int(result.get("pcm_bytes", 0))
    if pcm_bytes < 0:
        raise ValueError("pcm_bytes must be non-negative")
    useful_audio_seconds = pcm_bytes / (24000.0 * 2.0)
    sessions = result.get("sessions", ())
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        raise ValueError("sessions must be a sequence")

    def _sum_session_metric(name: str) -> int:
        total = 0
        for session in sessions:
            if not isinstance(session, Mapping):
                raise ValueError("session metric row must be a mapping")
            value = session.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            total += int(value)
        return total

    first_audio = result.get("time_to_first_audio_s")
    completion_latency = result.get("completion_latency_s")
    if first_audio is not None:
        first_audio = _finite_number(first_audio, name="time_to_first_audio_s")
        if first_audio < 0:
            raise ValueError("time_to_first_audio_s must be non-negative")
    if completion_latency is not None:
        completion_latency = _finite_number(
            completion_latency, name="completion_latency_s"
        )
        if completion_latency < 0:
            raise ValueError("completion_latency_s must be non-negative")
    return {
        "run_id": str(result.get("run_id", "")),
        "system": str(result.get("system", "")),
        "workload": str(result.get("workload", "")),
        "concurrency": int(result.get("concurrency", 0)),
        "repeat": int(result.get("repeat", 0)),
        "elapsed_s": elapsed_s,
        "session_throughput_sps": completed_sessions / elapsed_s,
        "useful_audio_seconds": useful_audio_seconds,
        "useful_audio_throughput_sps": useful_audio_seconds / elapsed_s,
        "time_to_first_audio_s": first_audio,
        "completion_latency_s": completion_latency,
        "completed_sessions": completed_sessions,
        "pcm_chunks": int(result.get("pcm_chunks", 0)),
        "pcm_bytes": pcm_bytes,
        "checkpoint_count": _sum_session_metric("checkpoint_count"),
        "restore_count": _sum_session_metric("restore_count"),
        "worker_switch_count": _sum_session_metric("worker_switch_count"),
        "acoustic_progression_duration_s": result.get(
            "acoustic_progression_duration_s"
        ),
        "migration_time_s": result.get("migration_time_s"),
    }


PRIMARY_METRICS_FIELDS = (
    "run_id",
    "system",
    "workload",
    "concurrency",
    "repeat",
    "elapsed_s",
    "completed_sessions",
    "pcm_chunks",
    "pcm_bytes",
    "session_throughput_sps",
    "useful_audio_seconds",
    "useful_audio_throughput_sps",
    "time_to_first_audio_s",
    "completion_latency_s",
    "checkpoint_count",
    "restore_count",
    "worker_switch_count",
    "acoustic_progression_duration_s",
    "migration_time_s",
    "gpu0_mean_utilization",
    "gpu0_p95_utilization",
    "gpu0_memory_mean",
    "gpu0_memory_max",
    "gpu1_mean_utilization",
    "gpu1_p95_utilization",
    "gpu1_memory_mean",
    "gpu1_memory_max",
)
INVALID_ATTEMPT_FIELDS = (
    "run_id",
    "system",
    "workload",
    "concurrency",
    "repeat",
    "warmup",
    "valid",
    "failure_signature",
)
WARMUP_FIELDS = ("run_id", "system", "workload", "concurrency", "repeat", "warmup")


def _append_csv_row(path: Path, fields: Sequence[str], row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def append_performance_metrics(out_root: Path, result: Mapping[str, Any]) -> bool:
    """Append exactly one valid counted result, or record it as excluded."""
    out_root = Path(out_root)
    if bool(result.get("warmup")):
        _append_csv_row(out_root / "warmup_attempts.csv", WARMUP_FIELDS, result)
        return False
    if not bool(result.get("valid")):
        _append_csv_row(
            out_root / "invalid_attempts.csv",
            INVALID_ATTEMPT_FIELDS,
            result,
        )
        return False
    gpu_metrics = result.get("gpu_metrics")
    if not isinstance(gpu_metrics, Mapping):
        raise PerformanceMeasurementError(
            "valid counted attempt has no GPU telemetry summary"
        )
    metrics = build_performance_metrics(result)
    for name in PRIMARY_METRICS_FIELDS:
        if name in gpu_metrics:
            metrics[name] = _finite_number(gpu_metrics[name], name=name)
    missing_gpu = {
        "gpu0_mean_utilization",
        "gpu0_p95_utilization",
        "gpu0_memory_mean",
        "gpu0_memory_max",
        "gpu1_mean_utilization",
        "gpu1_p95_utilization",
        "gpu1_memory_mean",
        "gpu1_memory_max",
    } - set(metrics)
    if missing_gpu:
        raise PerformanceMeasurementError(
            f"valid counted attempt lacks GPU metrics: {sorted(missing_gpu)}"
        )
    metrics_path = out_root / "formal_primary_metrics.csv"
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            if any(
                row.get("run_id") == metrics.get("run_id")
                for row in csv.DictReader(handle)
            ):
                raise PerformanceMeasurementError(
                    f"duplicate counted run_id: {metrics.get('run_id')}"
                )
    _append_csv_row(metrics_path, PRIMARY_METRICS_FIELDS, metrics)
    return True


def _csv_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_performance_report(out_root: Path) -> Path:
    """Write a concise count-based report from persisted aggregate artifacts."""
    out_root = Path(out_root)
    valid_count = _csv_row_count(out_root / "formal_primary_metrics.csv")
    invalid_count = _csv_row_count(out_root / "invalid_attempts.csv")
    warmup_count = _csv_row_count(out_root / "warmup_attempts.csv")
    report = out_root / "APR_E2E_PERFORMANCE_REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# APR E2E Performance Report",
                "",
                "This report is generated only from persisted real-model/local-acoustic attempt artifacts.",
                "",
                f"- Valid counted runs: {valid_count}",
                f"- Invalid counted attempts: {invalid_count}",
                f"- Warmup attempts: {warmup_count}",
                "- Measurement contract: external 5 Hz nvidia-smi collector; GPU0/GPU1 kept separate",
                "",
                "Formal metrics are fail-closed: warmups and invalid attempts are excluded from `formal_primary_metrics.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def _read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_formal_metrics(
    out_root: Path, source_roots: Sequence[Path]
) -> Path:
    """Merge per-cell formal CSVs without importing warmups or invalid attempts."""
    out_root = Path(out_root)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for source_root in source_roots:
        for row in _read_metric_rows(Path(source_root) / "formal_primary_metrics.csv"):
            run_id = str(row.get("run_id") or "")
            if not run_id:
                raise PerformanceMeasurementError("formal row has no run_id")
            if run_id in seen:
                raise PerformanceMeasurementError(f"duplicate formal run_id: {run_id}")
            seen.add(run_id)
            rows.append(row)
    out_root.mkdir(parents=True, exist_ok=True)
    destination = out_root / "formal_primary_metrics.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PRIMARY_METRICS_FIELDS))
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in PRIMARY_METRICS_FIELDS}
            for row in rows
        )
    return destination


def _summary_stat(rows: Sequence[Mapping[str, str]], field: str) -> tuple[float, float, float]:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    return (
        statistics.mean(values),
        statistics.median(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def write_full_performance_report(
    out_root: Path, source_roots: Sequence[Path]
) -> Path:
    """Write the paper-facing summary for the complete N=2/4/8 pilot."""
    out_root = Path(out_root)
    destination = merge_formal_metrics(out_root, source_roots)
    rows = _read_metric_rows(destination)
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["system"], int(row["concurrency"])), []).append(row)
    invalid_count = sum(
        len(_read_metric_rows(Path(root) / "invalid_attempts.csv"))
        for root in source_roots
    )
    warmup_count = sum(
        len(_read_metric_rows(Path(root) / "warmup_attempts.csv"))
        for root in source_roots
    )
    expected_complete = True
    lines = [
        "# APR E2E Performance Report",
        "",
        "Real-model Lychee/vLLM execution with local Token2Wav; APR and original_affinity share the same model and external 5 Hz GPU collector.",
        "",
        f"- Valid counted runs: {len(rows)}",
        f"- Invalid counted attempts: {invalid_count}",
        f"- Warmup attempts: {warmup_count}",
        "- Measurement contract: GPU0/GPU1 kept separate; only valid counted rows are formal.",
        "- Superseded pre-fix N=4 evidence is retained under `n4_pilot/` and excluded from the formal merge; it exposed the runner's single-plan coverage gap and was corrected with Bmax-preserving multi-plan progression.",
        "",
        "| System | N | Valid / expected | Completion | Session throughput mean/median/std (sps) | Useful audio throughput mean/median/std (sps) | TTFA mean (s) | Completion latency mean (s) | Checkpoints mean | Restores mean | GPU0 util mean | GPU1 util mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, concurrency in sorted(groups):
        group = groups[(system, concurrency)]
        expected = 3 if concurrency == 2 else 5
        expected_complete = expected_complete and len(group) == expected
        throughput = _summary_stat(group, "session_throughput_sps")
        audio = _summary_stat(group, "useful_audio_throughput_sps")
        ttfa = _summary_stat(group, "time_to_first_audio_s")
        completion = _summary_stat(group, "completion_latency_s")
        checkpoints = _summary_stat(group, "checkpoint_count")
        restores = _summary_stat(group, "restore_count")
        gpu0 = _summary_stat(group, "gpu0_mean_utilization")
        gpu1 = _summary_stat(group, "gpu1_mean_utilization")
        completion_rate = len(group) / expected
        lines.append(
            f"| {system} | {concurrency} | {len(group)} / {expected} | {completion_rate:.3f} | "
            f"{throughput[0]:.4f}/{throughput[1]:.4f}/{throughput[2]:.4f} | "
            f"{audio[0]:.4f}/{audio[1]:.4f}/{audio[2]:.4f} | "
            f"{ttfa[0]:.4f} | {completion[0]:.4f} | {checkpoints[0]:.2f} | "
            f"{restores[0]:.2f} | {gpu0[0]:.2f} | {gpu1[0]:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired APR vs original_affinity comparison",
            "",
            "| N | APR/original session-throughput ratio | APR/original useful-audio-throughput ratio | TTFA delta (s) | Completion-latency delta (s) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    paired_ratios: list[float] = []
    for concurrency in sorted({n for _, n in groups}):
        apr_rows = groups.get(("apr", concurrency), [])
        original_rows = groups.get(("original_affinity", concurrency), [])
        if not apr_rows or not original_rows:
            continue
        apr_throughput = _summary_stat(apr_rows, "session_throughput_sps")[0]
        original_throughput = _summary_stat(
            original_rows, "session_throughput_sps"
        )[0]
        apr_audio = _summary_stat(apr_rows, "useful_audio_throughput_sps")[0]
        original_audio = _summary_stat(
            original_rows, "useful_audio_throughput_sps"
        )[0]
        apr_ttfa = _summary_stat(apr_rows, "time_to_first_audio_s")[0]
        original_ttfa = _summary_stat(
            original_rows, "time_to_first_audio_s"
        )[0]
        apr_completion = _summary_stat(apr_rows, "completion_latency_s")[0]
        original_completion = _summary_stat(
            original_rows, "completion_latency_s"
        )[0]
        ratio = apr_throughput / original_throughput
        audio_ratio = apr_audio / original_audio
        paired_ratios.append(audio_ratio)
        lines.append(
            f"| {concurrency} | {ratio:.4f} | {audio_ratio:.4f} | "
            f"{apr_ttfa - original_ttfa:.4f} | "
            f"{apr_completion - original_completion:.4f} |"
        )
    direction_consistent = bool(paired_ratios) and all(
        ratio > 1.0 for ratio in paired_ratios
    )
    lines.extend(
        [
            "",
            f"FORMAL_COUNT_MATRIX_COMPLETE: {'YES' if expected_complete else 'NO'}",
            f"PERFORMANCE_DIRECTION_CONSISTENT: {'YES' if direction_consistent else 'NO'}",
            "PERFORMANCE_ADVANTAGE_STATUS: NOT_ESTABLISHED when direction is not consistent across N cells; this is descriptive and uses no arbitrary gain threshold.",
            "",
            "The report is descriptive. It does not claim a speedup threshold or generalize beyond this real local-backend workload.",
            "",
        ]
    )
    report = out_root / "APR_E2E_PERFORMANCE_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


@contextmanager
def _cuda_device(index: int):
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.cuda.is_available():
        with torch.cuda.device(index):
            yield
    else:
        yield


def _command(*args: str) -> str:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT, timeout=10
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "container": "duplexpilot",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _command("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
        "git_branch": _command("git", "-C", str(REPO_ROOT), "branch", "--show-current"),
        "gpu": _command(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ),
    }
    try:
        import torch

        snapshot["torch_version"] = str(torch.__version__)
        snapshot["cuda_version"] = str(torch.version.cuda)
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["gpu_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        snapshot["torch_version"] = f"UNAVAILABLE:{type(exc).__name__}"
        snapshot["cuda_version"] = "UNAVAILABLE"
        snapshot["cuda_available"] = False
        snapshot["gpu_count"] = 0
    return snapshot


def build_run_manifest(
    config: RealE2EConfig,
    *,
    system: str,
    concurrency: int,
    repeat: int,
    warmup: bool,
    rounds: int,
    performance: bool = False,
    workload_trace: WorkloadTrace | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    if system not in SUPPORTED_SYSTEMS:
        raise ValueError(f"unsupported real E2E system: {system}")
    if concurrency not in SUPPORTED_CONCURRENCIES:
        raise ValueError(
            f"unsupported concurrency {concurrency}; expected {SUPPORTED_CONCURRENCIES}"
        )
    if repeat <= 0 or rounds <= 0:
        raise ValueError("repeat and rounds must be positive")
    env = environment_snapshot()
    workload_name = (
        workload_trace.workload_id
        if workload_trace is not None
        else "W1-NATURAL-FULL-DUPLEX-REAL-BRINGUP-v1"
    )
    return {
        "run_id": run_id or (
            f"apr-real-{system}-n{concurrency}-r{repeat}"
            f"-{'warmup' if warmup else 'counted'}"
        ),
        "baseline": system,
        "system": system,
        "workload": workload_name,
        "concurrency": int(concurrency),
        "repeat": int(repeat),
        "warmup": bool(warmup),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": "FROZEN-BEFORE-N2-CORRECTNESS",
        "model_checkpoint": str(config.model_path),
        "hardware": env["gpu"],
        "gpu_mapping": {
            "model_execution": "GPU0",
            "local_token2wav": "GPU1",
            "visible_devices": "0,1",
        },
        "input_assets": {
            "prompt_wav": str(config.prompt_wav),
            "model_checkpoint": str(config.model_path),
            "token2wav_checkpoint": str(config.token2wav_path),
        },
        "runtime_config": config.manifest(),
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "measurement_mode": "performance" if performance else "correctness",
        "environment": env,
        "rounds": int(rounds),
        "remote_token2wav": False,
        "production_selector_changed": False,
        "workload_trace_hash": trace_hash(workload_trace) if workload_trace else "",
        "workload_summary": summarize_trace(workload_trace) if workload_trace else {},
    }


def _initial_state(request_id: str, index: int) -> LogicalRequestState:
    if index % 2 == 0:
        return LogicalRequestState(
            request_id=request_id,
            phase="speaking",
            text_history=(11,),
            stoken_history=(152418,),
            control_history=(153228,),
            synthesis_cache_key=request_id,
        )
    return LogicalRequestState(
        request_id=request_id,
        phase="listening",
        text_history=(11, 12, 13),
        stoken_history=(152418, 152419),
        control_history=(153228, 153229),
        synthesis_cache_key=request_id,
    )


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def _record_model_event(
    trace: list[dict[str, Any]],
    *,
    plan: Any,
    round_index: int,
    batches: tuple[AcousticTokenBatch, ...],
    model_result: Mapping[str, Any] | None = None,
    output_summary: Sequence[Mapping[str, Any]] = (),
) -> None:
    model_output_summary = []
    for output in (model_result or {}).get("outputs", ()):
        if not isinstance(output, Mapping):
            continue
        values = output.get("stoken_ids")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            token_count = len(values)
        elif "stoken" in output:
            token_count = 1
        else:
            token_count = 0
        model_output_summary.append(
            {"request_id": str(output.get("request_id", "")), "stoken_count": token_count}
        )
    trace.append(
        {
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "event": "MODEL_EXECUTION",
            "round": int(round_index),
            "request_ids": list(plan.row_to_request),
            "physical_batch_size": int(plan.batch_size),
            "policy": str(plan.policy),
            "virtualized": bool(plan.virtualized),
            "exact_compatible": bool(plan.exact_compatible),
            "handoff_batches": [
                {
                    "request_id": batch.request_id,
                    "stream_id": batch.stream_id,
                    "generation_id": batch.generation_id,
                    "sequence_no": batch.sequence_no,
                    "state_version": batch.state_version,
                    "stoken_count": len(batch.stoken_ids),
                    "source_execution_id": batch.source_execution_id,
                }
                for batch in batches
            ],
            "model_output_summary": model_output_summary,
            "normalized_output_summary": [dict(item) for item in output_summary],
        }
    )


def build_virtualized_round_plans(
    model_store: RequestStateStore,
    request_ids: Sequence[str],
) -> tuple[Any, ...]:
    """Cover one logical round with Bmax-bounded physical plans."""
    remaining = tuple(str(request_id) for request_id in request_ids)
    plans: list[Any] = []
    while remaining:
        plan = model_store.plan(remaining, BatchPolicy.VIRTUALIZED)
        selected = tuple(plan.row_to_request)
        if not selected:
            raise RuntimeError("virtualized planner returned an empty physical plan")
        if not set(selected).issubset(set(remaining)):
            raise RuntimeError(
                f"physical plan selected foreign requests: {selected} from {remaining}"
            )
        plans.append(plan)
        remaining = tuple(request_id for request_id in remaining if request_id not in selected)
    return tuple(plans)


def build_workload_rounds(trace: WorkloadTrace) -> tuple[tuple[str, ...], ...]:
    """Validate a workload trace before it can drive real model execution."""
    if not isinstance(trace, WorkloadTrace):
        raise TypeError("trace must be a WorkloadTrace")
    rounds = progression_rounds(trace, quantum_s=0.01)
    if not rounds:
        raise ValueError("workload trace contains no progression opportunities")
    return rounds


def run_real_correctness_attempt(
    *,
    config: RealE2EConfig,
    system: str,
    concurrency: int,
    repeat: int,
    out_root: Path,
    model_runner: Any,
    lane: Any,
    rounds: int = 2,
    warmup: bool = False,
    performance: bool = False,
    handoff: RealModelAcousticHandoff | None = None,
    workload_trace: WorkloadTrace | None = None,
    profiler: StageProfiler | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run one fail-closed N=2 attempt with injected real components."""
    if concurrency not in SUPPORTED_CONCURRENCIES:
        raise ValueError(
            f"unsupported concurrency {concurrency}; expected {SUPPORTED_CONCURRENCIES}"
        )
    out_root = Path(out_root)
    repeat_dir = f"warmup-r{repeat}" if warmup else f"r{repeat}"
    attempt_dir = out_root / "attempts" / system / f"N{concurrency}" / repeat_dir
    attempt_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config,
        system=system,
        concurrency=concurrency,
        repeat=repeat,
        warmup=warmup,
        rounds=rounds,
        performance=performance,
        workload_trace=workload_trace,
        run_id=run_id,
    )
    (attempt_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if workload_trace is not None:
        if workload_trace.concurrency != concurrency:
            raise ValueError(
                "workload trace concurrency does not match attempt concurrency"
            )
        write_trace_jsonl(workload_trace, attempt_dir / "workload_trace.jsonl")

    session_ids = tuple(f"session-{index}" for index in range(concurrency))
    stream_ids = {
        request_id: f"stream-{request_id}" for request_id in session_ids
    }
    generation_ids = {request_id: 0 for request_id in session_ids}
    handoff = handoff or RealModelAcousticHandoff(
        model_runner,
        stream_ids=stream_ids,
        generation_ids=generation_ids,
    )
    model_store = RequestStateStore()
    rows = {
        request_id: {
            "session_id": request_id,
            "accepted": False,
            "started": False,
            "done": False,
            "cleanup": False,
            "pcm_chunks": 0,
            "pcm_bytes": 0,
            "ownership_errors": 0,
            "checkpoint_count": 0,
            "restore_count": 0,
            "worker_switch_count": 0,
            "first_playable_audio_ns": None,
            "completion_ns": None,
            "failure_signature": "",
        }
        for request_id in session_ids
    }
    canonical_events: list[dict[str, Any]] = []
    model_trace: list[dict[str, Any]] = []
    acoustic_trace: list[dict[str, Any]] = []
    run_start_ns = time.monotonic_ns()
    failure_signature = ""
    cleanup_ok = False
    ownership_errors = 0
    collector = (
        GPUCollectorSession(
            attempt_dir,
            run_id=manifest["run_id"],
            system=system,
        )
        if performance
        else None
    )
    collector_started = False
    collector_status: dict[str, Any] = {}
    gpu_metrics: dict[str, float] = {}
    measurement_error = ""

    try:
        if collector is not None:
            collector.start()
            collector_started = True
        for index, request_id in enumerate(session_ids):
            model_store.register(_initial_state(request_id, index))
        admitted_ids: set[str] = set()

        def admit(request_ids: Sequence[str]) -> None:
            pending = tuple(request_id for request_id in request_ids if request_id not in admitted_ids)
            if not pending:
                return
            admitted = tuple(lane.start(pending))
            if set(admitted) != set(pending):
                raise RuntimeError(
                    f"not all pending sessions admitted: expected={pending} got={admitted}"
                )
            for request_id in admitted:
                if request_id in rows:
                    admitted_ids.add(request_id)
                    rows[request_id]["accepted"] = True
                    rows[request_id]["started"] = True
                    canonical_events.append(
                        {
                            "timestamp_monotonic_ns": time.monotonic_ns(),
                            "event_type": "INPUT_ACCEPTED",
                            "session_id": request_id,
                            "workload_timestamp_s": (
                                min(
                                    (
                                        event.timestamp_s
                                        for event in workload_trace.events
                                        if event.session_id == request_id
                                        and event.event_type == "arrival"
                                    ),
                                    default=None,
                                )
                                if workload_trace is not None
                                else None
                            ),
                        }
                    )

        def consume_acoustic_batches(batches: Sequence[AcousticTokenBatch]) -> None:
            nonlocal ownership_errors
            batches = tuple(batches)
            if not batches:
                return
            by_request = {batch.request_id: batch for batch in batches}
            if len(by_request) != len(batches) or not set(by_request).issubset(
                set(rows)
            ):
                raise RuntimeError(
                    f"handoff request set mismatch: {tuple(by_request)}"
                )
            for batch in batches:
                with _cuda_device(1):
                    lane.submit(batch)
                canonical_events.append(
                    {
                        "timestamp_monotonic_ns": time.monotonic_ns(),
                        "event_type": "ACOUSTIC_SUBMIT",
                        "session_id": batch.request_id,
                        "sequence_no": batch.sequence_no,
                        "state_version": batch.state_version,
                    }
                )
            for _ in batches:
                with _cuda_device(1):
                    progress = lane.process_one()
                if progress is None:
                    raise RuntimeError("acoustic lane made no progress")
                batch = by_request[progress.request_id]
                handoff.acknowledge(
                    batch,
                    committed_state_version=progress.state_version,
                )
                row = rows[progress.request_id]
                row["checkpoint_count"] += int(progress.checkpoint_count)
                row["restore_count"] += int(progress.restore_count)
                row["worker_switch_count"] += int(progress.worker_switch)
                if row["first_playable_audio_ns"] is None and progress.pcm_records:
                    row["first_playable_audio_ns"] = time.monotonic_ns() - run_start_ns
                for record in progress.pcm_records:
                    if (
                        record.request_id != progress.request_id
                        or not record.pcm_bytes
                    ):
                        row["ownership_errors"] += 1
                        ownership_errors += 1
                    row["pcm_chunks"] += 1
                    row["pcm_bytes"] += len(record.pcm_bytes)
                    canonical_events.append(
                        {
                            "timestamp_monotonic_ns": time.monotonic_ns(),
                            "event_type": "PCM_EGRESS",
                            "session_id": progress.request_id,
                            "pcm_seq": record.pcm_seq,
                            "pcm_bytes": len(record.pcm_bytes),
                            "worker_id": progress.worker_id,
                        }
                    )
                acoustic_trace.append(
                    {
                        "timestamp_monotonic_ns": time.monotonic_ns(),
                        "event": "ACOUSTIC_PROGRESS",
                        "session_id": progress.request_id,
                        "worker_id": progress.worker_id,
                        "checkpoint_count": progress.checkpoint_count,
                        "restore_count": progress.restore_count,
                        "state_version": progress.state_version,
                    }
                )

        runtime_rounds = (
            build_workload_rounds(workload_trace)
            if workload_trace is not None
            else tuple(session_ids for _ in range(rounds))
        )
        if workload_trace is None:
            admit(session_ids)
        for round_index, requested_ids in enumerate(runtime_rounds):
            admit(requested_ids)
            active_ids = tuple(request_id for request_id in requested_ids if request_id in admitted_ids)
            if not active_ids:
                continue
            plans = build_virtualized_round_plans(model_store, active_ids)
            for plan_index, plan in enumerate(plans):
                with _cuda_device(0):
                    batches = tuple(handoff.run_plan(model_store, plan))
                _record_model_event(
                    model_trace,
                    plan=plan,
                    round_index=(round_index * len(plans)) + plan_index,
                    batches=batches,
                    model_result=handoff.last_model_result,
                    output_summary=handoff.last_output_summary,
                )
                consume_acoustic_batches(batches)
        if admitted_ids != set(session_ids):
            raise RuntimeError(
                f"not all sessions admitted by workload trace: expected={session_ids} got={tuple(sorted(admitted_ids))}"
            )
        consume_acoustic_batches(handoff.flush_pending())
        for request_id in session_ids:
            model_store.transition(request_id, finished=True)
            rows[request_id]["done"] = (
                rows[request_id]["pcm_chunks"] > 0
                and rows[request_id]["ownership_errors"] == 0
            )
            rows[request_id]["completion_ns"] = time.monotonic_ns() - run_start_ns
            canonical_events.append(
                {
                    "timestamp_monotonic_ns": time.monotonic_ns(),
                    "event_type": "DONE",
                    "session_id": request_id,
                }
            )
    except Exception as exc:
        failure_signature = f"{type(exc).__name__}:{exc}"
        for row in rows.values():
            if not row["done"]:
                row["failure_signature"] = failure_signature
    finally:
        try:
            lane.close()
        except Exception as exc:
            cleanup_ok = False
            failure_signature = failure_signature or (
                f"CLEANUP_{type(exc).__name__}:{exc}"
            )
        try:
            for request_id in model_store.request_ids():
                model_store.transition(request_id, finished=True)
        except Exception as exc:
            cleanup_ok = False
            failure_signature = failure_signature or (
                f"MODEL_CLEANUP_{type(exc).__name__}:{exc}"
            )
        cleanup_ok = bool(
            getattr(lane, "cleanup_ok", False)
            and not model_store.request_ids()
        )
        for row in rows.values():
            row["cleanup"] = cleanup_ok and (
                row["done"] or not row["accepted"]
            )
            canonical_events.append(
                {
                    "timestamp_monotonic_ns": time.monotonic_ns(),
                    "event_type": "CLEANUP",
                    "session_id": row["session_id"],
                    "cleanup": row["cleanup"],
                }
            )
        if collector_started and collector is not None:
            collector_status = collector.stop()
            if collector_status.get("status") == "PASS":
                gpu_metrics = dict(collector_status.get("summary") or {})
            else:
                measurement_error = str(
                    collector_status.get("error") or "GPU telemetry failed"
                )
                failure_signature = failure_signature or (
                    f"MEASUREMENT_GPU:{measurement_error}"
                )
    profile_error = ""
    if profiler is not None:
        try:
            profiler.flush_jsonl(attempt_dir / "pipeline_spans.jsonl")
        except Exception as exc:
            profile_error = f"{type(exc).__name__}:{exc}"
            failure_signature = failure_signature or f"MEASUREMENT_PROFILE:{profile_error}"
            measurement_error = measurement_error or profile_error
    completed_sessions = sum(bool(row["done"]) for row in rows.values())
    failed_sessions = [
        request_id for request_id, row in rows.items() if not row["done"]
    ]
    valid = (
        not failure_signature
        and not failed_sessions
        and cleanup_ok
        and ownership_errors == 0
        and not measurement_error
    )
    elapsed_s = (time.monotonic_ns() - run_start_ns) / 1e9
    result = {
        "run_id": manifest["run_id"],
        "system": system,
        "workload": manifest["workload"],
        "workload_trace_hash": manifest.get("workload_trace_hash", ""),
        "workload_summary": manifest.get("workload_summary", {}),
        "concurrency": concurrency,
        "repeat": repeat,
        "warmup": warmup,
        "valid": bool(valid),
        "completed_sessions": completed_sessions,
        "failed_sessions": failed_sessions,
        "failure_signature": failure_signature,
        "ownership_errors": ownership_errors,
        "cleanup_ok": cleanup_ok,
        "elapsed_s": elapsed_s,
        "time_to_first_audio_s": min(
            (
                row["first_playable_audio_ns"]
                for row in rows.values()
                if row["first_playable_audio_ns"] is not None
            ),
            default=None,
        ) / 1e9 if any(
            row["first_playable_audio_ns"] is not None for row in rows.values()
        ) else None,
        "completion_latency_s": (
            sum(row["completion_ns"] for row in rows.values() if row["completion_ns"] is not None)
            / max(1, completed_sessions)
            / 1e9
        ),
        "pcm_chunks": sum(row["pcm_chunks"] for row in rows.values()),
        "pcm_bytes": sum(row["pcm_bytes"] for row in rows.values()),
        "sessions": list(rows.values()),
        "canonical_events": canonical_events,
        "model_trace": model_trace,
        "acoustic_trace": acoustic_trace,
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "measurement_mode": "performance" if performance else "correctness",
        "measurement_error": measurement_error,
        "profile_error": profile_error,
        "gpu_collector_status": collector_status,
        "gpu_metrics": gpu_metrics,
    }
    (attempt_dir / "canonical_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in canonical_events),
        encoding="utf-8",
    )
    _write_jsonl(attempt_dir / "model_trace.jsonl", model_trace)
    _write_jsonl(attempt_dir / "acoustic_trace.jsonl", acoustic_trace)
    if not performance:
        (attempt_dir / "gpu_collector_status.json").write_text(
            json.dumps(
                {
                    "status": "NOT_STARTED_FOR_CORRECTNESS_GATE",
                    "collector": "external_5hz_nvidia_smi",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (attempt_dir / "attempt_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def validate_real_correctness(result: Mapping[str, Any]) -> dict[str, bool]:
    sessions = list(result.get("sessions", ()))
    accepted = bool(sessions) and all(row.get("accepted") for row in sessions)
    completed = bool(sessions) and all(row.get("done") for row in sessions)
    pcm = bool(sessions) and all(
        int(row.get("pcm_chunks", 0)) > 0 and int(row.get("pcm_bytes", 0)) > 0
        for row in sessions
    )
    ownership = int(result.get("ownership_errors", 0)) == 0 and all(
        int(row.get("ownership_errors", 0)) == 0 for row in sessions
    )
    cleanup = bool(result.get("cleanup_ok")) and all(
        bool(row.get("cleanup")) for row in sessions
    )
    events = list(result.get("canonical_events", ()))
    timestamps = [int(event.get("timestamp_monotonic_ns", 0)) for event in events]
    monotonic = all(
        left <= right for left, right in zip(timestamps, timestamps[1:])
    )
    return {
        "all_sessions_accepted": accepted,
        "all_sessions_completed": completed,
        "pcm_nonempty": pcm,
        "ownership_preserved": ownership,
        "cleanup_observed": cleanup,
        "event_timestamps_monotonic": monotonic,
    }


def write_e2e_correctness_report(
    out_root: Path, results: list[dict[str, Any]]
) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Production APR E2E Correctness Report",
        "",
        "This is a correctness gate for real Lychee/vLLM model execution plus the local Token2Wav checkpoint backend. It is not a throughput or latency claim.",
        "",
        "| System | Repeat | Accepted | Completed | PCM | Ownership | Cleanup | Timestamps | Gate |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    gate_pass = bool(results)
    for result in results:
        checks = validate_real_correctness(result)
        row_pass = all(checks.values())
        gate_pass = gate_pass and row_pass
        status = lambda key: "PASS" if checks[key] else "FAIL"
        lines.append(
            f"| {result.get('system')} | {result.get('repeat')} | "
            f"{status('all_sessions_accepted')} | {status('all_sessions_completed')} | "
            f"{status('pcm_nonempty')} | {status('ownership_preserved')} | "
            f"{status('cleanup_observed')} | {status('event_timestamps_monotonic')} | "
            f"{'PASS' if row_pass else 'FAIL'} |"
        )
    if not results:
        gate_pass = False
        lines.append("| ? | ? | ? | ? | ? | ? | ? | ? | FAIL (no attempts) |")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"REAL_APR_N2_CORRECTNESS_GATE: {'PASS' if gate_pass else 'FAIL'}",
            "",
            "A PASS is required before the N=2 performance pilot. N=4/N=8 are not admitted by this report.",
        ]
    )
    path = out_root / "e2e_correctness_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_real_components(
    config: RealE2EConfig,
    system: str,
    *,
    profiler: StageProfiler | None = None,
):
    """Load the real model and local Token2Wav only for the CLI bring-up."""
    import torch

    from real_runner import RealLycheeBatchRunner

    torch.cuda.set_device(0)
    model_runner = RealLycheeBatchRunner(
        str(config.model_path),
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
    )
    torch.cuda.set_device(1)
    from token2wav import Token2wav

    acoustic_model = Token2wav(str(config.token2wav_path), float16=False)
    if system == "no_rsv_dsv_apr":
        raise RuntimeError(
            "no_rsv_dsv_apr requires a separate no-RSV/DSV model runner; "
            "it is pre-registered but not silently proxied"
        )
    lane_class = APRLocalAcousticLane if system == "apr" else FixedAffinityAcousticLane
    lane = lane_class(
        worker_count=config.worker_count,
        model=acoustic_model,
        prompt_wav=str(config.prompt_wav),
        profiler=profiler,
    )
    return model_runner, lane


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=SUPPORTED_SYSTEMS, required=True)
    parser.add_argument("--concurrency", choices=SUPPORTED_CONCURRENCIES, type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--workload", choices=("A", "B", "C"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-process", choices=("poisson", "burst"), default="poisson")
    parser.add_argument(
        "--performance",
        action="store_true",
        help="enable counted-run metrics and the external 5 Hz GPU collector",
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "reports" / "apr_e2e_performance",
    )
    args = parser.parse_args(argv)
    config = RealE2EConfig.from_env()
    config.validate_paths()
    if config.max_num_seqs < args.concurrency:
        raise SystemExit(
            f"MAX_NUM_SEQS={config.max_num_seqs} is below requested N={args.concurrency}"
        )
    model_runner, lane = build_real_components(config, args.system)
    session_ids = tuple(f"session-{index}" for index in range(args.concurrency))
    workload_trace = (
        generate_workload(
            args.workload,
            concurrency=args.concurrency,
            seed=args.seed,
            arrival_process=args.arrival_process,
        )
        if args.workload
        else None
    )
    handoff = RealModelAcousticHandoff(
        model_runner,
        stream_ids={request_id: f"stream-{request_id}" for request_id in session_ids},
        generation_ids={request_id: 0 for request_id in session_ids},
        acoustic_chunk_size=config.acoustic_chunk_size,
    )
    result = run_real_correctness_attempt(
        config=config,
        system=args.system,
        concurrency=args.concurrency,
        repeat=args.repeat,
        warmup=args.warmup,
        rounds=args.rounds,
        performance=args.performance,
        out_root=args.out_root,
        model_runner=model_runner,
        lane=lane,
        handoff=handoff,
        workload_trace=workload_trace,
    )
    report = write_e2e_correctness_report(args.out_root, [result])
    metrics_appended = False
    performance_report = None
    if args.performance:
        metrics_appended = append_performance_metrics(args.out_root, result)
        performance_report = str(write_performance_report(args.out_root))
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "report": str(report),
                "metrics_appended": metrics_appended,
                "performance_report": performance_report,
            },
            indent=2,
        )
    )
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
