"""Generate diagnostic-only reports from bounded Flow operator evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from profiling.token2wav_flow_profile.contracts import FlowOperatorRecord
from profiling.token2wav_flow_profile.profiler import aggregate_operator_records


REPORT_NAMES = {
    "operator_profile": "TOKEN2WAV_FLOW_OPERATOR_PROFILE.md",
    "candidates": "FLOW_OPTIMIZATION_CANDIDATE_ANALYSIS.md",
    "batching": "APR_ACOUSTIC_BATCHING_ANALYSIS.md",
    "decision": "ACOUSTIC_OPTIMIZATION_DECISION.md",
}


def _cuda_status(records: tuple[FlowOperatorRecord, ...]) -> str:
    return "available" if records and all(record.cuda_time_us is not None for record in records) else "unavailable"


def _ranked_operators(aggregates: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    def rank_key(item: Mapping[str, Any]) -> tuple[bool, float, float]:
        cuda_time = item.get("total_cuda_time_us")
        return (
            cuda_time is not None,
            float(cuda_time if cuda_time is not None else 0.0),
            float(item.get("total_cpu_time_us", 0.0)),
        )

    return sorted(aggregates.values(), key=rank_key, reverse=True)[:10]


def _operator_report(
    records: tuple[FlowOperatorRecord, ...],
    aggregates: Mapping[str, Mapping[str, Any]],
    invalid_attempts: tuple[Mapping[str, Any], ...],
) -> str:
    lines = [
        "# Token2Wav Flow Operator Profile",
        "",
        "Diagnostic-only evidence; this report does not establish E2E speedup.",
        "",
        f"valid operator records: {len(records)}",
        f"invalid attempts: {len(invalid_attempts)}",
        f"CUDA attribution: {_cuda_status(records)}",
        "",
        "## Top operators",
        "",
    ]
    for index, item in enumerate(_ranked_operators(aggregates), start=1):
        cuda_time = item.get("total_cuda_time_us")
        cuda_text = "unknown" if cuda_time is None else f"{float(cuda_time):.6f}"
        input_shapes = tuple(tuple(shape) for shape in item["input_shapes"])
        output_shapes = tuple(tuple(shape) for shape in item["output_shapes"])
        lines.extend(
            [
                f"{index}. `{item['operator_name']}`",
                f"   - cpu_time_us: {float(item['total_cpu_time_us']):.6f}",
                f"   - cuda_time_us: {cuda_text}",
                f"   - kernel_count: {int(item['total_kernel_count'])}",
                f"   - input_shapes={input_shapes}",
                f"   - output_shapes={output_shapes}",
                "",
            ]
        )
    return "\n".join(lines)


def _candidate_report(cuda_status: str) -> str:
    attribution_note = (
        "CUDA operator attribution is available."
        if cuda_status == "available"
        else "CUDA operator attribution is unavailable; kernel-level selection is deferred."
    )
    return "\n".join(
        [
            "# Flow Optimization Candidate Analysis",
            "",
            "Diagnostic-only; candidates are not production changes.",
            "",
            attribution_note,
            "",
            "| Candidate | Decision | Reason |",
            "| --- | --- | --- |",
            "| Kernel fusion | DEFERRED | Requires operator attribution and semantic equivalence. |",
            "| CUDA Graph | DEFERRED | Requires stable shapes/control flow. |",
            "| Mixed precision | DEFERRED | Requires numerical contract validation. |",
            "| TensorRT | DEFERRED | Requires operator coverage and state compatibility. |",
            "| Cache/reuse | DIAGNOSTIC CANDIDATE | Requires measured reuse opportunity. |",
        ]
    )


def _batching_report() -> str:
    return "\n".join(
        [
            "# APR Acoustic Batching Analysis",
            "",
            "The isolated adapter groups only equal-shape, equal-device, equal-dtype Flow states.",
            "",
            "APR can expose compatible logical states because state ownership is explicit, but this artifact only validates the Flow stage.",
            "HiFT, vocoder, PCM, and end-to-end serving remain outside this diagnostic gate.",
            "",
            "No APR selector or production serving path was enabled.",
        ]
    )


def _decision_report(cuda_status: str, record_count: int) -> str:
    return "\n".join(
        [
            "# Acoustic Optimization Decision",
            "",
            "DECISION: DIAGNOSTIC_ONLY",
            "E2E_GAIN_ESTABLISHED: NO",
            "",
            f"Observed valid operator records: {record_count}",
            f"CUDA attribution: {cuda_status}",
            "",
            "A Flow microbenchmark or larger physical batch is not an end-to-end throughput claim.",
            "No production optimization is selected until operator attribution and semantic equivalence gates are complete.",
        ]
    )


def generate_flow_reports(
    records: Iterable[FlowOperatorRecord],
    output_dir: str | Path,
    *,
    invalid_attempts: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Path]:
    """Write the four required diagnostic reports from valid records only."""

    normalized_records = tuple(records)
    if any(not isinstance(record, FlowOperatorRecord) for record in normalized_records):
        raise ValueError("records must contain FlowOperatorRecord values")
    normalized_invalid = tuple(invalid_attempts)
    if any(not isinstance(item, Mapping) for item in normalized_invalid):
        raise ValueError("invalid_attempts must contain mappings")
    aggregates = aggregate_operator_records(normalized_records)
    cuda_status = _cuda_status(normalized_records)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    contents = {
        "operator_profile": _operator_report(
            normalized_records, aggregates, normalized_invalid
        ),
        "candidates": _candidate_report(cuda_status),
        "batching": _batching_report(),
        "decision": _decision_report(cuda_status, len(normalized_records)),
    }
    paths: dict[str, Path] = {}
    for key, filename in REPORT_NAMES.items():
        path = target / filename
        path.write_text(contents[key] + "\n", encoding="utf-8")
        paths[key] = path
    return paths


__all__ = ["REPORT_NAMES", "generate_flow_reports"]
