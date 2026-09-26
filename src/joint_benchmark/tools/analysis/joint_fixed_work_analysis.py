"""Build an auditable fixed-work factorial view of the joint experiment.

The model and acoustic fixed-work measurements were produced by separate,
validated A100 runners.  This module joins their *component* measurements by
envelope so the four cells can be compared without pretending that local
speedups multiply or that an unmeasured end-to-end wall time was observed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


JOINTS = {
    "J11": (1, 1, "serial", "b1"),
    "J21": (2, 1, "row_batch", "b1"),
    "J12": (1, 2, "serial", "b2"),
    "J22": (2, 2, "row_batch", "b2"),
}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values: Iterable[float]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.median(numbers) if numbers else None


def _read_model_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mode = str(row.get("mode") or "")
            shape = str(row.get("shape") or "").lower()
            if mode not in {"serial", "row_batch"} or shape not in {"p10", "p50", "p90"}:
                continue
            grouped.setdefault((shape, mode), []).append(row)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        elapsed = _median(_number(row.get("median_elapsed_ms")) for row in rows)
        reserved = max((_number(row.get("peak_reserved_bytes")) or 0.0 for row in rows), default=0.0)
        explicit = [
            str(row.get("token_digest_equal_serial_row_batch") or "").lower()
            for row in rows
            if str(row.get("token_digest_equal_serial_row_batch") or "").strip()
        ]
        # The per-mode rows do not carry a cross-mode digest field.  Treating
        # an absent field as False made a validated result look like a failed
        # equivalence check.  Preserve UNKNOWN until a paired field is
        # actually available.
        digest_equal: bool | str = (
            all(value == "true" for value in explicit) if explicit else "UNKNOWN"
        )
        result[key] = {
            "median_elapsed_ms": elapsed,
            "peak_reserved_bytes": int(reserved),
            "token_digest_equal": digest_equal,
            "source_rows": len(rows),
        }
    return result


def _read_acoustic(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, Mapping) else None
    records = records if isinstance(records, list) else []
    selected: list[dict[str, Any]] = []
    shape_selected: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        # The current fixed-work runner emits one record per registered
        # P10/P50/P90 envelope.  Keep those measurements keyed by shape so a
        # later factorial report does not reuse one envelope as three.
        if "mixed_chunk_padding_b2_wall_ms" in record and "independent_two_b1_wall_ms" in record:
            if bool(record.get("warmup")):
                continue
            candidate_wall = _number(record.get("mixed_chunk_padding_b2_wall_ms"))
            independent_wall = _number(record.get("independent_two_b1_wall_ms"))
            if candidate_wall is None or independent_wall is None:
                continue
            peak = 0
            memory = record.get("gpu_memory")
            if isinstance(memory, list):
                for item in memory:
                    if isinstance(item, Mapping):
                        peak = max(
                            peak,
                            int(_number(item.get("max_reserved_bytes")) or 0),
                            int(_number(item.get("max_allocated_bytes")) or 0),
                        )
            actual = {
                "b2_wall_ms": candidate_wall,
                "b1_wall_ms": independent_wall,
                "peak_memory_bytes": peak,
                "pcm_contract_pass": bool(record.get("pcm_contract_pass", False)),
                "config": record.get("config") if isinstance(record.get("config"), Mapping) else {},
                "measured_scope": str(record.get("measured_scope") or "legacy_single_transition"),
                "combined_batch_calls": _number(record.get("combined_batch_calls")),
                "combined_singleton_tail_calls": _number(record.get("combined_singleton_tail_calls")),
            }
            selected.append(actual)
            shape = str(record.get("shape") or "").lower()
            if shape in {"p10", "p50", "p90"}:
                shape_selected.setdefault(shape, []).append(actual)
            continue
        scenario = str(record.get("scenario") or "")
        candidate = record.get("candidate")
        independent = record.get("independent")
        if scenario not in {"mixed_chunk_padding", "mixed_chunk_padding_b2"}:
            continue
        if not isinstance(candidate, Mapping) or not isinstance(independent, Mapping):
            continue
        candidate_wall = _number(candidate.get("wall_ms"))
        independent_wall = _number(independent.get("wall_ms"))
        if candidate_wall is None or independent_wall is None:
            continue
        selected.append({
            "b2_wall_ms": candidate_wall,
            "b1_wall_ms": independent_wall,
            "peak_memory_bytes": int(_number(candidate.get("peak_memory_bytes")) or 0),
            "pcm_contract_pass": bool((record.get("comparison") or {}).get("all_pcm_contract_pass", False)),
            "config": record.get("config") if isinstance(record.get("config"), Mapping) else {},
            "measured_scope": "legacy_single_transition",
            "combined_batch_calls": None,
            "combined_singleton_tail_calls": None,
        })
    if not selected:
        raise ValueError(f"no mixed_chunk_padding acoustic records found in {path}")
    def summarize(items: list[dict[str, Any]], source_shape: str) -> dict[str, Any]:
        scopes = {str(item.get("measured_scope") or "") for item in items}
        return {
            "b1_wall_ms": _median(item["b1_wall_ms"] for item in items),
            "b2_wall_ms": _median(item["b2_wall_ms"] for item in items),
            "peak_memory_bytes": max(item["peak_memory_bytes"] for item in items),
            "pcm_contract_pass": all(item["pcm_contract_pass"] for item in items),
            "records": len(items),
            "source_shape": source_shape,
            "measured_scope": next(iter(scopes)) if len(scopes) == 1 else "mixed",
            "combined_batch_calls": _median(item.get("combined_batch_calls") for item in items),
            "combined_singleton_tail_calls": _median(item.get("combined_singleton_tail_calls") for item in items),
        }

    by_shape = {
        shape: summarize(items, "shape-keyed")
        for shape, items in sorted(shape_selected.items())
        if items
    }
    result = {
        "b1_wall_ms": _median(item["b1_wall_ms"] for item in selected),
        "b2_wall_ms": _median(item["b2_wall_ms"] for item in selected),
        "peak_memory_bytes": max(item["peak_memory_bytes"] for item in selected),
        "pcm_contract_pass": all(item["pcm_contract_pass"] for item in selected),
        "records": len(selected),
        "source_shape": "shape-keyed" if by_shape else "mixed-step-current-shape-padded envelope (prior validated run)",
        "measured_scope": next(
            iter({str(item.get("measured_scope") or "") for item in selected}),
            "unknown",
        ) if len({str(item.get("measured_scope") or "") for item in selected}) == 1 else "mixed",
        "by_shape": by_shape,
    }
    return result


def build_joint_fixed_work_rows(model_csv: Path | str, acoustic_json: Path | str) -> list[dict[str, Any]]:
    """Return all 12 registered cells with component provenance.

    The available acoustic JSON is a validated mixed-step/current-shape
    envelope rather than three separately labelled P10/P50/P90 runs.  Its
    measured value is therefore reused as a common acoustic component for the
    three model prompt envelopes and explicitly marked as a proxy.  No joint
    wall time is synthesized.
    """
    model_path = Path(model_csv)
    acoustic_path = Path(acoustic_json)
    models = _read_model_rows(model_path)
    acoustic = _read_acoustic(acoustic_path)
    rows: list[dict[str, Any]] = []
    for shape in ("p10", "p50", "p90"):
        for joint_id, (b_model, b_acoustic, model_mode, acoustic_mode) in JOINTS.items():
            model = models.get((shape, model_mode))
            if model is None:
                raise ValueError(f"missing model measurement for shape={shape} mode={model_mode}")
            shape_acoustic = acoustic.get("by_shape", {}).get(shape, acoustic)
            rows.append({
                "shape": shape,
                "joint_id": joint_id,
                "B_model": b_model,
                "B_acoustic": b_acoustic,
                "model_mode": model_mode,
                "acoustic_mode": "mixed_chunk_padding_b2" if b_acoustic == 2 else "independent_b1",
                "model_component_median_ms": model["median_elapsed_ms"],
                "acoustic_component_median_ms": shape_acoustic["b2_wall_ms"] if b_acoustic == 2 else shape_acoustic["b1_wall_ms"],
                "model_peak_reserved_bytes": model["peak_reserved_bytes"],
                "acoustic_peak_memory_bytes": shape_acoustic["peak_memory_bytes"],
                "model_token_digest_equal": model["token_digest_equal"],
                "acoustic_pcm_contract_pass": shape_acoustic["pcm_contract_pass"],
                "model_source": str(model_path),
                "acoustic_source": str(acoustic_path),
                "acoustic_source_shape": shape_acoustic["source_shape"],
                "acoustic_measured_scope": shape_acoustic.get("measured_scope", "unknown"),
                "acoustic_combined_batch_calls": shape_acoustic.get("combined_batch_calls"),
                "acoustic_combined_singleton_tail_calls": shape_acoustic.get("combined_singleton_tail_calls"),
                "component_measurement_status": "NEW_SHAPE_KEYED_COMPONENT" if shape in acoustic.get("by_shape", {}) else "REUSED_VALIDATED_COMPONENT",
                "causal_scope": "factorial_component_join",
                "joint_wall_time_ms": None,
                "joint_wall_time_claim": "not_measured",
                "interaction_status": "not_estimable_without_joint_wall_time",
            })
    return rows


def summarize_factorial_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize component main effects, never a fabricated joint effect."""
    by_shape: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_shape.setdefault(str(row["shape"]), {})[str(row["joint_id"])] = row
    result: dict[str, dict[str, Any]] = {}
    for shape, cells in sorted(by_shape.items()):
        def value(joint: str, field: str) -> float:
            raw = _number(cells[joint].get(field))
            if raw is None:
                raise ValueError(f"missing {field} for {shape}/{joint}")
            return raw

        result[shape] = {
            "model_effect_b1_ratio": value("J11", "model_component_median_ms") / value("J21", "model_component_median_ms"),
            "model_effect_b2_ratio": value("J12", "model_component_median_ms") / value("J22", "model_component_median_ms"),
            "acoustic_effect_b1_ratio": value("J11", "acoustic_component_median_ms") / value("J12", "acoustic_component_median_ms"),
            "acoustic_effect_b2_ratio": value("J21", "acoustic_component_median_ms") / value("J22", "acoustic_component_median_ms"),
            "joint_wall_time_claim": "not_measured",
            "interaction_status": "not_estimable_without_joint_wall_time",
        }
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_fixed_work_trace(path: Path, acoustic_json: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a bounded metadata trace for the fixed-work component evidence."""
    payload = json.loads(acoustic_json.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, Mapping) else []
    if not isinstance(records, list):
        records = []
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if not isinstance(record, Mapping):
                continue
            # Keep only timing, shape and contract metadata; no tensors or PCM
            # payloads are copied into the trace.
            event = {
                "event_type": "FIXED_WORK_ACOUSTIC_COMPONENT",
                "shape": record.get("shape"),
                "repeat": record.get("repeat"),
                "warmup": bool(record.get("warmup")),
                "independent_two_b1_wall_ms": record.get("independent_two_b1_wall_ms"),
                "mixed_chunk_padding_b2_wall_ms": record.get("mixed_chunk_padding_b2_wall_ms"),
                "speedup_independent_over_combined": record.get("speedup_independent_over_combined"),
                "pcm_contract_pass": record.get("pcm_contract_pass"),
                "start_step_indices": record.get("start_step_indices"),
                "prior_lengths": record.get("prior_lengths"),
                "current_lengths": record.get("current_lengths"),
                "measured_scope": record.get("measured_scope"),
                "combined_batch_calls": record.get("combined_batch_calls"),
                "combined_singleton_tail_calls": record.get("combined_singleton_tail_calls"),
                "gpu_memory": record.get("gpu_memory"),
            }
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def write_report(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "APR_JOINT_B22_FIXED_WORK_METRICS.csv"
    report_path = output_dir / "APR_JOINT_B22_FIXED_WORK_REPORT.md"
    trace_path = output_dir / "APR_JOINT_B22_FIXED_WORK_TRACE.jsonl"
    _write_csv(csv_path, rows)
    acoustic_sources = {str(row.get("acoustic_source") or "") for row in rows}
    for source in acoustic_sources:
        if source:
            source_path = Path(source)
            if source_path.exists():
                _write_fixed_work_trace(trace_path, source_path, rows)
                break
    effects = summarize_factorial_effects(rows)
    lines = [
        "# APR joint fixed-work 2×2 evidence",
        "",
        "This artifact joins previously validated A100 model and acoustic component measurements. It is intentionally not a synthetic end-to-end wall-time measurement: joint wall time and interaction are left unmeasured because the source runners execute the two components separately.",
        "",
        "## Scope",
        "",
        "- Model rows: validated serial (`B_model=1`) and row-aware batch (`B_model=2`) measurements.",
        "- Acoustic rows: validated `mixed_chunk_padding_b2` versus independent B=1 measurements.",
        "- Shape-keyed acoustic records are joined to the corresponding P10/P50/P90 model envelope when available. Older unkeyed records remain explicitly labelled as a proxy.",
        "- Local component ratios are reported for diagnosis. They must not be multiplied and are not an online E2E claim.",
        "",
        "## Component effects",
        "",
        "| envelope | model B1 time ratio J11/J21 | model B2 time ratio J12/J22 | acoustic B1 time ratio J11/J12 | acoustic B2 time ratio J21/J22 | joint interaction |",
        "|---|---:|---:|---:|---:|---|",
    ]
    scopes = {
        str(row.get("acoustic_measured_scope") or "")
        for row in rows
        if row.get("acoustic_measured_scope")
    }
    if scopes == {"all_remaining_steps_with_mixed_prefix_and_singleton_tail"}:
        lines.insert(
            8,
            "- The acoustic timing covers the full remaining trajectory: a mixed B=2 prefix followed by the singleton tail required when the earlier-terminal row finishes first.",
        )
    elif scopes:
        lines.insert(
            8,
            f"- Acoustic timing scope observed: `{', '.join(sorted(scopes))}`; legacy single-transition records, if present, are not silently treated as full-trajectory measurements.",
        )
    for shape, effect in effects.items():
        lines.append(
            f"| {shape} | {effect['model_effect_b1_ratio']:.4f} | {effect['model_effect_b2_ratio']:.4f} | {effect['acoustic_effect_b1_ratio']:.4f} | {effect['acoustic_effect_b2_ratio']:.4f} | not measured |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A ratio above 1 here means the left component's measured wall time divided by the right component's measured wall time. It is not an end-to-end throughput ratio. The live public-trace matrix is the source for E2E observations; work-fingerprint divergence must remain descriptive.",
        "",
        f"Fixed-work metadata trace: `{trace_path}`.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-csv", type=Path, required=True)
    parser.add_argument("--acoustic-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = build_joint_fixed_work_rows(args.model_csv, args.acoustic_json)
    csv_path, report_path = write_report(rows, args.output_dir)
    print(json.dumps({"metrics": str(csv_path), "report": str(report_path), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
