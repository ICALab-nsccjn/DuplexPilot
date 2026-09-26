"""Build the APR-aware Flow operator profile and Amdahl gate artifacts.

This tool consumes preserved profiler/online traces only.  It never invokes a
model, changes scheduling, or mutates a runtime configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from profiling.apr_operator_profile.analysis import (
    CategoryTiming,
    aggregate_profile_categories,
    analyze_categories,
    choose_candidate,
    load_jsonl,
    summarize_critical_path_events,
    summarize_online_events,
)


def _online_case(path: str | Path) -> dict[str, Any]:
    records = load_jsonl(path)
    summary = summarize_online_events(records)
    registers = [
        int(record["timestamp_monotonic_ns"])
        for record in records
        if str(record.get("event") or record.get("event_type")) == "REQUEST_REGISTER"
        and record.get("timestamp_monotonic_ns") is not None
    ]
    finishes = [
        int(record["timestamp_monotonic_ns"])
        for record in records
        if str(record.get("event") or record.get("event_type")) == "REQUEST_FINISH"
        and record.get("timestamp_monotonic_ns") is not None
    ]
    if registers and finishes:
        e2e_window_ms = (max(finishes) - min(registers)) / 1_000_000.0
    else:
        starts = [float(record.get("start_ms", 0.0)) for record in records if record.get("start_ms") is not None]
        ends = [float(record.get("end_ms", 0.0)) for record in records if record.get("end_ms") is not None]
        e2e_window_ms = max(0.0, max(ends) - min(starts)) if starts and ends else 0.0
    flow_wall = float(summary["flow_wall_ms"])
    critical_flow = float(summary["flow_critical_ms"])
    critical_fraction = critical_flow / flow_wall if flow_wall > 0 else 0.0
    return {
        "path": str(path),
        "flow_wall_ms": flow_wall,
        "flow_cuda_ms": float(summary["flow_cuda_ms"]),
        "critical_flow_ms": critical_flow,
        "critical_flow_fraction": critical_fraction,
        "e2e_window_ms": e2e_window_ms,
        "finalize_critical_ms": float(summary["finalize_critical_ms"]),
        "pcm_critical_ms": float(summary["pcm_critical_ms"]),
        "flow_step_count": int(summary["flow_step_count"]),
        "b2_step_count": int(summary["b2_step_count"]),
    }


def _critical_case(path: str | Path) -> dict[str, Any]:
    """Summarize a row-level critical-path trace without double counting B=2."""

    records = load_jsonl(path)
    summary = summarize_critical_path_events(records)
    flow_wall = float(summary["flow_wall_ms"])
    critical_flow = float(summary["critical_flow_wall_ms"])
    return {
        "path": str(path),
        "flow_wall_ms": flow_wall,
        "flow_cuda_ms": float(summary["flow_cuda_ms"]),
        "critical_flow_ms": critical_flow,
        "critical_flow_fraction": critical_flow / flow_wall if flow_wall > 0 else 0.0,
        "b2_flow_wall_ms": float(summary["b2_flow_wall_ms"]),
        "b2_critical_flow_ms": float(summary["b2_critical_flow_wall_ms"]),
        "b2_step_count": int(summary["b2_step_count"]),
        "flow_step_count": int(summary["flow_step_count"]),
        "unique_event_count": int(summary["unique_event_count"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    profile_paths: list[str | Path],
    online_paths: list[str | Path],
    output_dir: str | Path,
    critical_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    categories = aggregate_profile_categories(profile_paths)
    online = [_online_case(path) for path in online_paths]
    critical = [_critical_case(path) for path in (critical_paths or [])]
    if critical:
        median_critical_fraction = statistics.median(
            float(item["critical_flow_fraction"]) for item in critical
        )
        critical_source = "deduplicated critical-path traces"
    elif online:
        median_critical_fraction = statistics.median(
            float(item["critical_flow_fraction"]) for item in online
        )
        critical_source = "online traces (only if critical annotations are present)"
    else:
        median_critical_fraction = 1.0
        critical_source = "optimistic fixed-work fallback"
    median_e2e_window = statistics.median(
        float(item["e2e_window_ms"]) for item in online if float(item["e2e_window_ms"]) > 0
    ) if any(float(item["e2e_window_ms"]) > 0 for item in online) else 0.0
    median_online_flow_ms = statistics.median(
        float(item["flow_wall_ms"]) for item in online if float(item["flow_wall_ms"]) > 0
    ) if any(float(item["flow_wall_ms"]) > 0 for item in online) else 0.0
    flow_e2e_share = (
        statistics.median(
            float(item["flow_wall_ms"]) / float(item["e2e_window_ms"])
            for item in online
            if float(item["flow_wall_ms"]) > 0 and float(item["e2e_window_ms"]) > 0
        )
        if any(
            float(item["flow_wall_ms"]) > 0 and float(item["e2e_window_ms"]) > 0
            for item in online
        )
        else 1.0
    )

    total_profile_cuda = sum(float(item["cuda_ms"]) for item in categories.values())
    # The profile is an upper-bound attribution: a category is assumed to have
    # the same observed critical fraction as Flow as a whole.  This is stated
    # in the report and is deliberately conservative for online admission.
    category_timings: list[CategoryTiming] = []
    for name in ("state_cache_transition", "dispatch_pack_reuse", "single_dit_family", "unattributed"):
        item = categories.get(name, {"cuda_ms": 0.0, "cpu_ms": 0.0, "kernel_count": 0, "record_count": 0, "explicit_boundary_ms": 0.0, "heuristic_ms": 0.0})
        flow_ms = float(item["cuda_ms"])
        flow_share = flow_ms / total_profile_cuda if total_profile_cuda > 0 else 0.0
        # A single-stream profile and a multi-session online trace have
        # different absolute scales.  Project the profile category share onto
        # the observed online Flow timeline before applying the critical-path
        # fraction; never divide raw profile milliseconds by E2E makespan.
        critical_scale_ms = (
            median_online_flow_ms
            if median_online_flow_ms > 0
            else max(total_profile_cuda, 1.0)
        )
        critical_ms = flow_share * critical_scale_ms * median_critical_fraction
        explicit_ms = float(item.get("explicit_boundary_ms", 0.0))
        heuristic_ms = float(item.get("heuristic_ms", 0.0))
        if explicit_ms > 0:
            evidence = "explicit phase/category instrumentation"
        elif heuristic_ms > 0 and name == "single_dit_family":
            evidence = "operator-family mapping; critical attribution remains conservative"
        elif flow_ms > 0:
            evidence = "unattributed high-level profile; no safe boundary attribution"
        else:
            evidence = "not observed in available profile"
        category_timings.append(
            CategoryTiming(
                name=name,
                flow_wall_ms=flow_ms,
                critical_path_ms=critical_ms,
                avoidable_flow_ms=flow_ms,
                flow_cuda_ms=flow_ms,
                critical_cuda_ms=critical_ms,
                evidence=evidence,
            )
        )
    # Use the observed E2E window if available; otherwise the fixed-work
    # profile is treated as an optimistic all-critical upper bound.
    e2e_critical_ms = (
        median_e2e_window
        if median_e2e_window > 0
        else max(median_online_flow_ms, total_profile_cuda, 1.0)
    )
    optimistic_all_flow_critical_share = (
        flow_e2e_share * median_critical_fraction
        if online
        else median_critical_fraction
    )
    optimistic_all_flow_bound = (
        float("inf")
        if optimistic_all_flow_critical_share >= 1.0
        else 1.0 / (1.0 - optimistic_all_flow_critical_share)
    )
    rows = analyze_categories(category_timings, e2e_critical_ms=e2e_critical_ms)
    selected = choose_candidate(rows)

    metric_rows: list[dict[str, Any]] = []
    for row in rows:
        metric_rows.append(
            {
                "category": row.name,
                "flow_wall_ms": f"{row.flow_wall_ms:.6f}",
                "critical_path_ms": f"{row.critical_path_ms:.6f}",
                "flow_wall_share": f"{row.flow_wall_share:.8f}",
                "critical_path_share": f"{row.critical_path_share:.8f}",
                "avoidable_flow_share": f"{row.avoidable_flow_share:.8f}",
                "candidate_score": f"{row.candidate_score:.8f}",
                "ideal_bound": "inf" if row.ideal_bound == float("inf") else f"{row.ideal_bound:.8f}",
                "eligible": row.eligible,
                "decision_reason": row.decision_reason,
                "evidence": row.evidence,
            }
        )
    _write_csv(target / "APR_OPERATOR_CRITICAL_PATH_METRICS.csv", metric_rows)
    _write_csv(target / "APR_OPERATOR_WATERFALL.csv", metric_rows)

    profile_lines = [
        "# APR Operator Critical-Path Profile",
        "",
        "This is a read-only analysis of preserved real Flow and online traces.",
        "No scheduler, Flow numerical path, checkpoint, or PCM behavior was changed.",
        "",
        f"profile files: {len(profile_paths)}",
        f"online trace files: {len(online_paths)}",
        f"critical-path trace files: {len(critical)}",
        f"profile high-level CUDA total (ms): {total_profile_cuda:.3f}",
        f"median observed Flow critical fraction: {median_critical_fraction:.6f}",
        f"critical-fraction source: {critical_source}",
        f"median online Flow wall (ms): {median_online_flow_ms:.3f}",
        f"estimated online Flow/E2E share: {flow_e2e_share:.6f}",
        f"optimistic all-Flow E2E critical share: {optimistic_all_flow_critical_share:.6f}",
        f"optimistic all-Flow ideal bound: {optimistic_all_flow_bound:.6f}",
        f"median observed E2E window (ms): {median_e2e_window:.3f}",
        "",
        "## Category attribution",
        "",
        "| category | Flow ms | critical-path ms | critical share | score | ideal bound | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        bound = "inf" if row.ideal_bound == float("inf") else f"{row.ideal_bound:.3f}x"
        profile_lines.append(
            f"| {row.name} | {row.flow_wall_ms:.3f} | {row.critical_path_ms:.3f} | "
            f"{row.critical_path_share:.4f} | {row.candidate_score:.4f} | {bound} | "
            f"{row.decision_reason} |"
        )
    profile_lines.extend(
        [
            "",
            "The category-to-critical mapping assumes each measured category has the observed whole-Flow critical fraction; it is not a causal claim.",
            "Absolute profile milliseconds are not divided by the multi-session E2E window; category shares are projected onto the observed online Flow/E2E share.",
            "Even the optimistic all-Flow bound assumes the entire observed Flow critical contribution can be eliminated with zero overhead.",
            "Generic `addmm` is intentionally not assigned to a candidate because this profile cannot prove that its calls are non-QKV.",
            "Generic cat/copy/clone records are also not treated as dispatch or state-transition work without explicit boundary instrumentation.",
            "The prior QKV fusion result remains a negative, superseded candidate and is not reimplemented.",
            "",
            "## Preserved online cases",
            "",
            "| trace | Flow wall ms | critical Flow ms | critical fraction | E2E window ms | B2 steps |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in online:
        profile_lines.append(
            f"| {item['path']} | {item['flow_wall_ms']:.3f} | {item['critical_flow_ms']:.3f} | "
            f"{item['critical_flow_fraction']:.4f} | {item['e2e_window_ms']:.3f} | {item['b2_step_count']} |"
        )
    profile_lines.extend(
        [
            "",
            "## Critical-path traces",
            "",
            "| trace | unique Flow events | Flow ms | critical Flow ms | critical fraction | B2 steps | B2 critical ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in critical:
        profile_lines.append(
            f"| {item['path']} | {item['unique_event_count']} | {item['flow_wall_ms']:.3f} | "
            f"{item['critical_flow_ms']:.3f} | {item['critical_flow_fraction']:.4f} | "
            f"{item['b2_step_count']} | {item['b2_critical_flow_ms']:.3f} |"
        )
    (target / "APR_OPERATOR_CRITICAL_PATH_PROFILE.md").write_text(
        "\n".join(profile_lines) + "\n", encoding="utf-8"
    )

    amdahl_lines = [
        "# APR Operator Amdahl Analysis",
        "",
        "## Gate",
        "",
        "The registered candidate must have an ideal critical-path bound of at least 1.15x and satisfy its profile-share thresholds.",
        "The estimate is intentionally optimistic: it assumes the avoidable portion of the selected category can be eliminated without new overhead.",
        "",
        f"e2e critical denominator used (ms): {e2e_critical_ms:.6f}",
        f"selected candidate: {selected.name if selected else 'NONE'}",
        "",
        "| candidate | Flow share | critical share | avoidable share | score | ideal bound | eligible | reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        bound = "inf" if row.ideal_bound == float("inf") else f"{row.ideal_bound:.6f}"
        amdahl_lines.append(
            f"| {row.name} | {row.flow_wall_share:.6f} | {row.critical_path_share:.6f} | "
            f"{row.avoidable_flow_share:.6f} | {row.candidate_score:.6f} | {bound} | "
            f"{row.eligible} | {row.decision_reason} |"
        )
    if selected is None:
        decision = "APR_OPERATOR_AMDAHL_BLOCKED"
        amdahl_lines.extend(
            [
                "",
                "## Decision",
                "",
                "**APR_OPERATOR_AMDAHL_BLOCKED**: no pre-registered candidate has enough measured critical-path headroom.",
                "This stops production operator implementation under the plan. The B>1 mechanism and generic runtime evidence remain frozen evidence.",
            ]
        )
    else:
        decision = "CANDIDATE_ELIGIBLE"
        amdahl_lines.extend(
            [
                "",
                "## Decision",
                "",
                f"**CANDIDATE_ELIGIBLE**: implement only `{selected.name}` after the required TDD gate.",
            ]
        )
    (target / "APR_OPERATOR_AMDAHL_ANALYSIS.md").write_text(
        "\n".join(amdahl_lines) + "\n", encoding="utf-8"
    )
    if selected is None:
        blocked_lines = [
            "# APR Operator Amdahl Blocked Report",
            "",
            "No pre-registered operator candidate passed the critical-path and share gates.",
            "",
            f"critical-fraction source: {critical_source}",
            f"e2e denominator (ms): {e2e_critical_ms:.6f}",
            "",
            "This is a bounded negative decision for the registered operator candidates, not evidence that every possible kernel optimization is impossible.",
            "The plan therefore stops implementation before changing Flow numerical or state semantics.",
            "",
            "Preserved evidence: B>1 correctness/mechanism results, generic CUDA Graph results, Inductor consistency failure, and QKV fusion negative result.",
        ]
        (target / "APR_OPERATOR_AMDAHL_BLOCKED_REPORT.md").write_text(
            "\n".join(blocked_lines) + "\n", encoding="utf-8"
        )
    return {
        "decision": decision,
        "selected_candidate": selected.name if selected else None,
        "profile_categories": categories,
        "online_cases": online,
        "critical_cases": critical,
        "e2e_critical_ms": e2e_critical_ms,
        "optimistic_all_flow_critical_share": optimistic_all_flow_critical_share,
        "optimistic_all_flow_bound": optimistic_all_flow_bound,
        "metrics_path": str(target / "APR_OPERATOR_CRITICAL_PATH_METRICS.csv"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--online-trace", action="append", default=[])
    parser.add_argument("--critical-trace", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_report(
        args.profile,
        args.online_trace,
        args.output_dir,
        critical_paths=args.critical_trace,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
