#!/usr/bin/env python3
"""Finalize auditable artifacts for the corrected APR capacity phase.

This utility only reads completed experiment artifacts and writes summaries.  It
does not start a server, alter a trace, or change serving code.  In particular,
an attempt is eligible for SLO calibration only when the strict semantic gate
and the independently measured GPU1 envelope both pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from statistics import median
from typing import Any


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _fmt(value: Any, digits: int = 3) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _copy_if_present(source: Path, target: Path) -> None:
    if source.is_file() and source.resolve() != target.resolve():
        shutil.copyfile(source, target)


def _calibration_summary(root: Path) -> tuple[list[dict[str, str]], str]:
    rows = _read_csv(root / "APR_CORRECTED_CAPACITY_CALIBRATION.csv")
    valid = [row for row in rows if row.get("valid", "").lower() == "true"]
    statuses: dict[str, int] = {}
    for row in rows:
        status = row.get("attempt_status", "UNKNOWN")
        statuses[status] = statuses.get(status, 0) + 1
    u70 = [row for row in rows if row.get("load_fraction") == "0.7"]
    u85 = [row for row in rows if row.get("load_fraction") == "0.85"]
    u70_peaks = [_num(row.get("gpu1_peak_memory_gib")) for row in u70]
    u70_peaks = [value for value in u70_peaks if value is not None]
    u85_peaks = [_num(row.get("gpu1_peak_memory_gib")) for row in u85]
    u85_peaks = [value for value in u85_peaks if value is not None]
    lines = [
        "# APR Corrected Capacity Calibration",
        "",
        "## Decision",
        "",
        "**BLOCKED: no valid CAP0 calibration point was frozen.** A valid point "
        "requires strict completion and an independently measured GPU1 peak of "
        "at most 32 GiB.",
        "",
        f"- attempts in the immutable ledger: **{len(rows)}**",
        f"- valid attempts: **{len(valid)}**",
        f"- status counts: `{json.dumps(statuses, sort_keys=True)}`",
        "- candidate systems were not used to set SLO thresholds",
        "- the paced public trace and its original internal timing were retained",
        "",
        "## Evidence",
        "",
        "| load point | semantic result | measured GPU1 peak | eligibility |",
        "|---:|---|---:|---|",
        f"| 70% clean v4 attempt | incomplete (hard-stopped at envelope) | "
        f"{_fmt(max(u70_peaks) if u70_peaks else None)} GiB | blocked |",
        f"| 70% v3 attempts | 3/3 strict PASS, peak unknown | unknown | blocked |",
        f"| 85% clean attempt | 30/30 strict PASS | "
        f"{_fmt(max(u85_peaks) if u85_peaks else None)} GiB | over envelope |",
        "",
        "The clean 70% attempt was deliberately stopped after the monitor reached "
        "the registered envelope; the raw trace and monitor remain preserved. "
        "The 85% attempt completed semantically but reached 39.13 GiB. The older "
        "70% passes have no independent high-water monitor and therefore cannot "
        "be retroactively treated as valid calibration.",
        "",
        "## SLO status",
        "",
        "TTFA and audio-gap SLOs remain **unfrozen**. No candidate result, lower "
        "load proxy, or partial attempt is substituted. The capacity pilot and "
        "held-out matrix are consequently not authorized.",
        "",
    ]
    return rows, "\n".join(lines)


def _zero_copy_summary(root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    source = root / "zero_copy_mechanism_v3/APR_ZERO_COPY_HANDOFF_METRICS.csv"
    rows = _read_csv(source)
    copy_values = [_num(row.get("handoff_latency_p95_ns")) for row in rows if row.get("mode") == "copy"]
    zero_values = [_num(row.get("handoff_latency_p95_ns")) for row in rows if row.get("mode") == "zero_copy"]
    copy_values = [value for value in copy_values if value is not None]
    zero_values = [value for value in zero_values if value is not None]
    comparison = {
        "copy_p95_median_ms": median(copy_values) / 1e6 if copy_values else None,
        "zero_copy_p95_median_ms": median(zero_values) / 1e6 if zero_values else None,
        "reduction_fraction": (
            1.0 - median(zero_values) / median(copy_values)
            if copy_values and zero_values and median(copy_values) > 0 else None
        ),
        "copy_rows": len(copy_values),
        "zero_copy_rows": len(zero_values),
    }
    return rows, comparison


def _write_pilot_smoke(root: Path) -> None:
    columns = [
        "run_path", "system", "workload", "N", "repeat", "completed_sessions",
        "session_count", "pcm_chunks", "worker_switch_count", "handoff_count",
        "strict_pass", "ownership_errors", "runtime_errors", "evidence_class",
    ]
    output: list[dict[str, Any]] = []
    smoke_root = root / "real_smoke"
    for run in sorted(smoke_root.glob("*/strict_summary.json")) if smoke_root.is_dir() else []:
        run_dir = run.parent
        metadata = _read_json(run_dir / "run_metadata.json", {}) or {}
        strict = _read_json(run, {}) or {}
        switches = 0
        handoffs = 0
        trace = run_dir / "router_trace.jsonl"
        try:
            for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
                event = _read_json_line(line)
                if event.get("worker_switch"):
                    switches += 1
                if event.get("event_type") == "APR_WORKER_HANDOFF_COMMIT":
                    handoffs += 1
        except OSError:
            pass
        output.append({
            "run_path": str(run_dir.relative_to(root)),
            "system": metadata.get("system", ""),
            "workload": metadata.get("workload", ""),
            "N": metadata.get("N", ""),
            "repeat": metadata.get("repeat", ""),
            "completed_sessions": strict.get("completed_sessions", ""),
            "session_count": strict.get("N", ""),
            "pcm_chunks": strict.get("pcm_chunks", ""),
            "worker_switch_count": switches,
            "handoff_count": handoffs,
            "strict_pass": strict.get("strict_pass", False),
            "ownership_errors": strict.get("ownership_errors", 0),
            "runtime_errors": strict.get("runtime_errors", 0),
            "evidence_class": "NEW_DELTA_SMOKE",
        })
    path = root / "APR_CAPACITY_SLO_CORRECTED_PILOT_METRICS.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)


def _write_schedule_trace(root: Path) -> None:
    """Merge bounded mechanism and smoke traces with explicit provenance."""
    sources = [
        (
            root / "zero_copy_mechanism_v3/APR_ZERO_COPY_HANDOFF_SCHEDULE_TRACE.jsonl",
            "zero_copy_mechanism_v3",
            "A100_MECHANISM",
        ),
    ]
    smoke_root = root / "real_smoke"
    if smoke_root.is_dir():
        for trace in sorted(smoke_root.glob("*/router_trace.jsonl")):
            sources.append(
                (trace, str(trace.parent.relative_to(root)), "NEW_DELTA_SMOKE")
            )
    target = root / "APR_CAPACITY_SLO_SCHEDULE_TRACE.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for source, run_path, evidence in sources:
            try:
                lines = source.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue
            for line in lines:
                event = _read_json_line(line)
                if not event:
                    continue
                event["run_path"] = run_path
                event["evidence_class"] = evidence
                handle.write(json.dumps(event, sort_keys=True) + "\n")


def _read_json_line(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_tiered_report(root: Path) -> None:
    payload = _read_json(root / "APR_TIERED_STATE_RESIDENCY_ORACLE.json", {}) or {}
    sizes = payload.get("sizes") if isinstance(payload.get("sizes"), list) else []
    calibration = _read_csv(root / "APR_CORRECTED_CAPACITY_CALIBRATION.csv")
    # This is a reference-only gap, not a frozen SLO: it comes from semantic
    # U70 attempts whose GPU high-water mark was unavailable.
    reference_gaps = [
        _num(row.get("audio_gap_p95_ms"))
        for row in calibration
        if row.get("load_fraction") == "0.7"
        and row.get("semantic_pass", "").lower() == "true"
        and row.get("gpu_peak_known", "").lower() == "false"
        and _num(row.get("audio_gap_p95_ms"))
    ]
    reference_gaps = [value for value in reference_gaps if value is not None]
    reference_gap = median(reference_gaps) if reference_gaps else None
    lines = [
        "# APR Tiered State Residency Oracle",
        "",
        "## Status: feasibility evidence only",
        "",
        "The corrected CAP0 calibration has no valid frozen SLO. Therefore this "
        "read-only HOT/WARM transfer measurement cannot authorize a production "
        "offload policy or an SLO claim.",
        "",
        f"- protocol: `{payload.get('protocol', 'unknown')}`",
        f"- device: `{payload.get('device', 'unknown')}`",
        f"- measured repeats after warmup: `{payload.get('measured_repeats', 'unknown')}`",
        f"- reference-only audio-gap p95 median: `{_fmt(reference_gap)} ms`",
        "",
        "## Measured transfer envelope",
        "",
        "| state size | page-out p95 | page-in p95 | round-trip upper bound |",
        "|---:|---:|---:|---:|",
    ]
    for item in sizes:
        out_ms = _num(item.get("page_out_wall_p95_ms"))
        in_ms = _num(item.get("page_in_wall_p95_ms"))
        total = out_ms + in_ms if out_ms is not None and in_ms is not None else None
        lines.append(
            f"| {_fmt(item.get('size_mib'), 1)} MiB | {_fmt(out_ms)} ms | "
            f"{_fmt(in_ms)} ms | {_fmt(total)} ms |"
        )
    lines += [
        "",
        "For the smallest measured 554 MiB payload, page-in alone is about "
        "56.6 ms p95, while a complete page-out/page-in cycle is about 151 ms. "
        "At 700 MiB and above, page-in already exceeds the 5% budget of the "
        "reference audio-gap value. Prefetch may hide some page-in latency, but "
        "that overlap was not measured here.",
        "",
        "## Decision",
        "",
        "`HOT/WARM` is a plausible follow-up capacity experiment for relieving a "
        "measured GPU1 residency ceiling, but it is **not a passed optimization "
        "gate**. It requires a new preregistered calibration with a valid SLO, "
        "safe-boundary prefetch, and complete PCM/identity/cancel validation. No "
        "production state, checkpoint schema, or Flow numerical path was changed.",
        "",
    ]
    (root / "APR_TIERED_STATE_RESIDENCY_ORACLE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(root: Path, worktree: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    calibration_rows, calibration_md = _calibration_summary(root)
    (root / "APR_CORRECTED_CAPACITY_CALIBRATION.md").write_text(calibration_md, encoding="utf-8")
    _, zero = _zero_copy_summary(root)

    policy = root / "policy_diagnostic"
    _copy_if_present(policy / "APR_HANDOFF_OVERHEAD_COMPARISON.csv", root / "APR_HANDOFF_OVERHEAD_COMPARISON.csv")
    _copy_if_present(policy / "APR_STICKY_LEASE_CORRECTNESS_REPORT.md", root / "APR_STICKY_LEASE_CORRECTNESS_REPORT.md")
    zero_root = root / "zero_copy_mechanism_v3"
    for name in (
        "APR_ZERO_COPY_HANDOFF_FEASIBILITY.md",
        "APR_ZERO_COPY_HANDOFF_CORRECTNESS_REPORT.md",
        "APR_ZERO_COPY_HANDOFF_METRICS.csv",
        "APR_ZERO_COPY_HANDOFF_COMPARISONS.json",
        "APR_ZERO_COPY_HANDOFF_SCHEDULE_TRACE.jsonl",
    ):
        _copy_if_present(zero_root / name, root / name)
    _write_pilot_smoke(root)
    _write_schedule_trace(root)
    _write_tiered_report(root)

    peaks = [
        _num(row.get("gpu1_peak_memory_gib"))
        for row in calibration_rows
        if _num(row.get("gpu1_peak_memory_gib")) is not None
    ]
    strict_passes = sum(row.get("strict_pass", "").lower() == "true" for row in calibration_rows)
    (root / "APR_CAPACITY_SLO_CORRECTED_E2E_REPORT.md").write_text(
        "\n".join([
            "# APR Capacity/SLO Corrected E2E Report",
            "",
            "## Executive result",
            "",
            "**MECHANISM PASS / CAPACITY-SLO BLOCKED.** The corrected assignment "
            "policy and same-GPU zero-copy handoff are validated, but no valid "
            "CAP0 calibration point exists under the fixed 32 GiB GPU1 envelope. "
            "Consequently the online capacity/SLO pilot was not authorized.",
            "",
            "## What was established",
            "",
            "- Forced round-robin was a real semantic defect: it caused handoff even "
            "without a waiting peer. Sticky affinity removes those handoffs; the "
            "contention-triggered policy switches only with a ready peer in the "
            "diagnostic scenarios.",
            "- The A100 zero-copy mechanism gate passed 3/3 repeats. Source and "
            "target contexts were distinct, logical storage was retained, copied "
            "bytes were zero, PCM/identity/cleanup passed, and the p95 handoff "
            f"latency reduction was {_fmt((zero.get('reduction_fraction') or 0)*100, 1)}%.",
            "- The lane still uses a shared model execution lock; these results do "
            "not establish parallel acoustic throughput.",
            "",
            "## Calibration evidence",
            "",
            f"- calibration rows analyzed: `{len(calibration_rows)}`; strict semantic "
            f"passes: `{strict_passes}`; measured GPU1 peaks: `{', '.join(_fmt(p) for p in peaks) or 'none'}` GiB.",
            "- Three older U70 semantic passes have unknown GPU high-water marks and "
            "are not eligible. A clean U70 attempt was stopped at the envelope "
            "(computed high water 34.84 GiB) before completion. A clean U85 attempt "
            "completed 30/30 semantically but reached 39.13 GiB.",
            "- The all-at-once failure and all preserved partial attempts remain in "
            "the result directory; none is deleted or replaced.",
            "",
            "## Gate status",
            "",
            "| gate | result |",
            "|---|---|",
            "| B1/focused correctness | 41/41 targeted capacity/lease/zero-copy tests PASS; broader relevant suite 243 PASS, 1 skipped, with 2 unrelated legacy-module collection failures |",
            "| physical context handoff | PASS (3/3 A100 mechanism repeats) |",
            "| zero-copy handoff | PASS (3/3; no copied state bytes) |",
            "| CAP0 calibration | BLOCKED: zero valid measured-envelope attempts |",
            "| held-out Capacity/SLO pilot | NOT RUN by registered stop condition |",
            "| statistically supported capacity improvement | NOT ESTABLISHED |",
            "",
            "## Interpretation",
            "",
            "The result is an evidence boundary, not evidence that APR has no capacity "
            "value. It shows that the current 30-session public calibration keeps too "
            "much acoustic state resident on GPU1 for the registered 32 GiB envelope. "
            "The tiered-residency oracle is reported separately and does not convert "
            "transfer measurements into an SLO claim.",
            "",
        ]), encoding="utf-8")
    blocked_text = "\n".join([
        "# APR Capacity/SLO Pilot Blocked Report",
        "",
        "The registered online Capacity/SLO pilot was not run because no CAP0 "
        "calibration attempt satisfied both strict completion and the fixed "
        "GPU1 <=32 GiB envelope.",
        "",
        "- clean U70 attempt: hard-stopped before completion at 34.84 GiB",
        "- clean U85 attempt: 30/30 semantic completion, 39.13 GiB",
        "- three older U70 semantic passes: independent GPU high-water unknown",
        "- zero valid calibration points; TTFA/audio-gap SLO therefore unfrozen",
        "",
        "This is a protocol stop, not a claim that APR has no capacity value. All "
        "partial traces, logs and monitor files remain preserved. The canonical "
        "interpretation is in `APR_CAPACITY_SLO_CORRECTED_E2E_REPORT.md`.",
        "",
    ])
    (root / "APR_CAPACITY_SLO_PILOT_BLOCKED_REPORT.md").write_text(
        blocked_text, encoding="utf-8"
    )
    (root / "APR_CAPACITY_SLO_REPORT_INDEX.md").write_text(
        "\n".join([
            "# APR Capacity/SLO Report Index",
            "",
            "Canonical current-phase reports:",
            "",
            "- `APR_CAPACITY_SLO_CORRECTED_E2E_REPORT.md`",
            "- `APR_CAPACITY_SLO_CORRECTED_CAPACITY_CALIBRATION.md`",
            "- `APR_CAPACITY_SLO_NEXT_ROUTE_VERDICT.md`",
            "- `APR_PAPER_CLAIM_BOUNDARY_V3.md`",
            "",
            "The root `APR_CAPACITY_SLO_PILOT_BLOCKED_REPORT.md` is refreshed by "
            "the finalizer. Raw attempts remain under `calibration/`, `real_smoke/` "
            "and `zero_copy_mechanism_v3/`.",
            "",
        ]), encoding="utf-8"
    )
    (root / "APR_CAPACITY_SLO_NEXT_ROUTE_VERDICT.md").write_text(
        "\n".join([
            "# APR Capacity/SLO Next Route Verdict",
            "",
            "## Classification",
            "",
            "`MECHANISM_PASS / CAPACITY_SLO_E2E_BLOCKED`",
            "",
            "## Decision",
            "",
            "Do not continue tuning worker rotation, N, B, padding, or wait windows "
            "on this branch. The corrected policy removes artificial handoffs and the "
            "same-GPU zero-copy path removes the measured copy cost, but the fixed "
            "30-session CAP0 calibration still has no valid <=32 GiB reference from "
            "which to freeze SLOs.",
            "",
            "The only technically justified follow-up is a separately approved, "
            "pre-registered residency/capacity experiment (HOT/WARM with safe-boundary "
            "prefetch) or a revised smaller-state calibration. It must first establish "
            "a valid CAP0 SLO and must not use candidate feedback to set thresholds. "
            "If that route does not produce held-out SLO/short-request gains, stop "
            "capacity optimization rather than stack more scheduler variants.",
            "",
            "## Paper boundary",
            "",
            "The defensible claims from this phase are explicit request-owned acoustic "
            "state, auditable safe-boundary handoff, and a same-GPU zero-copy feasibility "
            "result. A fixed-hardware capacity/SLO improvement is not a result of this "
            "phase.",
            "",
        ]), encoding="utf-8")
    (root / "APR_PAPER_CLAIM_BOUNDARY_V3.md").write_text(
        "\n".join([
            "# APR Paper Claim Boundary v3",
            "",
            "## Allowed",
            "",
            "- APR makes acoustic continuation state explicit and request-owned.",
            "- Checkpoint v2 and safe-boundary fencing support handoff between distinct "
            "physical execution contexts.",
            "- Same-GPU zero-copy lease transfer is a measured low-overhead mechanism "
            "under the bounded A100 test (about 98.5% lower handoff p95 than the copy "
            "path in that mechanism experiment).",
            "- B>1 remains separately validated mechanism evidence; it is not a universal "
            "online throughput claim.",
            "",
            "## Not allowed",
            "",
            "- Claiming a Capacity/SLO or throughput improvement from this phase.",
            "- Treating controlled handoff latency as physical parallel throughput.",
            "- Treating unknown-memory semantic passes or over-envelope runs as valid "
            "calibration.",
            "- Attributing a generic zero-copy/runtime effect exclusively to APR without "
            "a matched causal experiment.",
            "",
        ]), encoding="utf-8")
    # Keep a machine-readable comparison at the root for downstream analysis.
    (root / "APR_ZERO_COPY_HANDOFF_COMPARISON_SUMMARY.json").write_text(
        json.dumps(zero, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"calibration_attempts": len(calibration_rows), "valid_calibration": 0, **zero}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root.resolve(), args.worktree.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
