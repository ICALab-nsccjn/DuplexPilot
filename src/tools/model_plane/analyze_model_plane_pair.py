#!/usr/bin/env python3
"""Compact, auditable analysis for a legacy/row-aware model-plane pair.

This is deliberately an analysis tool, not a serving component.  It treats
the public online pair as descriptive unless the per-request work fingerprints
match exactly.  Request ids are generated afresh by each server run, so
requests are aligned by registration order and round order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(value)
    return rows


def event_name(row: dict[str, Any]) -> str:
    return str(row.get("event_type", row.get("event", "")))


def timestamp(row: dict[str, Any]) -> int:
    return int(row.get("timestamp_monotonic_ns", row.get("timestamp_ns", 0)) or 0)


def _median_probe(path: Path, batch_size: int) -> float | None:
    if not path or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(step.get("elapsed_ns", 0))
        for step in payload.get("steps", [])
        if int(step.get("batch_size", batch_size)) == batch_size
        and float(step.get("elapsed_ns", 0)) > 0
    ]
    return median(values) if values else None


def _request_order(records: Iterable[dict[str, Any]]) -> list[str]:
    first: dict[str, int] = {}
    for row in records:
        if event_name(row) != "MODEL_REQUEST_REGISTER":
            continue
        request_id = str(row.get("request_id", ""))
        if request_id and request_id not in first:
            first[request_id] = timestamp(row)
    return [key for key, _ in sorted(first.items(), key=lambda item: item[1])]


def _fingerprints(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in records if event_name(row) == "MODEL_WORK_FINGERPRINT"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("request_id", ""))].append(row)
    ordered: list[dict[str, Any]] = []
    for request_id, values in sorted(
        groups.items(), key=lambda item: min(timestamp(row) for row in item[1])
    ):
        values.sort(key=timestamp)
        ordered.append(
            {
                "request_id": request_id,
                "rounds": len(values),
                "token_count": sum(int(row.get("generated_token_count", 0) or 0) for row in values),
                "round_token_counts": [int(row.get("generated_token_count", 0) or 0) for row in values],
                "round_hashes": [str(row.get("token_sequence_sha256", "")) for row in values],
                "termination": [str(row.get("termination_reason")) for row in values],
                "finish_reason": [str(row.get("request_finished_reason")) for row in values],
            }
        )
    return ordered


def _step_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    ends = [row for row in records if event_name(row) == "MODEL_ENGINE_STEP_END"]
    # The initial physical prefill has no requested row ids; decode steps have
    # output ids.  Use output ids to exclude the warmup/prefill record.
    decode = [row for row in ends if row.get("output_request_ids")]
    batch_sizes = [int(row.get("model_batch_size", 0) or 0) for row in decode]
    durations = [int(row.get("duration_ns", 0) or 0) for row in decode if int(row.get("duration_ns", 0) or 0) > 0]
    all_times = [timestamp(row) for row in decode]
    start_times = [max(0, timestamp(row) - int(row.get("duration_ns", 0) or 0)) for row in decode]
    locks = [row for row in records if event_name(row) == "MODEL_LOCK"]
    lock_wait = sum(int(row.get("wait_ns", 0) or 0) for row in locks)
    lock_hold = sum(int(row.get("hold_ns", 0) or 0) for row in locks)
    model_span = max(all_times) - min(start_times) if all_times else 0
    return {
        "decode_steps": len(decode),
        "batch_distribution": dict(sorted(Counter(batch_sizes).items())),
        "logical_rows": sum(batch_sizes),
        "b2_steps": sum(size >= 2 for size in batch_sizes),
        "b2_row_fraction": (sum(size for size in batch_sizes if size >= 2) / sum(batch_sizes)) if batch_sizes and sum(batch_sizes) else 0.0,
        "duration_median_ns": median(durations) if durations else None,
        "duration_p95_ns": sorted(durations)[max(0, int(len(durations) * .95) - 1)] if durations else None,
        "busy_ns": sum(durations),
        "model_span_ns": model_span,
        "lock_count": len(locks),
        "lock_wait_ns": lock_wait,
        "lock_hold_ns": lock_hold,
        "driver_dispatches": sum(event_name(row) == "MODEL_ENGINE_BATCH_DISPATCH" for row in records),
        "driver_waits": sum(event_name(row) == "MODEL_DRIVER_DEMAND_WAIT" for row in records),
    }


def _work_comparison(base: list[dict[str, Any]], cand: list[dict[str, Any]]) -> dict[str, Any]:
    same_request_count = len(base) == len(cand)
    pairs: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(base, cand)):
        exact = (
            left["rounds"] == right["rounds"]
            and left["round_token_counts"] == right["round_token_counts"]
            and left["round_hashes"] == right["round_hashes"]
            and left["termination"] == right["termination"]
            and left["finish_reason"] == right["finish_reason"]
        )
        pairs.append(
            {
                "ordinal": index,
                "baseline_rounds": left["rounds"],
                "candidate_rounds": right["rounds"],
                "baseline_tokens": left["token_count"],
                "candidate_tokens": right["token_count"],
                "token_delta": right["token_count"] - left["token_count"],
                "round_counts_equal": left["round_token_counts"] == right["round_token_counts"],
                "hashes_equal": left["round_hashes"] == right["round_hashes"],
                "termination_equal": left["termination"] == right["termination"],
                "exact": exact,
            }
        )
    base_total = sum(item["token_count"] for item in base)
    cand_total = sum(item["token_count"] for item in cand)
    return {
        "request_count_equal": same_request_count,
        "paired_requests": pairs,
        "exact": same_request_count and bool(pairs) and all(item["exact"] for item in pairs),
        "baseline_total_tokens": base_total,
        "candidate_total_tokens": cand_total,
        "total_token_delta": cand_total - base_total,
        "relative_token_delta": ((cand_total - base_total) / base_total) if base_total else None,
        "round_count_equal": same_request_count and all(item["baseline_rounds"] == item["candidate_rounds"] for item in pairs),
    }


def _client_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    sessions = payload.get("sessions", [])
    elapsed = [float(row["elapsed_s"]) for row in sessions if row.get("elapsed_s") is not None]
    return {
        "exists": True,
        "valid": bool(payload.get("valid", False)),
        "session_count": int(payload.get("session_count", len(sessions))),
        "completed_sessions": int(payload.get("completed_sessions", 0)),
        "pcm_chunks": int(payload.get("pcm_chunks", 0)),
        "ownership_errors": int(payload.get("ownership_errors", 0)),
        "runtime_errors": int(payload.get("runtime_errors", 0)),
        "elapsed_median_s": median(elapsed) if elapsed else None,
        "elapsed_mean_s": mean(elapsed) if elapsed else None,
        "elapsed_max_s": max(elapsed) if elapsed else None,
        "session_elapsed_s": elapsed,
    }


def analyze(label: str, trace_path: Path, client_path: Path) -> dict[str, Any]:
    records = read_jsonl(trace_path)
    events = Counter(event_name(row) for row in records)
    stats = _step_stats(records)
    fps = _fingerprints(records)
    client = _client_summary(client_path)
    return {
        "label": label,
        "trace": str(trace_path),
        "event_count": len(records),
        "event_counts": dict(events),
        "request_order": _request_order(records),
        "fingerprints": fps,
        "steps": stats,
        "client": client,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--baseline-client", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--candidate-client", type=Path, required=True)
    parser.add_argument("--probe-b1", type=Path)
    parser.add_argument("--probe-b2", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workload", default="HD-Balanced")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    base = analyze("legacy_serialized", args.baseline_trace, args.baseline_client)
    cand = analyze("row_aware_driver", args.candidate_trace, args.candidate_client)
    comparison = _work_comparison(base["fingerprints"], cand["fingerprints"])
    b1 = _median_probe(args.probe_b1, 1) if args.probe_b1 else None
    b2 = _median_probe(args.probe_b2, 2) if args.probe_b2 else None
    base_span = base["steps"]["model_span_ns"]
    cand_span = cand["steps"]["model_span_ns"]
    base_e2e = base["client"].get("elapsed_median_s")
    cand_e2e = cand["client"].get("elapsed_median_s")
    rows = []
    for item in (base, cand):
        s = item["steps"]
        c = item["client"]
        rows.append({
            "workload": args.workload,
            "concurrency": args.concurrency,
            "system": item["label"],
            "valid": c.get("valid"),
            "decode_steps": s["decode_steps"],
            "logical_rows": s["logical_rows"],
            "b2_steps": s["b2_steps"],
            "b2_row_fraction": s["b2_row_fraction"],
            "step_median_ms": (s["duration_median_ns"] / 1e6) if s["duration_median_ns"] else None,
            "step_p95_ms": (s["duration_p95_ns"] / 1e6) if s["duration_p95_ns"] else None,
            "model_span_s": s["model_span_ns"] / 1e9,
            "lock_wait_s": s["lock_wait_ns"] / 1e9,
            "lock_hold_s": s["lock_hold_ns"] / 1e9,
            "session_elapsed_median_s": c.get("elapsed_median_s"),
            "session_elapsed_mean_s": c.get("elapsed_mean_s"),
            "session_elapsed_max_s": c.get("elapsed_max_s"),
            "completed_sessions": c.get("completed_sessions"),
            "pcm_chunks": c.get("pcm_chunks"),
            "runtime_errors": c.get("runtime_errors"),
        })
    if base_span and cand_span:
        rows.append({
            "workload": args.workload,
            "concurrency": args.concurrency,
            "system": "descriptive_speedup",
            "model_span_speedup": base_span / cand_span,
            "e2e_median_speedup": (base_e2e / cand_e2e) if base_e2e and cand_e2e else None,
            "work_fingerprint_exact": comparison["exact"],
            "round_count_equal": comparison["round_count_equal"],
            "relative_token_delta": comparison["relative_token_delta"],
        })
    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "APR_MODEL_BATCH_ORACLE.csv", rows)
    (out / "APR_MODEL_BATCH_ORACLE.json").write_text(
        json.dumps({
            "schema": "apr-model-plane-pair-analysis-v1",
            "workload": args.workload,
            "concurrency": args.concurrency,
            "baseline": base,
            "candidate": cand,
            "work_comparison": comparison,
            "probe_median_b1_ns": b1,
            "probe_median_b2_ns": b2,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    profile = [
        "# APR Model Execution Plane Critical-Path Profile",
        "",
        f"Workload: `{args.workload}`, N={args.concurrency}. This is a diagnostic pair; no artificial barrier or arrival rewrite was used.",
        "",
        "## Observed execution",
        "",
        f"- Legacy decode steps: **{base['steps']['decode_steps']}**, all physical batch distribution `{base['steps']['batch_distribution']}`.",
        f"- Row-aware decode steps: **{cand['steps']['decode_steps']}**, physical batch distribution `{cand['steps']['batch_distribution']}`; B=2 rows account for {cand['steps']['b2_row_fraction']:.3%} of logical rows.",
        f"- Model span: {base_span/1e9:.3f}s → {cand_span/1e9:.3f}s (descriptive ratio {(base_span/cand_span if cand_span else 0):.4f}×).",
        f"- Public client median session elapsed: {base_e2e!s}s → {cand_e2e!s}s (descriptive ratio {(base_e2e/cand_e2e if base_e2e and cand_e2e else 0):.4f}×).",
        f"- Legacy lock wait: {base['steps']['lock_wait_ns']/1e9:.6f}s; row-aware driver demand waits: {cand['steps']['driver_waits']}.",
        "",
        "## Work comparability",
        "",
        f"- Exact per-request fingerprint match: **{comparison['exact']}**.",
        f"- Round counts equal: **{comparison['round_count_equal']}**; total generated token delta: `{comparison['total_token_delta']}` ({comparison['relative_token_delta']!s}).",
        "- Because sampling order changes when rows are scheduled together, differing token hashes/termination metadata make the online wall-time ratio descriptive rather than a strict causal speedup.",
        "- The pair nevertheless has the same request count and (when the round counts match) nearly the same logical decode workload; a deterministic fixed-work probe is required for causal attribution.",
        "",
        "## Boundary",
        "",
        "This artifact separates model execution time from Flow/PCM time. It does not infer an E2E critical-path fraction without a clock-aligned acoustic timeline.",
    ]
    (out / "APR_MODEL_CRITICAL_PATH_PROFILE.md").write_text("\n".join(profile) + "\n", encoding="utf-8")
    oracle = [
        "# APR Model Batch Oracle Report",
        "",
        "The Oracle is a scheduling/measurement aid, not a paper benchmark. It uses the observed model traces and isolated B=1/B=2 service probes.",
        "",
        "## Results",
        "",
        f"- Isolated probe medians: B=1 `{b1/1e6 if b1 else None}` ms; B=2 `{b2/1e6 if b2 else None}` ms.",
        f"- Legacy observed model span: `{base_span/1e9:.3f}` s; row-aware observed span: `{cand_span/1e9:.3f}` s.",
        f"- Row-aware physical B=2 steps: `{cand['steps']['b2_steps']}/{cand['steps']['decode_steps']}`; logical-row fraction `{cand['steps']['b2_row_fraction']:.3%}`.",
        f"- Online pair exact-work gate: **{'PASS' if comparison['exact'] else 'BLOCKED'}**.",
        "",
        "## Interpretation",
        "",
        "The row-aware driver clearly changes the execution schedule and forms natural model B=2 batches. The current public pair cannot be promoted to a strict causal speedup because stochastic per-row token fingerprints differ, even though round counts and total token counts are close. The correct next evidence is a fixed-work deterministic/recorded-token model execution experiment, followed by a real-arrival online pilot reported descriptively unless the work contract matches.",
    ]
    (out / "APR_MODEL_BATCH_ORACLE_REPORT.md").write_text("\n".join(oracle) + "\n", encoding="utf-8")
    interaction = [
        "# APR Model Execution Plane Interaction Report",
        "",
        "The online pair is not a factorial causal result because its stochastic model work fingerprints do not match exactly.",
        "",
        f"- B=1 model plane comparison (legacy → row-aware): descriptive model-span ratio `{base_span/cand_span if cand_span else None:.4f}×`.",
        f"- Frozen B>1 acoustic path was not changed in this pair; the observed row-aware batch distribution is `{cand['steps']['batch_distribution']}`.",
        f"- Exact work fingerprint: `{comparison['exact']}`; round-count equality: `{comparison['round_count_equal']}`.",
        "",
        "No multiplicative combination of model and acoustic speedups is claimed. A positive result requires fixed-work evidence and then a held-out public online confirmation.",
    ]
    (out / "APR_MODEL_EXECUTION_PLANE_INTERACTION_REPORT.md").write_text("\n".join(interaction) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
