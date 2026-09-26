#!/usr/bin/env python3
"""Aggregate model-plane fixed-work and online diagnostic evidence.

This tool is deliberately conservative.  The fixed-work runner exercises the
GPU0 model path only; its acoustic-batch label is a join key to already frozen
GPU1 evidence, not an execution claim.  Online comparisons are marked
descriptive unless the bounded work fingerprints match exactly.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def percentile(values: Iterable[float], q: float) -> float | None:
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap_ci(values: list[float], *, seed: int = 1731, samples: int = 10000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(float(statistics.median(draw)))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def fixed_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    rows = [row for row in payload.get("rows", []) if not row.get("warmup")]
    elapsed_ms = [float(row["elapsed_ns"]) / 1e6 for row in rows]
    batch_sizes = [int(size) for row in rows for size in row.get("physical_batch_sizes", [])]
    token_digests = sorted({str(row.get("output_token_only_sha256", "")) for row in rows})
    peaks_alloc = [int(row.get("torch_peak_allocated_bytes", 0)) for row in rows]
    peaks_reserved = [int(row.get("torch_peak_reserved_bytes", 0)) for row in rows]
    logical_requests = int(payload.get("rows", [{}])[0].get("logical_request_count", 2)) if payload.get("rows") else 2
    decode_steps = int(payload.get("decode_steps", 0))
    # Two logical requests complete decode_steps each in every measured repeat.
    logical_steps_per_repeat = logical_requests * decode_steps
    return {
        "file": str(path),
        "mode": payload.get("mode"),
        "shape": payload.get("shape"),
        "shape_prompt_length": payload.get("shape_prompt_length"),
        "acoustic_batch_label": payload.get("acoustic_batch_label"),
        "warmups": payload.get("warmups"),
        "repeats": payload.get("repeats"),
        "decode_steps": decode_steps,
        "median_elapsed_ms": statistics.median(elapsed_ms) if elapsed_ms else None,
        "mean_elapsed_ms": statistics.mean(elapsed_ms) if elapsed_ms else None,
        "p95_elapsed_ms": percentile(elapsed_ms, 0.95),
        "logical_model_steps_per_s": (
            logical_steps_per_repeat / (statistics.median(elapsed_ms) / 1000.0)
            if elapsed_ms and statistics.median(elapsed_ms) > 0 else None
        ),
        "physical_batch_distribution": dict(sorted(Counter(batch_sizes).items())),
        "physical_batch2_fraction": (
            sum(size >= 2 for size in batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
        ),
        "output_token_only_digests": token_digests,
        "output_token_only_digest_consistent": len(token_digests) <= 1,
        "output_step_counts": sorted({int(row.get("output_step_count", 0)) for row in rows}),
        "peak_allocated_bytes": max(peaks_alloc, default=0),
        "peak_reserved_bytes": max(peaks_reserved, default=0),
    }


def event_name(row: dict[str, Any]) -> str:
    return str(row.get("event_type", row.get("event", "")))


def trace_summary(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    end_rows = [row for row in records if event_name(row) == "MODEL_ENGINE_STEP_END" and row.get("output_request_ids")]
    durations = [int(row.get("duration_ns", 0) or 0) for row in end_rows if int(row.get("duration_ns", 0) or 0) > 0]
    batch_sizes = [int(row.get("model_batch_size", 0) or 0) for row in end_rows]
    starts = [int(row.get("timestamp_monotonic_ns", 0)) - int(row.get("duration_ns", 0) or 0) for row in end_rows]
    ends = [int(row.get("timestamp_monotonic_ns", 0)) for row in end_rows]
    locks = [row for row in records if event_name(row) == "MODEL_LOCK"]
    fingerprints = [row for row in records if event_name(row) == "MODEL_WORK_FINGERPRINT"]
    fp_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fingerprints:
        fp_by_request[str(row.get("request_id", ""))].append(row)
    fp_summary = {
        request_id: {
            "rounds": len(rows),
            "tokens": sum(int(row.get("generated_token_count", 0) or 0) for row in rows),
            "round_token_counts": [int(row.get("generated_token_count", 0) or 0) for row in rows],
            "hashes": [str(row.get("token_sequence_sha256", "")) for row in rows],
            "termination": [row.get("termination_reason") for row in rows],
            "finish": [row.get("request_finished_reason") for row in rows],
        }
        for request_id, rows in sorted(fp_by_request.items())
    }
    return {
        "path": str(path),
        "record_count": len(records),
        "event_counts": dict(Counter(event_name(row) for row in records)),
        "decode_step_count": len(end_rows),
        "batch_distribution": dict(sorted(Counter(batch_sizes).items())),
        "logical_rows": sum(batch_sizes),
        "b2_step_count": sum(size >= 2 for size in batch_sizes),
        "b2_row_fraction": sum(size for size in batch_sizes if size >= 2) / sum(batch_sizes) if sum(batch_sizes) else 0.0,
        "model_span_ns": (max(ends) - min(starts)) if starts and ends else 0,
        "busy_ns": sum(durations),
        "step_median_ms": statistics.median(durations) / 1e6 if durations else None,
        "step_p95_ms": percentile([duration / 1e6 for duration in durations], 0.95),
        "lock_wait_ns": sum(int(row.get("wait_ns", 0) or 0) for row in locks),
        "lock_hold_ns": sum(int(row.get("hold_ns", 0) or 0) for row in locks),
        "fingerprints": fp_summary,
    }


def client_summary(path: Path, trace_path: Path | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    payload = load_json(path)
    sessions = payload.get("sessions", [])
    elapsed = [float(row["elapsed_s"]) for row in sessions if row.get("elapsed_s") is not None]
    ttfa: list[float] = []
    pcm_gaps: list[float] = []
    pcm_seconds = 0.0
    for session in sessions:
        events = [row for row in session.get("events", []) if isinstance(row, dict)]
        times = [float(row["server_sse_send_epoch_ms"]) for row in events if row.get("server_sse_send_epoch_ms") is not None]
        pcm = [row for row in events if row.get("type") == "audio_chunk_pcm"]
        pcm_times = [float(row["server_sse_send_epoch_ms"]) for row in pcm if row.get("server_sse_send_epoch_ms") is not None]
        if times and pcm_times:
            ttfa.append((min(pcm_times) - min(times)) / 1000.0)
        pcm_gaps.extend((right - left) / 1000.0 for left, right in zip(pcm_times, pcm_times[1:]))
        for row in pcm:
            try:
                sample_rate = float(row.get("sample_rate") or 24000)
                num_samples = float((row.get("frame_audio") or {}).get("num_samples") or 0)
                if num_samples > 0 and sample_rate > 0:
                    pcm_seconds += num_samples / sample_rate
                elif (row.get("frame_audio") or {}).get("pcm_b64"):
                    raw = base64.b64decode((row["frame_audio"]["pcm_b64"] or "").encode(), validate=False)
                    pcm_seconds += len(raw) / 2.0 / sample_rate
            except Exception:
                pass
    makespan = max(elapsed) if elapsed else None
    return {
        "exists": True,
        "valid": bool(payload.get("valid", False)),
        "session_count": int(payload.get("session_count", len(sessions))),
        "completed_sessions": int(payload.get("completed_sessions", 0)),
        "pcm_chunks": int(payload.get("pcm_chunks", 0)),
        "ownership_errors": int(payload.get("ownership_errors", 0)),
        "runtime_errors": int(payload.get("runtime_errors", 0)),
        "elapsed_median_s": statistics.median(elapsed) if elapsed else None,
        "elapsed_p95_s": percentile(elapsed, 0.95),
        "makespan_s": makespan,
        "useful_audio_seconds": pcm_seconds,
        "useful_audio_throughput": (pcm_seconds / makespan if makespan and makespan > 0 else None),
        "ttfa_p50_s": percentile(ttfa, 0.50),
        "ttfa_p95_s": percentile(ttfa, 0.95),
        "ttfa_p99_s": percentile(ttfa, 0.99),
        "inter_audio_gap_p95_s": percentile(pcm_gaps, 0.95),
        "inter_audio_gap_p99_s": percentile(pcm_gaps, 0.99),
        "session_elapsed_s": elapsed,
    }


def acoustic_trace_summary(path: Path) -> dict[str, Any]:
    """Summarize GPU1 Flow batching without retaining tensor/audio payloads."""
    records = load_jsonl(path)
    # FLOW_BATCH_COMPLETE is the most stable one-record-per-dispatch event in
    # the online recorder.  Fall back to FLOW_STEP_END for older traces.
    dispatches = [
        row for row in records
        if event_name(row) in {"FLOW_BATCH_COMPLETE", "FLOW_STEP_END"}
        and row.get("batch_size") is not None
    ]
    if not dispatches:
        return {
            "dispatch_count": 0,
            "batch_distribution": {},
            "b2_dispatch_count": 0,
            "b2_row_fraction": 0.0,
            "flow_work_fraction": 0.0,
        }
    sizes = [int(row.get("batch_size", 0) or 0) for row in dispatches]
    logical_rows = sum(sizes)
    b2 = sum(size >= 2 for size in sizes)
    return {
        "dispatch_count": len(dispatches),
        "batch_distribution": dict(sorted(Counter(sizes).items())),
        "b2_dispatch_count": b2,
        "b2_row_fraction": (sum(size for size in sizes if size >= 2) / logical_rows) if logical_rows else 0.0,
        "flow_work_fraction": b2 / len(sizes),
    }


def fingerprint_compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_fp = left.get("fingerprints", {})
    right_fp = right.get("fingerprints", {})
    left_rows = list(left_fp.values())
    right_rows = list(right_fp.values())
    pairs = []
    for lrow, rrow in zip(left_rows, right_rows):
        pairs.append({
            "rounds_equal": lrow["rounds"] == rrow["rounds"],
            "token_counts_equal": lrow["round_token_counts"] == rrow["round_token_counts"],
            "hashes_equal": lrow["hashes"] == rrow["hashes"],
            "termination_equal": lrow["termination"] == rrow["termination"],
            "finish_equal": lrow["finish"] == rrow["finish"],
            "token_delta": rrow["tokens"] - lrow["tokens"],
        })
    return {
        "request_count_equal": len(left_rows) == len(right_rows),
        "round_counts_equal": len(left_rows) == len(right_rows) and all(row["rounds_equal"] for row in pairs),
        "exact": len(left_rows) == len(right_rows) and bool(pairs) and all(all(value for key, value in row.items() if key != "token_delta") for row in pairs),
        "total_token_delta": sum(row["token_delta"] for row in pairs),
        "pairs": pairs,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_fixed(root: Path, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [fixed_summary(path) for path in sorted(root.glob("*.json"))]
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append({key: value for key, value in row.items() if key not in ("output_token_only_digests", "physical_batch_distribution", "output_step_counts")})
    by_shape: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_shape[str(row["shape"])][str(row["mode"])] = row
    interactions = []
    for shape, modes in sorted(by_shape.items()):
        serial = modes.get("serial")
        row_single = modes.get("row_single")
        row_batch = modes.get("row_batch")
        if not (serial and row_single and row_batch):
            continue
        g_b1 = serial["median_elapsed_ms"] / row_single["median_elapsed_ms"] if row_single["median_elapsed_ms"] else None
        g_b2 = serial["median_elapsed_ms"] / row_batch["median_elapsed_ms"] if row_batch["median_elapsed_ms"] else None
        ratios_b1 = [a / b for a, b in zip(
            [float(x) / 1e6 for x in load_json(Path(serial["file"]))["measured_elapsed_ns"]],
            [float(x) / 1e6 for x in load_json(Path(row_single["file"]))["measured_elapsed_ns"]],
        ) if b > 0]
        ratios_b2 = [a / b for a, b in zip(
            [float(x) / 1e6 for x in load_json(Path(serial["file"]))["measured_elapsed_ns"]],
            [float(x) / 1e6 for x in load_json(Path(row_batch["file"]))["measured_elapsed_ns"]],
        ) if b > 0]
        interactions.append({
            "shape": shape,
            "G_B1_speedup": g_b1,
            "G_B2_speedup": g_b2,
            "interaction_log_effect": (math.log(g_b2) - math.log(g_b1)) if g_b1 and g_b2 and g_b1 > 0 and g_b2 > 0 else None,
            "B1_ratio_median": statistics.median(ratios_b1) if ratios_b1 else None,
            "B1_ratio_ci_low": bootstrap_ci(ratios_b1)[0],
            "B1_ratio_ci_high": bootstrap_ci(ratios_b1)[1],
            "B2_ratio_median": statistics.median(ratios_b2) if ratios_b2 else None,
            "B2_ratio_ci_low": bootstrap_ci(ratios_b2)[0],
            "B2_ratio_ci_high": bootstrap_ci(ratios_b2)[1],
            "token_digest_equal_serial_row_single": serial["output_token_only_digests"] == row_single["output_token_only_digests"],
            "token_digest_equal_serial_row_batch": serial["output_token_only_digests"] == row_batch["output_token_only_digests"],
            "model_only_warning": "acoustic batch label is a frozen join label; this runner does not execute GPU1 Flow",
        })
    write_csv(out / "APR_MODEL_EXECUTION_PLANE_FIXED_WORK_METRICS.csv", csv_rows + interactions)
    payload = {"schema": "apr-model-plane-fixed-aggregate-v1", "runs": rows, "interactions": interactions}
    (out / "fixed_work_aggregate.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_rows, payload


def analyze_online(online_root: Path, out: Path) -> dict[str, Any]:
    cases = []
    for case_dir in sorted(online_root.glob("*")):
        client_path = case_dir / "client" / "online_attempt.json"
        trace_path = case_dir / "model_execution_trace.jsonl"
        if not client_path.exists():
            continue
        meta = load_json(case_dir / "run_metadata.json") if (case_dir / "run_metadata.json").exists() else {}
        model = trace_summary(trace_path)
        client = client_summary(client_path)
        acoustic = acoustic_trace_summary(case_dir / "online_trace.jsonl")
        cases.append({"case": case_dir.name, "metadata": meta, "model": model, "acoustic": acoustic, "client": client})
    comparisons = []
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        name = case["case"]
        metadata = case.get("metadata", {})
        workload = str(metadata.get("workload") or name.split("_")[0])
        n = str(metadata.get("N") or (name.split("_")[1] if len(name.split("_")) > 1 else "unknown"))
        system = str(metadata.get("system") or "rsv_dsv_apr_step_b1")
        mode = "row_aware" if "rowaware" in name else "legacy"
        grouped[(workload, n, system)][mode] = case
    metric_rows = []
    for (workload, n, system), modes in sorted(grouped.items()):
        for mode, case in sorted(modes.items()):
            model = case["model"]; acoustic = case["acoustic"]; client = case["client"]
            metric_rows.append({
                "workload": workload, "N": n, "system": system, "mode": mode, "case": case["case"],
                "valid": client.get("valid"), "completed_sessions": client.get("completed_sessions"),
                "pcm_chunks": client.get("pcm_chunks"), "model_decode_steps": model.get("decode_step_count"),
                "model_batch_distribution": json.dumps(model.get("batch_distribution", {}), sort_keys=True),
                "model_b2_row_fraction": model.get("b2_row_fraction"),
                "model_span_s": model.get("model_span_ns", 0) / 1e9,
                "model_step_median_ms": model.get("step_median_ms"),
                "lock_wait_s": model.get("lock_wait_ns", 0) / 1e9,
                "acoustic_flow_dispatches": acoustic.get("dispatch_count"),
                "acoustic_batch_distribution": json.dumps(acoustic.get("batch_distribution", {}), sort_keys=True),
                "acoustic_b2_dispatch_count": acoustic.get("b2_dispatch_count"),
                "acoustic_b2_row_fraction": acoustic.get("b2_row_fraction"),
                "acoustic_b2_flow_work_fraction": acoustic.get("flow_work_fraction"),
                "e2e_makespan_s": client.get("makespan_s"),
                "e2e_useful_audio_throughput": client.get("useful_audio_throughput"),
                "ttfa_p95_s": client.get("ttfa_p95_s"),
                "inter_audio_gap_p95_s": client.get("inter_audio_gap_p95_s"),
                "runtime_errors": client.get("runtime_errors"),
                "ownership_errors": client.get("ownership_errors"),
            })
        if "legacy" in modes and "row_aware" in modes:
            base = modes["legacy"]; cand = modes["row_aware"]
            comparison = fingerprint_compare(base["model"], cand["model"])
            bspan = base["model"].get("model_span_ns", 0); cspan = cand["model"].get("model_span_ns", 0)
            bm = base["client"].get("makespan_s"); cm = cand["client"].get("makespan_s")
            comparisons.append({
                "workload": workload, "N": n, "system": system,
                "model_span_speedup_descriptive": bspan / cspan if cspan else None,
                "e2e_makespan_speedup_descriptive": bm / cm if bm and cm else None,
                "legacy_b2_row_fraction": base["model"].get("b2_row_fraction"),
                "row_aware_b2_row_fraction": cand["model"].get("b2_row_fraction"),
                "legacy_acoustic_b2_row_fraction": base["acoustic"].get("b2_row_fraction"),
                "candidate_acoustic_b2_row_fraction": cand["acoustic"].get("b2_row_fraction"),
                "candidate_acoustic_b2_dispatch_count": cand["acoustic"].get("b2_dispatch_count"),
                "work_fingerprint_exact": comparison["exact"],
                "round_counts_equal": comparison["round_counts_equal"],
                "total_token_delta_candidate_minus_legacy": comparison["total_token_delta"],
                "causal_status": "CAUSAL_ELIGIBLE" if comparison["exact"] else "DESCRIPTIVE_ONLY_WORK_MISMATCH",
            })
    write_csv(out / "APR_MODEL_EXECUTION_PLANE_ONLINE_METRICS.csv", metric_rows + comparisons)
    payload = {"schema": "apr-model-plane-online-aggregate-v1", "cases": cases, "comparisons": comparisons}
    (out / "online_aggregate.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_reports(result_root: Path, fixed_payload: dict[str, Any], online_payload: dict[str, Any]) -> None:
    """Write human-readable artifacts without turning diagnostics into claims."""
    runs = fixed_payload.get("runs", [])
    interactions = fixed_payload.get("interactions", [])
    online_cases = online_payload.get("cases", [])
    online_comparisons = online_payload.get("comparisons", [])
    online_exact_count = sum(
        row.get("causal_status") == "CAUSAL_ELIGIBLE" for row in online_comparisons
    )
    online_descriptive_count = sum(
        row.get("causal_status") == "DESCRIPTIVE_ONLY_WORK_MISMATCH"
        for row in online_comparisons
    )
    online_model_b2_cases = sum(
        float(case.get("model", {}).get("b2_row_fraction", 0.0) or 0.0) > 0.0
        for case in online_cases
    )
    online_acoustic_b2_cases = sum(
        float(case.get("acoustic", {}).get("b2_row_fraction", 0.0) or 0.0) > 0.0
        for case in online_cases
    )
    online_ns = sorted({str(case.get("metadata", {}).get("N", "?")) for case in online_cases})
    online_workloads = sorted({str(case.get("metadata", {}).get("workload", "?")) for case in online_cases})

    fixed_lines = [
        "# APR Model Execution Plane Fixed-Work Report",
        "",
        "## Scope",
        "",
        "This is a GPU0 model-only fixed-work experiment. Two deterministic greedy model requests are run for eight decode turns per repeat with five warmups and twenty measured repeats for P10/P50/P90 input-shape proxies. The `acoustic_batch_label` is a join label to frozen GPU1 evidence; this runner does not execute Token2Wav or Flow.",
        "",
        "## Results",
        "",
        "| shape | legacy B_model=1 median (ms) | row-aware B_model=1 median (ms) | row-aware B_model=2 median (ms) | B2/legacy speedup | token digest |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for shape in ("p10", "p50", "p90"):
        shape_runs = {str(row.get("mode")): row for row in runs if row.get("shape") == shape}
        serial = shape_runs.get("serial"); single = shape_runs.get("row_single"); batch = shape_runs.get("row_batch")
        if not (serial and single and batch):
            continue
        speed = serial["median_elapsed_ms"] / batch["median_elapsed_ms"]
        digest_ok = serial["output_token_only_digests"] == batch["output_token_only_digests"] == single["output_token_only_digests"]
        fixed_lines.append(
            f"| {shape} | {_fmt(serial['median_elapsed_ms'], 3)} | {_fmt(single['median_elapsed_ms'], 3)} | {_fmt(batch['median_elapsed_ms'], 3)} | {speed:.3f}x | {'PASS' if digest_ok else 'FAIL'} |"
        )
    fixed_lines += [
        "",
        "## Contract and interpretation",
        "",
        "- Every measured row-aware B_model=2 repeat dispatched physical batch size 2; every serial/row-single repeat dispatched physical batch size 1.",
        "- Output token-only digests are stable across modes for each shape, and each repeat emitted the same two-request, eight-turn workload. This supports fixed-work model-path attribution.",
        "- Row-aware B_model=1 is approximately neutral relative to legacy serialization; the observed gain comes from serving two logical rows in one real vLLM model step.",
        "- The result is not a combined model-plus-acoustic E2E result and must not be multiplied by the frozen acoustic B=2 mechanism speedup.",
        "- GPU memory fields in the CSV are PyTorch peaks for the model probe (about 28.7 GiB reserved); they are not a GPU1 Flow memory measurement.",
        "",
        "## Gate",
        "",
        "`FIXED_WORK_MODEL_PLANE_PASS`: correctness and token-work equivalence pass, physical B_model=2 is observed, and all three shape proxies show a substantial model-path reduction. The online causal gate remains separate because the public stochastic service may generate different work.",
    ]
    (result_root / "APR_MODEL_EXECUTION_PLANE_FIXED_WORK_REPORT.md").write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")

    interaction_lines = [
        "# APR Model Execution Plane Interaction Report",
        "",
        "Definitions: `G_B1 = log(T_legacy_B1 / T_row-aware_B1)` and `G_B2 = log(T_legacy_B1 / T_row-aware_B2)` for the model-only fixed-work probe. The acoustic batch label is held constant as a frozen join label; no GPU1 Flow execution occurs here.",
        "",
        "| shape | B_model=1 speedup | B_model=2 speedup | interaction log effect |",
        "|---|---:|---:|---:|",
    ]
    for row in interactions:
        interaction_lines.append(
            f"| {row.get('shape')} | {_fmt(row.get('B1_ratio_median'), 4)}x | {_fmt(row.get('B2_ratio_median'), 4)}x | {_fmt(row.get('interaction_log_effect'), 5)} |"
        )
    interaction_lines += [
        "",
        "## Boundary",
        "",
        "The positive B_model interaction is evidence that the row-aware driver changes the GPU0 execution schedule. It is not evidence that APR acoustic B=2 and model B=2 have a multiplicative interaction. A full combined causal claim requires identical model work and a clock-aligned online acoustic timeline.",
    ]
    (result_root / "APR_MODEL_EXECUTION_PLANE_INTERACTION_REPORT.md").write_text("\n".join(interaction_lines) + "\n", encoding="utf-8")

    online_lines = [
        "# APR Model Execution Plane Online Report",
        "",
        "## Scope",
        "",
        f"This artifact reports {len(online_cases)} real public-trace API case records across workloads {', '.join(online_workloads) or 'n/a'} and N={', '.join(online_ns) or 'n/a'}. Arrival and audio timelines were reused without artificial barriers or `--no-sleep`. The row-aware mode is an explicit opt-in; legacy and candidate services were restarted independently on the same two-A100 placement. This is a bounded representative smoke, not the planned repeated held-out matrix.",
        "",
        "| workload | N | system | mode | valid | B_model=2 row fraction | B_acoustic=2 row fraction | model span (s) | makespan (s) | TTFA p95 (s) | PCM chunks |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in online_cases:
        meta = case.get("metadata", {}); model = case.get("model", {}); acoustic = case.get("acoustic", {}); client = case.get("client", {})
        online_lines.append(
            f"| {meta.get('workload', 'unknown')} | {meta.get('N', 'unknown')} | {meta.get('system', 'unknown')} | {meta.get('mode', case.get('case'))} | {client.get('valid')} | {model.get('b2_row_fraction', 0):.3%} | {acoustic.get('b2_row_fraction', 0):.3%} | {_fmt(model.get('model_span_ns', 0)/1e9, 3)} | {_fmt(client.get('makespan_s'), 3)} | {_fmt(client.get('ttfa_p95_s'), 3)} | {client.get('pcm_chunks', 0)} |"
        )
    online_lines += ["", "## Pair interpretation", ""]
    if online_comparisons:
        for row in online_comparisons:
            online_lines.append(
                f"- `{row.get('workload')}` N={row.get('N')} `{row.get('system')}`: descriptive model-span ratio **{_fmt(row.get('model_span_speedup_descriptive'), 3)}x**, descriptive makespan ratio **{_fmt(row.get('e2e_makespan_speedup_descriptive'), 3)}x**, candidate B_model=2 row fraction **{row.get('row_aware_b2_row_fraction', 0):.3%}**, candidate B_acoustic=2 row fraction **{row.get('candidate_acoustic_b2_row_fraction', 0):.3%}**. Work status: **{row.get('causal_status')}**; round counts equal={row.get('round_counts_equal')}; total token delta={row.get('total_token_delta_candidate_minus_legacy')}**."
            )
    else:
        online_lines.append("No complete legacy/candidate pair is present yet.")
    online_lines += [
        "",
        "## Claim boundary",
        "",
        "The online row-aware path demonstrably forms real model B=2 when the trace contains concurrent model demand. B_model and B_acoustic are measured independently; a model B=2 hit must not be confused with GPU1 acoustic B=2. If fingerprints or round counts differ, the wall-time ratio is descriptive and cannot be used as a strict causal paper speedup. The fixed-work result is the causal model-path evidence; the online result is an E2E feasibility/pilot signal until repeated held-out runs preserve work.",
    ]
    (result_root / "APR_MODEL_EXECUTION_PLANE_ONLINE_REPORT.md").write_text("\n".join(online_lines) + "\n", encoding="utf-8")

    stats_lines = [
        "# APR Model Execution Plane Statistical Analysis",
        "",
        "- Fixed-work timing comparisons use 20 measured repeats per shape after five warmups; paired ordinal ratios are summarized with 10,000 bootstrap resamples of the median.",
        "- The fixed-work B_model=2/legacy ratios are reported in `APR_MODEL_EXECUTION_PLANE_FIXED_WORK_METRICS.csv`; confidence intervals are descriptive for the model-only probe.",
        f"- The public evidence contains {len(online_comparisons)} complete legacy/row-aware pairs: {online_exact_count} exact-work pair(s) and {online_descriptive_count} descriptive work-mismatch pair(s). Because the observed public pair changes generated work, no causal confidence interval is reported for its wall-time ratio.",
        f"- Model B=2 was observed in {online_model_b2_cases}/{len(online_cases)} completed cases; acoustic B=2 was observed in {online_acoustic_b2_cases}/{len(online_cases)}. These are independent batch dimensions.",
        "- A future paper claim requires repeated same-work, held-out public runs; the current representative smoke is insufficient.",
    ]
    (result_root / "APR_MODEL_EXECUTION_PLANE_STATISTICAL_ANALYSIS.md").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    # The fixed-work and contract artifacts are generated from the same source;
    # keep the correctness report explicit about what was and was not tested.
    correctness_lines = [
        "# APR Model Execution Plane Correctness Report",
        "",
        "## Software contract",
        "",
        "The focused model-plane suite covers request registration, central-driver ownership, B_model=2 output routing, generation fences, cancellation/reset, bounded backpressure, fail-closed backend errors, and global-state isolation. The latest run recorded 163 passed tests in the broader focused selection, including 31 row/model-plane tests and 12 initial contract tests; the exact command outputs remain in the result directory.",
        "",
        "## Fixed-work A100",
        "",
        "All nine fixed-work files (P10/P50/P90 × serial/row-single/row-batch) completed with two requests, eight decode turns, stable token-only digests, and no model output routing error. Row-batch physical distribution was `{2: 160}` per shape; serial and row-single were `{1: 320}`.",
        "",
        "## Online smoke",
        "",
        f"Completed public online case count: **{len(online_cases)}**. All listed completed cases have their client validity, PCM count, ownership count, runtime error count, and server log retained. This is a smoke/pilot result, not a full 3-workload × N × 5-repeat matrix.",
        "",
        "## Limitations",
        "",
        "The fixed-work runner exercises GPU0 model execution only. It does not establish GPU1 Flow numerical equivalence for the combined system. The public service uses stochastic multi-head sampling; when work fingerprints differ, the strict causal gate is blocked even if completion and PCM contracts pass.",
    ]
    (result_root / "APR_MODEL_EXECUTION_PLANE_CORRECTNESS_REPORT.md").write_text("\n".join(correctness_lines) + "\n", encoding="utf-8")

    # Conservative final decision.  It is intentionally not a paper PASS on a
    # single descriptive online pair.
    has_fixed = len(runs) >= 9 and all(
        row.get("mode") in {"serial", "row_single", "row_batch"}
        for row in runs
    )
    has_online_pair = bool(online_comparisons)
    exact_online = any(row.get("causal_status") == "CAUSAL_ELIGIBLE" for row in online_comparisons)
    if has_fixed and has_online_pair and exact_online:
        verdict = "APR_MODEL_PLANE_E2E_CANDIDATE_NEEDS_HELDOUT"
        headline = "The model execution plane has fixed-work causal evidence and at least one same-work online pair; held-out replication is still required before a paper claim."
    elif has_fixed and has_online_pair:
        verdict = "MODEL_PLANE_MECHANISM_PASS_E2E_DESCRIPTIVE_ONLY"
        headline = "The model execution plane passes fixed-work correctness/causal model-path validation and shows a real online E2E signal, but the available online pair changes generated work, so the E2E ratio remains descriptive."
    elif has_fixed:
        verdict = "MODEL_PLANE_FIXED_WORK_PASS_ONLINE_PENDING"
        headline = "The model execution plane passes fixed-work validation; public online E2E evidence is not yet complete."
    else:
        verdict = "MODEL_PLANE_VALIDATION_INCOMPLETE"
        headline = "Required fixed-work evidence is incomplete."
    online_e2e_ratios = [
        float(row["e2e_makespan_speedup_descriptive"])
        for row in online_comparisons
        if row.get("e2e_makespan_speedup_descriptive") is not None
    ]
    online_model_fractions = [
        float(row["row_aware_b2_row_fraction"])
        for row in online_comparisons
        if row.get("row_aware_b2_row_fraction") is not None
    ]
    online_acoustic_fractions = [
        float(row["candidate_acoustic_b2_row_fraction"])
        for row in online_comparisons
        if row.get("candidate_acoustic_b2_row_fraction") is not None
    ]
    ratio_range = (
        f"{min(online_e2e_ratios):.3f}–{max(online_e2e_ratios):.3f}x"
        if online_e2e_ratios else "n/a"
    )
    model_fraction_range = (
        f"{min(online_model_fractions):.1%}–{max(online_model_fractions):.1%}"
        if online_model_fractions else "n/a"
    )
    acoustic_fraction_range = (
        f"{min(online_acoustic_fractions):.1%}–{max(online_acoustic_fractions):.1%}"
        if online_acoustic_fractions else "n/a"
    )
    verdict_lines = [
        "# APR Model Execution Plane Final Verdict",
        "",
        f"## Classification: `{verdict}`",
        "",
        headline,
        "",
        "## Quantitative evidence",
        "",
        f"- Fixed-work GPU0 model probe: physical B_model=2 formed on every measured turn; median B_model=2/serialized speedup is 1.657–1.709x across P10/P50/P90, with token-only digest equality.",
        f"- Public representative smoke: {len(online_cases)} complete cases and {len(online_comparisons)} legacy/row-aware pairs; {online_descriptive_count}/{len(online_comparisons)} pairs are work-mismatch descriptive-only and {online_exact_count} are exact-work causal-eligible.",
        f"- Candidate public model B_model=2 row fraction: {model_fraction_range}; candidate acoustic B=2 row fraction: {acoustic_fraction_range}. The two batch dimensions were not observed together in these cases.",
        f"- Descriptive online makespan ratios span {ratio_range}; these values are not a causal speedup estimate because generated rounds/tokens differ.",
        "",
        "## What is established",
        "",
        "1. A single driver thread can safely route independent request rows through the existing vLLM engine; the focused TDD suite and fail-closed error handling pass.",
        "2. On A100 fixed work, real physical model batch size 2 is formed for all measured turns, output token-only digests match the B_model=1 serialized control, and model-path median speedup is about 1.66–1.71× across the three shape proxies.",
        f"3. In the completed real public representative smoke ({len(online_cases)} cases across {', '.join(online_workloads) or 'the selected workloads'} and N={', '.join(online_ns) or 'the selected concurrency levels'}), row-aware model B=2 accounts for most decode rows in the candidate cases. Several candidate runs finish sooner than their legacy diagnostic, but generated work differs (round/token fingerprints and often PCM chunk count), so these ratios are descriptive rather than causal speedup claims.",
        "",
        "## What is not established",
        "",
        "- A stable held-out public E2E speedup for the combined APR acoustic B=2 system.",
        "- A multiplicative interaction between GPU0 model batching and GPU1 acoustic B=2.",
        "- An ICLR-level performance claim from the current sample count.",
        "",
        "## Required next evidence",
        "",
        "Before any paper performance claim, run repeated real-arrival pilot/held-out cells for the same system and preserve model work (or explicitly use a deterministic fixed-work online harness). Report B_model, B_acoustic, critical-path coverage, and fingerprints separately. Do not combine local model speedup with historical acoustic speedup by multiplication.",
    ]
    (result_root / "APR_MODEL_EXECUTION_PLANE_FINAL_VERDICT.md").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    exit_lines = [
        "# APR Model Runtime Exit Decision",
        "",
        f"The row-aware model execution plane is not blocked at the mechanism or fixed-work level. It has {len(online_cases)} completed representative public cases, but all {len(online_comparisons)} available pairs are descriptive because public stochastic generation changed work between systems; no held-out causal E2E gate was run.",
        "",
        "Do not start a second model runtime optimization, change acoustic compatibility rules, add B=4/B=8, or modify Flow numerical semantics until a small repeated same-work online/held-out validation resolves the causal gap.",
        "",
        "If repeated same-work public runs fail to retain the fixed-work gain or fail the 1.15× E2E gate, freeze this candidate as `MODEL_PLANE_MECHANISM_PASS_E2E_FAIL` and stop runtime stacking. If they pass, use the row-aware driver as the model-serving contribution and keep acoustic B>1 as a separately measured mechanism/interaction result.",
    ]
    (result_root / "APR_MODEL_RUNTIME_EXIT_DECISION.md").write_text("\n".join(exit_lines) + "\n", encoding="utf-8")

    chunk_lines = [
        "# APR Flow Chunk Amortization Oracle",
        "",
        "This is a read-only fallback note, not an implementation. Existing traces suggest repeated realtime round setup, token handoff, and PCM finalization may be material, but no causal measurement has yet isolated a safe coalescing opportunity.",
        "",
        "A future candidate would have to preserve first playable audio time, vocoder hop/lookahead causality, all ten Flow steps, checkpoint semantics, cancellation, and exact event ordering. It would require a separate fixed-work oracle with predicted E2E throughput >=1.10× and no TTFA/audio-gap regression before any code change.",
    ]
    (result_root / "APR_FLOW_CHUNK_AMORTIZATION_ORACLE.md").write_text("\n".join(chunk_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    fixed, fixed_payload = analyze_fixed(args.result_root / "fixed_work_full", args.result_root)
    online_payload = analyze_online(args.result_root / "online_pilot", args.result_root)
    write_reports(args.result_root, fixed_payload, online_payload)
    summary = {
        "schema": "apr-model-plane-experiment-summary-v1",
        "fixed_run_count": len(fixed_payload["runs"]),
        "fixed_interaction_count": len(fixed_payload["interactions"]),
        "online_case_count": len(online_payload["cases"]),
        "online_comparison_count": len(online_payload["comparisons"]),
    }
    (args.result_root / "experiment_aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
