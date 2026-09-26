#!/usr/bin/env python3
"""Summarize paced CAP0 calibration attempts without changing their evidence.

The online runner deliberately stores one large JSON object per attempt.  This
tool reads those immutable artifacts, extracts bounded scalar metrics, and
writes a compact CSV/Markdown summary.  It never chooses or removes an
attempt, and it does not infer SLOs from an elastic candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _gpu1_peak_mib(root: Path) -> float | None:
    """Read the measured GPU1 high-water mark, if the attempt recorded one."""
    peak_path = root / "gpu_peak.csv"
    if peak_path.exists():
        text = peak_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"gpu\s*=\s*1\s*,\s*memory_used_mib\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        if match:
            return float(match.group(1))
    monitor_path = root / "gpu_monitor.csv"
    if not monitor_path.exists():
        return None
    maximum: float | None = None
    for line in monitor_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4 or fields[0] == "epoch_s":
            continue
        try:
            if int(fields[1]) != 1:
                continue
            value = float(fields[2])
        except (TypeError, ValueError):
            continue
        maximum = value if maximum is None else max(maximum, value)
    return maximum


def _quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _audio_timing(events: list[Any]) -> tuple[float | None, list[float]]:
    """Extract direct protocol-level first-audio and inter-emission timings."""
    start_values = [
        float(event["server_sse_send_epoch_ms"])
        for event in events
        if isinstance(event, dict)
        and event.get("event_type") == "status"
        and isinstance(event.get("server_sse_send_epoch_ms"), (int, float))
        and str(event.get("status", "")).lower().startswith("realtime session started")
    ]
    if not start_values:
        start_values = [
            float(event["server_sse_send_epoch_ms"])
            for event in events
            if isinstance(event, dict)
            and isinstance(event.get("server_sse_send_epoch_ms"), (int, float))
        ]
    pcm_times = sorted(
        float(event["server_audio_emit_epoch_ms"])
        for event in events
        if isinstance(event, dict)
        and event.get("event_type") == "audio_chunk_pcm"
        and isinstance(event.get("server_audio_emit_epoch_ms"), (int, float))
    )
    if not pcm_times:
        return None, []
    start_ms = start_values[0] if start_values else pcm_times[0]
    first_latency = max(0.0, pcm_times[0] - start_ms)
    gaps = [later - earlier for earlier, later in zip(pcm_times, pcm_times[1:])]
    return first_latency, [gap for gap in gaps if gap >= 0.0]


def _session_scalars(session: dict[str, Any]) -> dict[str, Any]:
    events = session.get("events")
    if not isinstance(events, list):
        events = []
    stage = [
        event
        for event in events
        if isinstance(event, dict) and event.get("event_type") == "stage_timing"
    ]
    round_ms = [
        float(event["latency"]["total_round_ms"])
        for event in stage
        if isinstance(event.get("latency"), dict)
        and isinstance(event["latency"].get("total_round_ms"), (int, float))
    ]
    first_pcm_ms = [
        float(event["latency"]["first_pcm_out_ms"])
        for event in stage
        if isinstance(event.get("latency"), dict)
        and isinstance(event["latency"].get("first_pcm_out_ms"), (int, float))
    ]
    first_audio_ms, audio_gaps_ms = _audio_timing(events)
    return {
        "done": bool(session.get("done")),
        "accepted": bool(session.get("accepted")),
        "pcm_bytes": int(session.get("pcm_bytes") or 0),
        "pcm_chunks": int(session.get("pcm_chunks") or 0),
        "sent_chunks": int(session.get("sent_chunks") or 0),
        "elapsed_s": float(session.get("elapsed_s") or 0.0),
        "round_count": len(stage),
        "round_p50_ms": _quantile(round_ms, 0.50) or 0.0,
        "round_p95_ms": _quantile(round_ms, 0.95) or 0.0,
        "first_pcm_p95_ms": _quantile(first_pcm_ms, 0.95) or 0.0,
        "first_audio_latency_ms": first_audio_ms,
        "audio_gap_p95_ms": _quantile(audio_gaps_ms, 0.95) or 0.0,
        "_audio_gap_values_ms": audio_gaps_ms,
    }


def _attempt_row(path: Path) -> dict[str, Any]:
    root = path.parent
    metadata = _read_json(root / "run_metadata.json") or {}
    strict = _read_json(root / "strict_summary.json") or {}
    attempt = _read_json(root / "client" / "online_attempt.json") or {}
    sessions = attempt.get("sessions")
    if not isinstance(sessions, list):
        sessions = []
    scalars = [_session_scalars(item) for item in sessions if isinstance(item, dict)]
    elapsed = [float(item["elapsed_s"]) for item in scalars]
    round_p95 = [float(item["round_p95_ms"]) for item in scalars]
    first_p95 = [float(item["first_pcm_p95_ms"]) for item in scalars]
    first_audio = [
        float(item["first_audio_latency_ms"])
        for item in scalars
        if isinstance(item.get("first_audio_latency_ms"), (int, float))
    ]
    all_audio_gaps = [
        float(gap)
        for item in scalars
        for gap in item.get("_audio_gap_values_ms", [])
    ]
    workload = str(metadata.get("workload", ""))
    match = re.search(r"U(\d+)", workload)
    if match is None:
        match = re.search(r"U(\d+)", str(metadata.get("trace", "")))
    load_fraction = (int(match.group(1)) / 100.0) if match else 0.0
    trace_path = Path(str(metadata.get("trace", "")))
    interarrival_s = 0.0
    provenance = _read_json(Path(str(trace_path) + ".provenance.json")) if trace_path else None
    if provenance and isinstance(provenance.get("schedule"), dict):
        interarrival_s = float(provenance["schedule"].get("interarrival_s") or 0.0)
    expected = int(
        metadata.get("expected_session_count") or attempt.get("session_count") or 0
    )
    completed = int(strict.get("completed_sessions") or attempt.get("completed_sessions") or 0)
    strict_pass = bool(strict.get("strict_pass"))
    attempt_valid = bool(attempt.get("valid"))
    gpu1_peak_mib = _gpu1_peak_mib(root)
    gpu1_peak_gib = gpu1_peak_mib / 1024.0 if gpu1_peak_mib is not None else None
    memory_envelope_pass = gpu1_peak_mib is not None and gpu1_peak_mib <= 32.0 * 1024.0
    semantic_pass = strict_pass and attempt_valid and completed == expected
    valid = semantic_pass and memory_envelope_pass
    if not semantic_pass:
        attempt_status = "INCOMPLETE_OR_FAILED"
    elif gpu1_peak_mib is None:
        attempt_status = "MEMORY_PEAK_UNKNOWN"
    elif not memory_envelope_pass:
        attempt_status = "MEMORY_ENVELOPE_EXCEEDED"
    else:
        attempt_status = "PASS"
    return {
        "attempt_dir": str(root),
        "repeat": int(metadata.get("repeat") or attempt.get("repeat") or 0),
        "workload": workload,
        "system": metadata.get("system", attempt.get("system", "")),
        "trace": metadata.get("trace", ""),
        "load_fraction": load_fraction,
        "interarrival_s": interarrival_s,
        "expected_sessions": expected,
        "observed_sessions": len(scalars),
        "completed_sessions": completed,
        "missing_pcm_sessions": int(strict.get("missing_pcm_sessions") or 0),
        "pcm_chunks": int(strict.get("pcm_chunks") or attempt.get("pcm_chunks") or 0),
        "ownership_errors": int(strict.get("ownership_errors") or attempt.get("ownership_errors") or 0),
        "runtime_errors": int(strict.get("runtime_errors") or attempt.get("runtime_errors") or 0),
        "strict_pass": strict_pass,
        "attempt_valid": attempt_valid,
        "semantic_pass": semantic_pass,
        "gpu1_peak_memory_gib": gpu1_peak_gib,
        "gpu_peak_known": gpu1_peak_mib is not None,
        "memory_envelope_pass": memory_envelope_pass,
        "valid": valid,
        "attempt_status": attempt_status,
        "session_elapsed_p50_s": _quantile(elapsed, 0.50) or 0.0,
        "session_elapsed_p95_s": _quantile(elapsed, 0.95) or 0.0,
        "round_latency_p50_ms": _quantile(round_p95, 0.50) or 0.0,
        "round_latency_p95_ms": _quantile(round_p95, 0.95) or 0.0,
        "first_pcm_p95_ms": _quantile(first_p95, 0.95) or 0.0,
        "first_audio_p95_ms": _quantile(first_audio, 0.95) or 0.0,
        "audio_gap_p95_ms": _quantile(all_audio_gaps, 0.95) or 0.0,
    }


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt_dir in sorted(root.glob("calibration/cap0_*")):
        if not attempt_dir.is_dir():
            continue
        marker = attempt_dir / "run_metadata.json"
        if not marker.exists() and not (attempt_dir / "launcher.log").exists():
            continue
        row = _attempt_row(attempt_dir / "strict_summary.json")
        rows.append(row)
    return rows


def write_outputs(rows: list[dict[str, Any]], root: Path) -> tuple[Path, Path]:
    csv_path = root / "APR_CORRECTED_CAPACITY_CALIBRATION.csv"
    fields = list(rows[0]) if rows else [
        "attempt_dir", "repeat", "workload", "system", "trace", "load_fraction",
        "interarrival_s",
        "expected_sessions", "observed_sessions", "completed_sessions",
        "missing_pcm_sessions", "pcm_chunks", "ownership_errors",
        "runtime_errors", "strict_pass", "attempt_valid", "semantic_pass",
        "gpu1_peak_memory_gib", "gpu_peak_known", "memory_envelope_pass",
        "valid", "session_elapsed_p50_s",
        "session_elapsed_p95_s", "round_latency_p50_ms", "round_latency_p95_ms",
        "first_pcm_p95_ms", "first_audio_p95_ms", "audio_gap_p95_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    valid = [
        row
        for row in rows
        if row["strict_pass"]
        and row["valid"]
        and row["completed_sessions"] == row["expected_sessions"]
    ]
    by_load: dict[float, list[dict[str, Any]]] = {}
    for row in valid:
        by_load.setdefault(float(row["load_fraction"]), []).append(row)
    stable_loads = sorted(
        load for load, attempts in by_load.items() if len(attempts) >= 3
    )
    selected_load = stable_loads[-1] if stable_loads else None
    selected = (
        [row for row in valid if float(row["load_fraction"]) == selected_load]
        if selected_load is not None
        else []
    )
    # Freeze thresholds only from valid CAP0 observations.  The report exposes
    # the rule and values; candidates must not feed back into this calculation.
    direct_first_values = [
        row["first_audio_p95_ms"] for row in selected if row["first_audio_p95_ms"] > 0
    ]
    direct_gap_values = [
        row["audio_gap_p95_ms"] for row in selected if row["audio_gap_p95_ms"] > 0
    ]
    if direct_first_values:
        ttfa_slo = 1.05 * (_quantile(direct_first_values, 0.95) or 0.0)
        ttfa_rule = "direct protocol first-playable-audio latency"
    else:
        elapsed_values = [row["session_elapsed_p95_s"] for row in selected]
        ttfa_slo = 1.05 * ((_quantile(elapsed_values, 0.95) or 0.0) * 1000.0)
        ttfa_rule = "conservative session elapsed proxy (no direct PCM timestamp)"
    if direct_gap_values:
        gap_slo = 1.05 * (_quantile(direct_gap_values, 0.95) or 0.0)
        gap_rule = "direct protocol PCM emission gaps"
    else:
        round_values = [row["round_latency_p95_ms"] for row in selected]
        gap_slo = 1.05 * (_quantile(round_values, 0.95) or 0.0)
        gap_rule = "stage-timing round proxy (no direct PCM gaps)"
    lines = [
        "# APR Corrected Capacity Calibration",
        "",
        "This report summarizes immutable, paced CAP0 physical-affinity attempts. It is a calibration artifact, not an elastic-system result.",
        "",
        f"- attempts discovered: {len(rows)}",
        f"- valid attempts (strict + measured GPU1 <=32 GiB): {len(valid)}",
        "- trace protocol: fixed public HumDial-derived timestamps; no all-at-once injection, barrier, or per-system retiming",
        "- candidate systems are excluded from SLO threshold calculation",
        "- a load point is called stable only when three valid CAP0 attempts are present; the highest stable point is selected",
        "",
        "## Attempt summary",
        "",
        "| load | repeat | expected | observed | completed | missing PCM | GPU1 peak (GiB) | status | elapsed p50 (s) | elapsed p95 (s) | round p95 (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        peak_text = (
            f"{row['gpu1_peak_memory_gib']:.3f}"
            if row["gpu1_peak_memory_gib"] is not None
            else "unknown"
        )
        lines.append(
            f"| {row['load_fraction']:.2f} | {row['repeat']} | {row['expected_sessions']} | {row['observed_sessions']} | "
            f"{row['completed_sessions']} | {row['missing_pcm_sessions']} | "
            f"{peak_text} | "
            f"{row['attempt_status']} | "
            f"{row['session_elapsed_p50_s']:.3f} | {row['session_elapsed_p95_s']:.3f} | "
            f"{row['round_latency_p95_ms']:.3f} |"
        )
    lines += [
        "",
        "## Frozen calibration rule",
        "",
        "The registered rule is `TTFA_SLO = 1.05 × CAP0 calibration p95` and `AUDIO_GAP_SLO = 1.05 × CAP0 calibration p95`. Direct protocol timestamps are preferred; a proxy is used only when an attempt has no usable PCM timestamp and is labeled explicitly.",
        "",
        f"- stable load points (>=3 valid attempts): **{', '.join(f'{load:.2f}' for load in stable_loads) or 'none'}**",
        f"- selected highest stable load: **{selected_load if selected_load is not None else 'none'}**",
        f"- frozen TTFA SLO (ms): **{ttfa_slo:.6f}** ({ttfa_rule})",
        f"- frozen audio-gap SLO (ms): **{gap_slo:.6f}** ({gap_rule})",
        f"- usable for pilot: **{'YES' if selected else 'NO'}**",
        "",
        "## Interpretation",
        "",
        "A missing, failed, memory-unknown, or over-envelope attempt remains in the CSV and does not lower or replace the threshold. The GPU1 envelope is 32 GiB; an attempt without a measured high-water mark is not eligible for SLO freezing. If the 95% offered-load point is not stable, it must be recorded as a capacity ceiling and the highest stable point is the only eligible calibration source.",
        "",
    ]
    md_path = root / "APR_CORRECTED_CAPACITY_CALIBRATION.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    write_outputs(rows, args.root)
    print(json.dumps({"attempts": len(rows), "valid": sum(bool(row["valid"]) for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
