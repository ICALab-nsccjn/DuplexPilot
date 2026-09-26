"""Finalize the joint ``(B_model, B_acoustic)`` experiment.

The online runner intentionally produces one directory per attempt.  This
module turns those immutable attempts into a small set of auditable artifacts:
case and session tables, paired descriptive/causal comparisons, deterministic
bootstrap summaries, and an explicit claim boundary.  It never reads or
decodes PCM payloads and it never drops a failed attempt.

The distinction between a *matched* fixed-work comparison and a live
descriptive comparison is kept in every paired row.  In particular, a ratio
is not called causal merely because two systems used the same trace path.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


JOINT_MODES = ("J11", "J21", "J12", "J22")
PAIR_ORDER = (
    ("J11", "J21"),
    ("J11", "J12"),
    ("J11", "J22"),
    ("J21", "J22"),
    ("J12", "J22"),
    ("J21", "J12"),
)

_COUNTED_DATASET_SHAPE = {
    "HD": (3, (8, 16), 3),       # workloads, N values, counted repeats
    "FD": (2, (4, 8), 3),
    "FDB": (1, (4, 8), 2),
}


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: Any, default: int | None = None) -> int | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "matched"}


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def expected_case_count() -> int:
    """Return the number of registered *counted* cases in the pilot.

    HD contributes 3*2*4*3=72, FD contributes 2*2*4*3=48, and the limited
    v1.5 lane contributes 1*2*4*2=16, for 136 counted attempts.  Warmups are
    deliberately not included.
    """
    return sum(
        workload_count * len(n_values) * len(JOINT_MODES) * repeats
        for workload_count, n_values, repeats in _COUNTED_DATASET_SHAPE.values()
    )


def case_scope(case_id: str) -> str:
    name = Path(case_id).name
    if name.startswith("counted_"):
        return "counted"
    if name.startswith("heldout_"):
        return "heldout"
    if name.startswith("warmup_"):
        return "warmup"
    return "diagnostic"


def dataset_role(workload: str) -> tuple[str, str]:
    value = str(workload)
    if value.startswith("HD-"):
        return "HumDial-FDBench", "primary_online"
    if value.startswith("FD-"):
        return "FD-Bench-Audio-Input", "secondary_online"
    if value.startswith("FDB-"):
        return "Full-Duplex-Bench-v1.5", "behavior_limited"
    return "unknown", "unclassified"


def read_metrics(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _valid_case(row: Mapping[str, Any]) -> bool:
    return (
        _bool(row.get("valid"))
        and (_float(row.get("completion_rate"), 0.0) or 0.0) >= 1.0
        and (_int(row.get("ownership_errors"), 0) or 0) == 0
        and (_int(row.get("runtime_errors"), 0) or 0) == 0
    )


def _digest_status(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    ld = str(left.get("work_fingerprint_digest") or "")
    rd = str(right.get("work_fingerprint_digest") or "")
    if ld and rd:
        return "MATCHED" if ld == rd else "DIVERGENT"
    return "UNKNOWN"


def build_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scopes: Iterable[str] = ("counted", "heldout"),
) -> list[dict[str, Any]]:
    """Build paired four-factor comparisons without hiding invalid attempts."""
    allowed = set(scopes)
    # Scope is part of the pairing key.  Counted discovery and held-out rows
    # intentionally reuse workload/N/repeat labels, so omitting it would make
    # ``setdefault`` silently discard one evidence split.
    grouped: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        mode = str(row.get("joint_mode") or "")
        if mode not in JOINT_MODES:
            continue
        scope = case_scope(str(row.get("case_id") or ""))
        if scope not in allowed:
            continue
        key = (
            scope,
            str(row.get("workload") or ""),
            str(row.get("N") or ""),
            str(row.get("repeat") or ""),
        )
        # A duplicate case id should never occur.  If an old diagnostic row
        # collides, retain the first deterministic row rather than silently
        # replacing evidence.
        grouped[key].setdefault(mode, row)

    result: list[dict[str, Any]] = []
    for (scope, workload, n, repeat), modes in sorted(grouped.items()):
        for left_mode, right_mode in PAIR_ORDER:
            left = modes.get(left_mode)
            right = modes.get(right_mode)
            if left is None or right is None:
                continue
            left_t = _float(left.get("useful_audio_throughput"))
            right_t = _float(right.get("useful_audio_throughput"))
            left_span = _float(left.get("session_span_s"))
            right_span = _float(right.get("session_span_s"))
            left_ttfa = _float(left.get("ttfa_p95_ms"))
            right_ttfa = _float(right.get("ttfa_p95_ms"))
            left_gap = _float(left.get("audio_gap_p95_ms"))
            right_gap = _float(right.get("audio_gap_p95_ms"))
            status = _digest_status(left, right)
            result.append({
                "scope": scope,
                "workload": workload,
                "dataset_id": dataset_role(workload)[0],
                "N": n,
                "repeat": repeat,
                "comparison": f"{right_mode}/{left_mode}",
                "left_mode": left_mode,
                "right_mode": right_mode,
                "left_case": left.get("case_id"),
                "right_case": right.get("case_id"),
                "left_valid": _valid_case(left),
                "right_valid": _valid_case(right),
                "paired_valid": _valid_case(left) and _valid_case(right),
                "work_comparability": status,
                "causal_eligible": status == "MATCHED" and _valid_case(left) and _valid_case(right),
                "useful_audio_throughput_ratio": right_t / left_t if left_t and right_t is not None else None,
                "session_span_ratio": left_span / right_span if left_span and right_span else None,
                "ttfa_p95_ratio": right_ttfa / left_ttfa if left_ttfa and right_ttfa is not None else None,
                "audio_gap_p95_ratio": right_gap / left_gap if left_gap and right_gap is not None else None,
                "left_work_fingerprint_digest": left.get("work_fingerprint_digest", ""),
                "right_work_fingerprint_digest": right.get("work_fingerprint_digest", ""),
                "left_acoustic_b2_work_fraction": _float(left.get("acoustic_b2_work_fraction"), 0.0) or 0.0,
                "right_acoustic_b2_work_fraction": _float(right.get("acoustic_b2_work_fraction"), 0.0) or 0.0,
                "left_model_b2_row_fraction": _float(left.get("model_decode_b2_row_fraction"), 0.0) or 0.0,
                "right_model_b2_row_fraction": _float(right.get("model_decode_b2_row_fraction"), 0.0) or 0.0,
            })
    return result


def summarize_bootstrap_ratios(
    values: Sequence[float],
    *,
    seed: int = 20260901,
    samples: int = 10_000,
) -> dict[str, Any]:
    """Summarize paired ratios with deterministic percentile bootstrap."""
    clean = [float(value) for value in values if _float(value) is not None]
    if not clean:
        return {
            "n": 0, "median": None, "ci_low": None, "ci_high": None,
            "wins": 0, "losses": 0, "ties": 0, "samples": samples, "seed": seed,
        }
    median = statistics.median(clean)
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(max(1, int(samples))):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        boot.append(statistics.median(draw))
    return {
        "n": len(clean),
        "median": median,
        "ci_low": _percentile(boot, 0.025),
        "ci_high": _percentile(boot, 0.975),
        "wins": sum(value > 1.0 for value in clean),
        "losses": sum(value < 1.0 for value in clean),
        "ties": sum(value == 1.0 for value in clean),
        "samples": int(samples),
        "seed": int(seed),
    }


def summarize_pairs(pair_rows: Sequence[Mapping[str, Any]], *, seed: int = 20260901) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row.get("scope") or ""), str(row.get("workload") or ""), str(row.get("N") or ""), str(row.get("comparison") or ""))].append(row)
    output: list[dict[str, Any]] = []
    for (scope, workload, n, comparison), rows in sorted(grouped.items()):
        all_values = [
            _float(row.get("useful_audio_throughput_ratio"))
            for row in rows
            if _bool(row.get("paired_valid"))
        ]
        matched_values = [
            _float(row.get("useful_audio_throughput_ratio"))
            for row in rows
            if _bool(row.get("causal_eligible"))
        ]
        all_summary = summarize_bootstrap_ratios([v for v in all_values if v is not None], seed=seed)
        matched_summary = summarize_bootstrap_ratios([v for v in matched_values if v is not None], seed=seed)
        output.append({
            "scope": scope,
            "workload": workload,
            "dataset_id": dataset_role(workload)[0],
            "N": n,
            "comparison": comparison,
            "paired_rows": len(rows),
            "valid_rows": len([row for row in rows if _bool(row.get("paired_valid"))]),
            "matched_rows": len(matched_values),
            "descriptive_median": all_summary["median"],
            "descriptive_ci_low": all_summary["ci_low"],
            "descriptive_ci_high": all_summary["ci_high"],
            "descriptive_wins": all_summary["wins"],
            "causal_median": matched_summary["median"],
            "causal_ci_low": matched_summary["ci_low"],
            "causal_ci_high": matched_summary["ci_high"],
            "causal_wins": matched_summary["wins"],
            "bootstrap_samples": 10_000,
            "bootstrap_seed": seed,
        })
    return output


def classify_joint_result(
    *,
    rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    expected: int,
    fixed_work_status: str,
) -> dict[str, Any]:
    """Classify evidence without imposing a performance stopping gate.

    A positive classification requires more than a single ratio.  It also
    records whether the positive signal is workload-dependent and whether the
    live work fingerprints were matched.
    """
    valid_rows = [row for row in rows if case_scope(str(row.get("case_id") or "")) in {"counted", "heldout"} and _valid_case(row)]
    observed_b2 = [
        row for row in valid_rows
        if (_float(row.get("acoustic_b2_work_fraction"), 0.0) or 0.0) > 0.0
        or (_int(row.get("mixed_padding_complete_count"), 0) or 0) > 0
    ]
    summaries = summarize_pairs(pair_rows)
    positive_workloads: set[str] = set()
    neutral_workloads: set[str] = set()
    matched_positive = False
    for summary in summaries:
        if summary.get("comparison") != "J22/J21":
            continue
        median = _float(summary.get("descriptive_median"))
        if median is None:
            continue
        workload = str(summary.get("workload") or "")
        if median > 1.0:
            positive_workloads.add(workload)
            causal = _float(summary.get("causal_median"))
            if causal is not None and causal > 1.0:
                matched_positive = True
        else:
            neutral_workloads.add(workload)

    if not valid_rows:
        label = "JOINT_B22_INVALID_OR_BLOCKED"
    elif positive_workloads and neutral_workloads:
        label = "JOINT_B22_WORKLOAD_DEPENDENT"
    elif positive_workloads and matched_positive and fixed_work_status == "PASS":
        label = "JOINT_B22_CAUSAL_E2E_POSITIVE"
    elif observed_b2:
        label = "JOINT_B22_MECHANISM_PASS_E2E_NEUTRAL"
    else:
        label = "JOINT_B22_INVALID_OR_BLOCKED"
    return {
        "classification": label,
        "expected_counted_cases": expected,
        "observed_counted_or_heldout_rows": len(rows),
        "valid_rows": len(valid_rows),
        "observed_acoustic_b2_rows": len(observed_b2),
        "positive_workloads": sorted(positive_workloads),
        "neutral_workloads": sorted(neutral_workloads),
        "fixed_work_status": fixed_work_status,
        "no_hard_performance_gate": True,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _session_fields(session: Mapping[str, Any]) -> dict[str, Any]:
    """Extract timing metadata without retaining event audio payloads."""
    events = session.get("events") if isinstance(session.get("events"), list) else []
    pcm_times: list[int] = []
    recorded_sample_count = _int(session.get("pcm_sample_count"))
    recorded_chunk_count = _int(session.get("pcm_chunk_count", session.get("pcm_chunks")))
    # Newer clients already aggregate these values in the session envelope.
    # Only derive them from SSE events for older attempts; otherwise the
    # parquet artifact would double-count every PCM frame.
    sample_count = recorded_sample_count if recorded_sample_count is not None else 0
    chunk_count = recorded_chunk_count if recorded_chunk_count is not None else 0
    for event in events:
        if not isinstance(event, Mapping) or str(event.get("type") or "") != "audio_chunk_pcm":
            continue
        timestamp = _int(event.get("server_sse_send_epoch_ms", event.get("server_audio_emit_epoch_ms")))
        if timestamp is not None:
            pcm_times.append(timestamp)
        if recorded_sample_count is None:
            frame = event.get("frame_audio")
            if isinstance(frame, Mapping):
                sample_count += max(0, _int(frame.get("num_samples", frame.get("samples")), 0) or 0)
            else:
                sample_count += max(0, _int(event.get("num_samples", event.get("sample_count")), 0) or 0)
        if recorded_chunk_count is None:
            chunk_count += 1
    start = _int(session.get("client_start_epoch_ms"))
    first = min(pcm_times) if pcm_times else None
    last = max(pcm_times) if pcm_times else None
    done_times = [
        _int(event.get("server_sse_send_epoch_ms"))
        for event in events
        if isinstance(event, Mapping) and str(event.get("type") or "") == "done"
    ]
    done_times = [value for value in done_times if value is not None]
    return {
        "request_id": str(session.get("request_id") or session.get("id") or ""),
        "generation_id": _int(session.get("generation_id")),
        "stream_id": str(session.get("stream_id") or ""),
        "done": _bool(session.get("done")),
        "elapsed_s": _float(session.get("elapsed_s")),
        "client_start_epoch_ms": start,
        "client_end_epoch_ms": _int(session.get("client_end_epoch_ms")),
        "ttfa_ms": _float(session.get("ttfa_ms"), (first - start) if first is not None and start is not None else None),
        "pcm_sample_count": sample_count,
        "pcm_chunk_count": chunk_count,
        "audio_gap_ms": _float(session.get("audio_gap_ms"), (last - first) if first is not None and last is not None else None),
        "done_epoch_ms": _int(session.get("done_epoch_ms"), min(done_times) if done_times else None),
        "termination_reason": str(session.get("termination_reason") or session.get("finish_reason") or ""),
    }


def build_session_rows(result_root: Path, case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_case = {str(row.get("case_id")): row for row in case_rows}
    output: list[dict[str, Any]] = []
    for attempt in sorted(result_root.rglob("online_attempt.json")):
        case_root = attempt.parent.parent
        case_id = str(case_root.relative_to(result_root))
        case = by_case.get(case_id)
        if case is None:
            continue
        payload = _read_json(attempt)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("sessions"), list):
            continue
        for index, raw in enumerate(payload["sessions"]):
            if not isinstance(raw, Mapping):
                continue
            fields = _session_fields(raw)
            fields.update({
                "case_id": case_id,
                "dataset_id": dataset_role(str(case.get("workload") or ""))[0],
                "scope": case_scope(case_id),
                "system": case.get("system", ""),
                "joint_mode": case.get("joint_mode", ""),
                "workload": case.get("workload", ""),
                "N": _int(case.get("N")),
                "repeat": _int(case.get("repeat")),
                "session_index": index,
            })
            output.append(fields)
    return output


def write_session_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("pandas is required for the session parquet artifact") from exc
    frame = pd.DataFrame(list(rows))
    # pyarrow is the approved bundled writer in the execution container.
    frame.to_parquet(path, index=False, engine="pyarrow")


_OPERATOR_EVENT_TYPES = (
    "FLOW_PADDING_PREPARE",
    "FLOW_CACHE_SPLIT",
    "FLOW_BATCH_TIMING",
    "FLOW_MIXED_CHUNK_PADDING_BATCH_ATTEMPT",
    "FLOW_MIXED_CHUNK_PADDING_BATCH_COMPLETE",
    "FLOW_BATCH_COMPLETE",
)
_OPERATOR_NUMERIC_FIELDS = (
    "wall_time_ms",
    "cuda_time_ms",
    "host_overhead_ns",
    "pack_time_ms",
    "padding_time_ms",
    "mask_time_ms",
    "split_time_ms",
    "writeback_time_ms",
    "allocation_time_ms",
)


def scan_operator_events(result_root: Path, case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate existing metadata events; no operator is implemented here."""
    selected = {
        str(row.get("case_id")): row
        for row in case_rows
        if case_scope(str(row.get("case_id") or "")) in {"counted", "heldout"}
        and str(row.get("joint_mode") or "") in {"J12", "J22"}
    }
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}
    for case_id, case in sorted(selected.items()):
        case_root = result_root / case_id
        counts: Counter[str] = Counter()
        sums: Counter[str] = Counter()
        present: Counter[str] = Counter()
        for filename in ("online_trace.jsonl", "model_execution_trace.jsonl", "opportunity_trace.jsonl"):
            path = case_root / filename
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, Mapping):
                        continue
                    event_type = str(event.get("event_type") or event.get("event") or "").upper()
                    if event_type not in _OPERATOR_EVENT_TYPES:
                        continue
                    counts[event_type] += 1
                    for field in _OPERATOR_NUMERIC_FIELDS:
                        number = _float(event.get(field))
                        if number is not None:
                            sums[field] += number
                            present[field] += 1
        for event_type in _OPERATOR_EVENT_TYPES:
            key = (str(case.get("workload") or ""), str(case.get("N") or ""), event_type)
            row = aggregate.setdefault(key, {
                "dataset_id": dataset_role(str(case.get("workload") or ""))[0],
                "workload": case.get("workload", ""),
                "N": case.get("N", ""),
                "event_type": event_type,
                "case_count": 0,
                "event_count": 0,
            })
            row["case_count"] += 1
            row["event_count"] += counts[event_type]
            for field in _OPERATOR_NUMERIC_FIELDS:
                row[f"{field}_sum"] = row.get(f"{field}_sum", 0.0) + sums[field]
                row[f"{field}_observations"] = row.get(f"{field}_observations", 0) + present[field]
    return list(aggregate.values())


def _fixed_work_status(path: Path | None) -> tuple[str, dict[str, Any]]:
    if path is None or not path.exists():
        return "NOT_AVAILABLE", {"source": str(path) if path else ""}
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return "BLOCKED", {"source": str(path), "reason": "invalid_json"}
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    passes: list[bool] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        comparison = record.get("comparison")
        if isinstance(comparison, Mapping) and "all_pcm_contract_pass" in comparison:
            passes.append(_bool(comparison.get("all_pcm_contract_pass")))
        else:
            passes.append(_bool(record.get("pcm_contract_pass")))
    status = "PASS" if records and all(passes) else ("PARTIAL" if records else "BLOCKED")
    return status, {
        "source": str(path),
        "record_count": len(records),
        "pcm_contract_pass_count": sum(passes),
        "schema": payload.get("schema", ""),
    }


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_statistical_report(
    path: Path,
    pair_summaries: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
) -> None:
    lines = [
        "# APR Joint B22 Statistical Analysis",
        "",
        "All paired rows and failed attempts remain in the underlying CSV. Ratios are grouped by the same workload, N, and repeat. A `MATCHED` work fingerprint is required for the causal subset; all other ratios are explicitly descriptive.",
        "",
        "Bootstrap: deterministic paired-ratio median, 10,000 resamples, seed 20260901. No performance threshold was used to stop data collection.",
        "",
        "## Paired summaries",
        "",
        "| scope | dataset/workload | N | comparison | valid | matched | descriptive median [95% CI] | causal median [95% CI] | descriptive wins | causal wins |",
        "|---|---|---:|---|---:|---:|---|---|---:|---:|",
    ]
    for row in pair_summaries:
        def fmt(prefix: str) -> str:
            med = _float(row.get(f"{prefix}_median"))
            lo = _float(row.get(f"{prefix}_ci_low"))
            hi = _float(row.get(f"{prefix}_ci_high"))
            return "NA" if med is None else f"{med:.4f} [{lo:.4f}, {hi:.4f}]"
        lines.append(
            f"| {row.get('scope','')} | {row.get('dataset_id','')}/{row.get('workload','')} | {row.get('N','')} | {row.get('comparison','')} | {row.get('valid_rows',0)} | {row.get('matched_rows',0)} | {fmt('descriptive')} | {fmt('causal')} | {row.get('descriptive_wins',0)} | {row.get('causal_wins',0)} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Classification at the time of generation: `{classification.get('classification')}`.",
        "- A ratio above 1 means the right-hand configuration has higher useful-audio throughput; it is not a percentage unless converted as `(ratio-1)*100`.",
        "- Live token/termination divergence is retained as descriptive evidence. Component and local Flow speedups are not multiplied.",
    ])
    _write_markdown(path, lines)


def _write_dataset_report(path: Path, rows: Sequence[Mapping[str, Any]], result_root: Path) -> None:
    counts: Counter[tuple[str, str, str]] = Counter()
    valid: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        scope = case_scope(str(row.get("case_id") or ""))
        if scope not in {"counted", "heldout"}:
            continue
        key = (dataset_role(str(row.get("workload") or ""))[0], str(row.get("workload") or ""), str(row.get("joint_mode") or ""))
        counts[key] += 1
        valid[key] += int(_valid_case(row))
    lines = [
        "# APR Joint B22 Dataset Comparison",
        "",
        "This table describes the data lanes actually present under the result root. HD is the primary online lane; FD-Bench-Audio-Input is a secondary audio source; Full-Duplex-Bench v1.5 is a limited behavior/example lane and is not silently treated as a full throughput benchmark.",
        "",
        "BurstGPT-derived timing is an arrival-pattern source, not an independent audio dataset.",
        "",
        "| dataset role | workload | joint | counted/heldout attempts | valid attempts |",
        "|---|---|---|---:|---:|",
    ]
    for (dataset, workload, mode), count in sorted(counts.items()):
        lines.append(f"| {dataset_role(workload)[1]} ({dataset}) | {workload} | {mode} | {count} | {valid[(dataset, workload, mode)]} |")
    lines.extend([
        "",
        "## Comparability notes",
        "",
        "- Every case must use the trace's original arrival/audio timeline; no artificial barrier or system-specific retiming is allowed.",
        "- FD-Bench and Full-Duplex-Bench are benchmark/data sources, not external serving systems. Their original licenses and source hashes are recorded in `APR_JOINT_B22_DATASET_MANIFEST.json`.",
        "- A dataset with protocol or PCM-lifecycle failures remains visible but is excluded from causal interpretation.",
        f"- Result root: `{result_root}`.",
    ])
    _write_markdown(path, lines)


def _write_operator_report(path: Path, operator_rows: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> None:
    mixed_cases = [
        row for row in rows
        if case_scope(str(row.get("case_id") or "")) in {"counted", "heldout"}
        and str(row.get("joint_mode") or "") in {"J12", "J22"}
    ]
    mixed_completes = sum(_int(row.get("mixed_padding_complete_count"), 0) or 0 for row in mixed_cases)
    lines = [
        "# APR Mixed-Chunk-Padding B2 Operator Report",
        "",
        "Status: `NOT_IMPLEMENTED_ON_THIS_BRANCH`. The joint experiment uses the frozen `mixed_chunk_padding_b2` public path. This artifact only attributes observed metadata overhead; it does not introduce a second operator or alter the experiment while it is running.",
        "",
        f"- Counted/heldout J12/J22 case rows observed: **{len(mixed_cases)}**",
        f"- Mixed-padding completed batches observed: **{mixed_completes}**",
        "- Full tensor, PCM and audio payloads were not inspected.",
        "",
        "## Observed event aggregates",
        "",
        "| workload | N | event | cases | events | wall ms sum | CUDA ms sum | host overhead ns sum |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(operator_rows, key=lambda item: (str(item.get("workload")), str(item.get("N")), str(item.get("event_type")))):
        def val(name: str) -> str:
            number = _float(row.get(name))
            return "NA" if number is None or number == 0 else f"{number:.3f}"
        lines.append(f"| {row.get('workload','')} | {row.get('N','')} | {row.get('event_type','')} | {row.get('case_count',0)} | {row.get('event_count',0)} | {val('wall_time_ms_sum')} | {val('cuda_time_ms_sum')} | {val('host_overhead_ns_sum')} |")
    has_breakdown = any(
        (_int(row.get("wall_time_ms_observations"), 0) or 0)
        + (_int(row.get("pack_time_ms_observations"), 0) or 0)
        + (_int(row.get("split_time_ms_observations"), 0) or 0)
        > 0
        for row in operator_rows
    )
    lines.extend([
        "",
        "## Decision boundary",
        "",
        "- The current branch contains no mixed-padding-specific execution operator. A follow-up operator is justified only if these counters show a dominant avoidable pack/pad/mask/split/allocation component and the same-work fixed-work lane confirms that the component, rather than changed model work, explains the loss.",
        f"- Separately measurable sub-stage timing fields present: **{has_breakdown}**.",
        "- If sub-stage fields are absent, the report must not infer that all `FLOW_BATCH_TIMING` is padding overhead; it is only aggregate Flow timing.",
        "- Any operator follow-up must start from a separate stable joint commit, use TDD, and retain J12/J22 no-operator controls.",
    ])
    _write_markdown(path, lines)


def _write_e2e_report(path: Path, rows: Sequence[Mapping[str, Any]], pair_summaries: Sequence[Mapping[str, Any]], classification: Mapping[str, Any], fixed_meta: Mapping[str, Any]) -> None:
    valid = [row for row in rows if case_scope(str(row.get("case_id") or "")) in {"counted", "heldout"}]
    b2 = [row for row in valid if (_float(row.get("acoustic_b2_work_fraction"), 0.0) or 0.0) > 0.0]
    monitor_rows = [
        row for row in valid
        if str(row.get("gpu_monitor_status") or "") == "OBSERVED"
    ]
    gpu1_peaks = [
        _float(row.get("gpu1_peak_memory_mib"))
        for row in monitor_rows
        if _float(row.get("gpu1_peak_memory_mib")) is not None
    ]
    lines = [
        "# APR Joint B22 Online E2E Report",
        "",
        "## Executive result",
        "",
        f"The current evidence classification is **`{classification.get('classification')}`**. Data collection uses the user-requested open reporting policy: no `(2,2)` formation-rate or 10% performance gate was used to stop the matrix.",
        "",
        f"- Counted/heldout case rows available: **{len(valid)}**; expected counted pilot cases: **{expected_case_count()}**.",
        f"- Valid case rows: **{sum(_valid_case(row) for row in valid)}**.",
        f"- Rows with observed acoustic B=2 work: **{len(b2)}**.",
        f"- Fixed-work acoustic artifact status: **{fixed_meta.get('status','NOT_AVAILABLE')}**.",
        f"- Runtime GPU monitor coverage: **{len(monitor_rows)}/{len(valid)}** case rows; older attempts without in-run sampling remain explicitly unmeasured.",
        f"- Observed GPU1 peak across monitored rows: **{max(gpu1_peaks):.1f} MiB**." if gpu1_peaks else "- Observed GPU1 peak across monitored rows: **NA**.",
        "",
        "## Four configuration meanings",
        "",
        "| joint | model | acoustic | interpretation |",
        "|---|---:|---:|---|",
        "| J11 | 1 | 1 | legacy serialized model + frozen B=1 Flow |",
        "| J21 | 2 | 1 | row-aware model cap=2 + frozen B=1 Flow |",
        "| J12 | 1 | 2 | legacy model + frozen `mixed_chunk_padding_b2` |",
        "| J22 | 2 | 2 | row-aware model cap=2 + frozen `mixed_chunk_padding_b2` |",
        "",
        "## Paired comparison reading",
        "",
        "The relevant acoustic incremental comparison is `J22/J21`; model incremental comparison is `J22/J12`; full joint comparison is `J22/J11`. Ratios in this report are useful-audio-throughput ratios, not raw Flow speedups. `DIVERGENT` live work fingerprints remain descriptive.",
        "",
        "| workload | N | comparison | valid | matched | descriptive median | causal median |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in pair_summaries:
        if row.get("comparison") not in {"J22/J21", "J22/J12", "J22/J11", "J21/J11", "J12/J11"}:
            continue
        d = _float(row.get("descriptive_median"))
        c = _float(row.get("causal_median"))
        lines.append(f"| {row.get('workload','')} | {row.get('N','')} | {row.get('comparison','')} | {row.get('valid_rows',0)} | {row.get('matched_rows',0)} | {'NA' if d is None else f'{d:.4f}'} | {'NA' if c is None else f'{c:.4f}'} |")
    lines.extend([
        "",
        "## Validity and claim boundary",
        "",
        "- Correctness, PCM, flush, cancel, cleanup, identity and memory status are reported per attempt; invalid attempts are not silently replaced.",
        "- Fixed-work measurements and live online measurements are separate lanes. Component speedups are not multiplied to synthesize a joint E2E number.",
        "- A stable positive result on J21/J11 alone supports model-side row-aware serving, not acoustic B>1. A positive J22/J21 result is the evidence needed for an acoustic B>1 interaction.",
        "- Full-Duplex-Bench v1.5 examples with interruption/PCM lifecycle limitations are not promoted to a throughput claim.",
    ])
    _write_markdown(path, lines)


def _write_verdict_files(
    result_root: Path,
    classification: Mapping[str, Any],
    fixed_meta: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    pair_summaries: Sequence[Mapping[str, Any]],
) -> None:
    label = str(classification.get("classification"))
    b2_rows = sum(
        1 for row in rows
        if case_scope(str(row.get("case_id") or "")) in {"counted", "heldout"}
        and (_float(row.get("acoustic_b2_work_fraction"), 0.0) or 0.0) > 0.0
    )
    verdict = [
        "# APR Joint B22 Final Verdict",
        "",
        f"**Classification:** `{label}`",
        "",
        "This verdict is an evidence classification, not a pre-registered stopping gate. The four cells were measured/reported with the user's requested open threshold policy.",
        "",
        "## Facts",
        "",
        f"- Fixed-work acoustic evidence: `{fixed_meta.get('status','NOT_AVAILABLE')}`.",
        f"- Online rows with observed acoustic B=2: `{b2_rows}`.",
        f"- Expected counted cases: `{expected_case_count()}`; observed counted/heldout rows: `{classification.get('observed_counted_or_heldout_rows')}`.",
        f"- Positive workload subsets from J22/J21 descriptive ratios: `{', '.join(classification.get('positive_workloads', [])) or 'none'}`.",
        f"- Neutral/non-positive workload subsets: `{', '.join(classification.get('neutral_workloads', [])) or 'none'}`.",
        "",
        "## Interpretation",
        "",
        "The result must be read together with `APR_JOINT_B22_STATISTICAL_ANALYSIS.md`. In particular, live ratios with divergent work fingerprints are descriptive. The classification does not turn a local Flow or model component ratio into an E2E claim.",
        "",
        "## Route decision",
        "",
    ]
    if label == "JOINT_B22_CAUSAL_E2E_POSITIVE":
        verdict.append("The combined `(2,2)` path has a matched fixed-work and live positive signal; preserve the exact workload/configuration boundary and run no broader claim without additional held-out evidence.")
    elif label == "JOINT_B22_WORKLOAD_DEPENDENT":
        verdict.append("The combined path is positive only for a subset of observed workload conditions. Treat it as workload-dependent and report both positive and neutral/negative cells.")
    elif label == "JOINT_B22_MECHANISM_PASS_E2E_NEUTRAL":
        verdict.append("Acoustic B=2 forms and remains valid, but the current data do not establish a stable E2E increment over J21. Preserve it as mechanism evidence; do not claim universal throughput gain.")
    else:
        verdict.append("The current evidence is insufficient or invalid for a joint E2E claim; retain all failed attempts and explain the blocking validity/workload issue.")
    _write_markdown(result_root / "APR_JOINT_B22_FINAL_VERDICT.md", verdict)

    claim = [
        "# APR Joint B22 Claim Boundary",
        "",
        "## Allowed",
        "",
        "- APR exposes independent model and acoustic execution planes and can execute the frozen row-aware model B=2 and `mixed_chunk_padding_b2` acoustic B=2 mechanisms.",
        "- The exact observed formation rates, critical-path fractions, memory envelope, correctness and workload dependence may be reported.",
        "",
        "## Conditional",
        "",
        "- `(2,2)` may be presented as an E2E performance interaction only when the same-work lane and live lane agree and the relevant paired fingerprints are matched.",
        "- A positive J21/J11 result is model execution-plane evidence; it is not evidence that acoustic B=2 accelerates the system.",
        "",
        "## Not allowed",
        "",
        "- Do not multiply model and acoustic local speedups.",
        "- Do not call a divergent live ratio causal.",
        "- Do not describe a controlled B=2 Flow speedup as online throughput without critical-path and work-fingerprint evidence.",
        "- Do not generalize a positive dataset/workload subset to all public loads.",
    ]
    _write_markdown(result_root / "APR_JOINT_B22_CLAIM_BOUNDARY.md", claim)

    route = [
        "# APR Joint B22 Route Decision",
        "",
        f"Current classification: `{label}`.",
        "",
        "The user-requested experiment policy was followed: no early performance gate stopped the four-cell matrix. The next decision is evidence-based:",
        "",
        "1. If J22/J21 is positive in matched same-work and live held-out cells, preserve this exact combined route as the APR performance candidate.",
        "2. If only a subset is positive, freeze the combined mechanism and document workload dependence rather than adding unbounded compatibility rules.",
        "3. If J22 forms but its E2E increment is neutral, retain B>1 as a verified capability and do not infer a universal throughput benefit.",
        "4. A mixed-padding operator may be considered only after the operator report identifies a measured avoidable pack/pad/mask/split component; it must be a separate branch with J12/J22 controls.",
        "",
        "No new B, N, wait-window, padding rule, Graph or Inductor change is implicitly authorized by this report.",
    ]
    _write_markdown(result_root / "APR_JOINT_B22_ROUTE_DECISION.md", route)


def finalize(
    result_root: Path,
    *,
    fixed_work_json: Path | None = None,
    refresh_analyzer: bool = True,
    seed: int = 20260901,
) -> dict[str, Any]:
    result_root = Path(result_root)
    metrics_path = result_root / "APR_JOINT_B22_ONLINE_METRICS.csv"
    if refresh_analyzer or not metrics_path.exists():
        from tools.analysis.analyze_joint_b22 import write_reports
        write_reports(result_root)
    rows = read_metrics(metrics_path)
    pair_rows = build_pair_rows(rows)
    pair_summaries = summarize_pairs(pair_rows, seed=seed)
    fixed_status, fixed_meta = _fixed_work_status(fixed_work_json)
    classification = classify_joint_result(
        rows=rows,
        pair_rows=pair_rows,
        expected=expected_case_count(),
        fixed_work_status=fixed_status,
    )

    counted_rows = [
        dict(row, scope=case_scope(str(row.get("case_id") or "")), dataset_id=dataset_role(str(row.get("workload") or ""))[0])
        for row in rows
        if case_scope(str(row.get("case_id") or "")) in {"counted", "heldout"}
    ]
    _write_csv(result_root / "APR_JOINT_B22_PRIMARY_METRICS.csv", counted_rows)
    _write_csv(result_root / "APR_JOINT_B22_ABLATION_METRICS.csv", pair_rows)
    _write_csv(result_root / "APR_JOINT_B22_ONLINE_STATISTICAL_SUMMARIES.csv", pair_summaries)

    sessions = build_session_rows(result_root, rows)
    write_session_parquet(result_root / "APR_JOINT_B22_ONLINE_SESSION_METRICS.parquet", sessions)

    operator_rows = scan_operator_events(result_root, rows)
    _write_csv(result_root / "APR_MIXED_PADDING_B2_OPERATOR_METRICS.csv", operator_rows)
    _write_operator_report(result_root / "APR_MIXED_PADDING_B2_OPERATOR_REPORT.md", operator_rows, rows)
    _write_dataset_report(result_root / "APR_JOINT_B22_DATASET_COMPARISON.md", rows, result_root)
    _write_statistical_report(result_root / "APR_JOINT_B22_STATISTICAL_ANALYSIS.md", pair_summaries, classification)
    _write_e2e_report(result_root / "APR_JOINT_B22_E2E_REPORT.md", rows, pair_summaries, classification, {"status": fixed_status, **fixed_meta})
    _write_verdict_files(result_root, classification, {"status": fixed_status, **fixed_meta}, rows, pair_summaries)
    return {
        "case_rows": len(rows),
        "counted_or_heldout_rows": len(counted_rows),
        "pair_rows": len(pair_rows),
        "session_rows": len(sessions),
        "classification": classification,
        "fixed_work": {"status": fixed_status, **fixed_meta},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--fixed-work-json", type=Path)
    parser.add_argument("--skip-analyzer", action="store_true")
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    result = finalize(
        args.result_root,
        fixed_work_json=args.fixed_work_json,
        refresh_analyzer=not args.skip_analyzer,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
