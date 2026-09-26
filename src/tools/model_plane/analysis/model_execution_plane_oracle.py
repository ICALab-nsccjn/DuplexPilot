#!/usr/bin/env python3
"""Run the metadata-only model execution-plane Oracle.

This command never imports the model or changes a serving process.  It reads
JSONL traces and an isolated model service probe, then writes one CSV row per
Oracle policy.  The output is an analysis artifact, not a benchmark result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from profiling.model_execution_plane.oracle import (
    ServiceCurve,
    analyze_model_trace,
    e2e_intervals_from_trace,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def _probe_median(path: Path, expected_batch: int) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(item["elapsed_ns"])
        for item in payload.get("steps", [])
        if int(item.get("batch_size", expected_batch)) == expected_batch
        and float(item.get("elapsed_ns", 0)) > 0
    ]
    if not values:
        raise ValueError(f"probe {path} has no batch-{expected_batch} samples")
    return float(median(values))


def _stable_fingerprint(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a descriptive fixed-work fingerprint from metadata only."""
    event_names = [str(row.get("event_type", row.get("event", ""))) for row in records]
    payload = "\n".join(event_names).encode("utf-8")
    return {
        "model_event_schema_hash": hashlib.sha256(payload).hexdigest(),
        "model_step_event_count": sum(name == "MODEL_ENGINE_STEP_END" for name in event_names),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-trace", type=Path, required=True)
    parser.add_argument("--acoustic-trace", type=Path)
    parser.add_argument("--probe-b1", type=Path, required=True)
    parser.add_argument("--probe-b2", type=Path, required=True)
    parser.add_argument("--workload", default="unknown")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument(
        "--assume-work-preserved",
        action="store_true",
        help="explicitly mark the fixed-work replay assumption; never infer it",
    )
    args = parser.parse_args()

    records = _jsonl(args.model_trace)
    acoustic_records = _jsonl(args.acoustic_trace) if args.acoustic_trace else []
    curve = ServiceCurve(
        b1_ns=_probe_median(args.probe_b1, 1),
        b2_ns=_probe_median(args.probe_b2, 2),
    )
    intervals = e2e_intervals_from_trace(acoustic_records) if acoustic_records else None
    fingerprint = _stable_fingerprint(records) if args.assume_work_preserved else None
    results = analyze_model_trace(
        records,
        service_curve=curve,
        e2e_intervals=intervals,
        baseline_work_fingerprint=fingerprint,
        candidate_work_fingerprint=fingerprint,
    )

    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "workload": args.workload,
            "model_trace": str(args.model_trace),
            "acoustic_trace": str(args.acoustic_trace) if args.acoustic_trace else "",
            "probe_b1_ns": curve.b1_ns,
            "probe_b2_ns": curve.b2_ns,
            "e2e_interval_count": len(intervals or {}),
        }
        row.update(result.as_dict())
        row["assumptions"] = ";".join(result.assumptions)
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["workload"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "schema": "apr-model-execution-plane-oracle-v1",
                    "workload": args.workload,
                    "model_trace": str(args.model_trace),
                    "acoustic_trace": str(args.acoustic_trace) if args.acoustic_trace else None,
                    "service_curve": {"b1_ns": curve.b1_ns, "b2_ns": curve.b2_ns},
                    "fixed_work_assumption": bool(args.assume_work_preserved),
                    "results": [result.as_dict() for result in results],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
