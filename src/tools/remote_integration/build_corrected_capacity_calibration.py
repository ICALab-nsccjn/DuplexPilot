#!/usr/bin/env python3
"""Build a paced, auditable capacity calibration trace.

The historical capacity calibration placed every arrival at zero.  This tool
keeps each source session's internal event offsets intact and applies one
pre-registered inter-arrival schedule across sessions.  It does not select
windows from observed candidate results and does not introduce barriers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "manifest":
        raise ValueError("input must start with a manifest")
    manifest = dict(records[0])
    events: dict[str, list[dict[str, Any]]] = {}
    for record in records[1:]:
        if record.get("record_type") != "event":
            raise ValueError("input contains a non-event record")
        event = dict(record)
        session_id = str(event.get("session_id") or "")
        if not session_id:
            raise ValueError("event is missing session_id")
        events.setdefault(session_id, []).append(event)
    if not events:
        raise ValueError("input contains no events")
    return manifest, events


def build(
    source: Path,
    output: Path,
    *,
    interarrival_s: float,
    session_count: int,
    load_fraction: float,
    selection: str,
) -> dict[str, Any]:
    if interarrival_s <= 0 or session_count <= 0 or load_fraction <= 0:
        raise ValueError("interarrival, session count, and load fraction must be positive")
    source_manifest, grouped = read_records(source)
    if session_count > len(grouped):
        raise ValueError("requested more sessions than source contains")

    # Hash-sort by source sample identity for a deterministic split that is
    # independent of filesystem order.  This also gives every calibration
    # trace a reproducible provenance list.
    def sort_key(session_id: str) -> tuple[str, str]:
        rows = grouped[session_id]
        sample = str(rows[0].get("source_sample_id") or session_id)
        return (hashlib.sha256(sample.encode("utf-8")).hexdigest(), session_id)

    selected = sorted(grouped, key=sort_key)[:session_count]
    output_events: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, source_session_id in enumerate(selected):
        rows = grouped[source_session_id]
        # The public converter can leave a terminal zero-length ``audio_chunk``
        # immediately before ``finish``.  It is a timeline marker, not data;
        # the online client correctly rejects it as an invalid upload.  Drop
        # only that invalid data event while preserving finish/barge-in/resume
        # markers and the source session timeline.
        rows = [
            row
            for row in rows
            if not (
                row.get("event_type") == "audio_chunk"
                and float(row.get("audio_duration_s", 0.0)) <= 0.0
            )
        ]
        if not rows:
            raise ValueError(f"source session {source_session_id} has no valid events")
        source_start = min(float(row["timestamp_s"]) for row in rows)
        new_arrival = round(index * interarrival_s, 6)
        first_sample = str(rows[0].get("source_sample_id") or "")
        provenance.append(
            {
                "source_session_id": source_session_id,
                "target_session_id": f"session-{index}",
                "source_sample_id": first_sample,
                "source_event_count": len(rows),
                "source_start_s": source_start,
                "target_arrival_s": new_arrival,
            }
        )
        for row in rows:
            event = dict(row)
            event["session_id"] = f"session-{index}"
            event["timestamp_s"] = round(
                new_arrival + float(row["timestamp_s"]) - source_start, 6
            )
            output_events.append(event)

    priority = {"arrival": 0, "audio_chunk": 1, "barge_in": 2, "resume": 3, "cancel": 4, "finish": 5}
    output_events.sort(
        key=lambda row: (
            float(row["timestamp_s"]),
            priority.get(str(row.get("event_type")), 99),
            str(row.get("session_id")),
        )
    )
    arrivals = [
        float(row["timestamp_s"])
        for row in output_events
        if row.get("event_type") == "arrival"
    ]
    if len(arrivals) != session_count or any(
        b <= a for a, b in zip(arrivals, arrivals[1:])
    ):
        raise ValueError("paced arrivals must be strictly increasing")

    digest_payload = "\n".join(
        f"{row['target_session_id']}={row['source_sample_id']}"
        for row in provenance
    ).encode("utf-8")
    selection_digest = hashlib.sha256(digest_payload).hexdigest()
    distribution = Counter(
        str(row.get("scenario"))
        for row in output_events
        if row.get("event_type") == "arrival"
    )
    manifest = {
        "record_type": "manifest",
        "schema_version": "public-trace-v1",
        "trace_id": f"HD-CapacityCalibration-N{session_count}-U{int(round(load_fraction * 100)):02d}",
        "dataset_revision": source_manifest.get("dataset_revision", ""),
        "source_hash": source_manifest.get("source_hash", ""),
        "converter_commit": source_manifest.get("converter_commit", ""),
        "arrival_window": f"paced-CAP0-U{load_fraction:.2f}-ia{interarrival_s:.6f}s",
        "arrival_scale": 1.0,
        "scenario_distribution": dict(sorted(distribution.items())),
        "session_count": session_count,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True) + "\n")
        for row in output_events:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    provenance_path.write_text(
        json.dumps(
            {
                "source_trace": str(source),
                "source_sha256": sha256(source),
                "selection": selection,
                "selection_digest": selection_digest,
                "schedule": {
                    "interarrival_s": interarrival_s,
                    "load_fraction": load_fraction,
                    "formula": "target_arrival_i = i * interarrival_s; internal offsets unchanged",
                },
                "sessions": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interarrival-s", type=float, required=True)
    parser.add_argument("--session-count", type=int, default=30)
    parser.add_argument("--load-fraction", type=float, required=True)
    parser.add_argument("--selection", default="sha256(source_sample_id),session_id")
    args = parser.parse_args()
    print(json.dumps(build(
        args.source,
        args.output,
        interarrival_s=args.interarrival_s,
        session_count=args.session_count,
        load_fraction=args.load_fraction,
        selection=args.selection,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
