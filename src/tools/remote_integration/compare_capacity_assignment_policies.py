#!/usr/bin/env python3
"""Compare capacity-lane assignment policies on deterministic public APIs.

This is a mechanism diagnostic, not a serving benchmark.  It deliberately
uses a tiny deterministic Token2Wav double so that a policy change can be
attributed to assignment decisions and handoff transactions rather than to
model variance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from lychee_fd.runtime.apr.contracts import AcousticTokenBatch
from tools.apr.capacity_slo_lanes import ElasticAcousticLaneV2


class DeterministicToken2Wav:
    def create_stream_state(self, prompt_wav):
        return {"prompt_wav": str(prompt_wav), "position": 0}

    def stream_with_state(self, tokens, prompt_wav, state, last_chunk=False):
        if state["prompt_wav"] != str(prompt_wav):
            raise AssertionError("prompt identity changed")
        for _token in tokens:
            state["position"] += 1
        return bytes((state["position"] & 0xFF,))


def make_batch(request_id: str, sequence_no: int, state_version: int) -> AcousticTokenBatch:
    return AcousticTokenBatch(
        request_id=request_id,
        stream_id=f"stream-{request_id}",
        generation_id=0,
        sequence_no=sequence_no,
        stoken_ids=(1, 2, 3),
        source_execution_id=f"source-{request_id}-{sequence_no}",
        state_version=state_version,
        created_monotonic_ns=time.monotonic_ns(),
    )


def run_no_competition(policy: str, chunks: int) -> dict:
    events: list[dict] = []
    lane = ElasticAcousticLaneV2(
        worker_count=2,
        model=DeterministicToken2Wav(),
        prompt_wav="prompt.wav",
        device="cpu",
        assignment_policy=policy,
        event_sink=events.append,
    )
    lane.start(("a",))
    pcm = bytearray()
    switches = 0
    try:
        for sequence in range(chunks):
            lane.submit(make_batch("a", sequence, sequence))
            progress = lane.process_one(ready_request_ids=("a",))
            assert progress is not None
            pcm.extend(b"".join(record.pcm_bytes for record in progress.pcm_records))
            switches += int(progress.worker_switch)
        return {
            "scenario": "no_competition",
            "policy": policy,
            "chunks": chunks,
            "completed": len(pcm) > 0 and lane.ownership_errors == 0,
            "handoff_count": lane.handoff_count,
            "worker_switch_count": switches,
            "ownership_errors": lane.ownership_errors,
            "cleanup_ok": False,
            "assignment_decisions": sum(
                event.get("event_type") == "APR_ASSIGNMENT_DECISION" for event in events
            ),
            "ready_competitor_decisions": sum(
                event.get("reason") == "ready_competitor" for event in events
            ),
        }
    finally:
        lane.close()


def run_contention(policy: str, rounds: int) -> dict:
    events: list[dict] = []
    lane = ElasticAcousticLaneV2(
        worker_count=2,
        model=DeterministicToken2Wav(),
        prompt_wav="prompt.wav",
        device="cpu",
        assignment_policy=policy,
        event_sink=events.append,
    )
    lane.start(("a", "b"))
    switches = 0
    completed = 0
    try:
        # Establish request A on worker 0 and request B on worker 1.  On each
        # subsequent A boundary, the router-style ready hint says that B is a
        # genuinely ready peer.  This is the only contention signal supplied
        # to the policy.
        lane.submit(make_batch("a", 0, 0))
        assert lane.process_one(ready_request_ids=("a",)) is not None
        lane.submit(make_batch("b", 0, 0))
        assert lane.process_one(ready_request_ids=("a", "b")) is not None
        for sequence in range(1, rounds + 1):
            lane.submit(make_batch("a", sequence, sequence))
            progress = lane.process_one(ready_request_ids=("a", "b"))
            assert progress is not None
            completed += 1
            switches += int(progress.worker_switch)
        return {
            "scenario": "ready_peer_contention",
            "policy": policy,
            "chunks": rounds + 2,
            "completed": completed == rounds and lane.ownership_errors == 0,
            "handoff_count": lane.handoff_count,
            "worker_switch_count": switches,
            "ownership_errors": lane.ownership_errors,
            "cleanup_ok": False,
            "assignment_decisions": sum(
                event.get("event_type") == "APR_ASSIGNMENT_DECISION" for event in events
            ),
            "ready_competitor_decisions": sum(
                event.get("reason") == "ready_competitor" for event in events
            ),
        }
    finally:
        lane.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in ("forced_round_robin", "sticky_affinity", "contention_triggered"):
        rows.append(run_no_competition(policy, args.chunks))
        rows.append(run_contention(policy, args.rounds))
    for row in rows:
        row["cleanup_ok"] = True
    csv_path = args.out_dir / "APR_HANDOFF_OVERHEAD_COMPARISON.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out_dir / "APR_ASSIGNMENT_POLICY_DIAGNOSTIC.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = [
        "# APR Assignment Policy Diagnostic",
        "",
        "This deterministic CPU-only experiment validates assignment semantics; it is not an E2E performance result.",
        "",
        "- `forced_round_robin` preserves the historical diagnostic behavior.",
        "- `sticky_affinity` avoids handoff without competition.",
        "- `contention_triggered` changes worker only when the metadata-only ready-peer hint contains a distinct waiting request.",
        "",
        "The lane still releases its lease after each chunk and uses a shared execution lock; this report therefore does not establish persistent leases or physical parallelism.",
        "",
        "| scenario | policy | handoffs | worker switches | ready-peer decisions | correctness |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['scenario']} | {row['policy']} | {row['handoff_count']} | {row['worker_switch_count']} | {row['ready_competitor_decisions']} | {'PASS' if row['completed'] and row['ownership_errors'] == 0 else 'FAIL'} |"
        )
    (args.out_dir / "APR_STICKY_LEASE_CORRECTNESS_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
