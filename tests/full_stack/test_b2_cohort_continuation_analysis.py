from __future__ import annotations

import json

from tools.benchmarks.analyze_b2_cohort_continuation import (
    analyze_online_trace,
)


def _submit(step: int, *, batch_size: int = 2, shape: int = 10, last: bool = False):
    requests = ["a", "b"] if batch_size == 2 else ["a"]
    return {
        "event": "FLOW_BATCH_SUBMIT",
        "batch_size": batch_size,
        "request_ids": requests,
        "generation_ids": [1] * len(requests),
        "sequence_no": [0] * len(requests),
        "step_index": step,
        "shape_signature": [shape],
        "last_chunk": [last] * len(requests),
        "timestamp_monotonic_ns": step + shape * 100,
    }


def _write(path, events):
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def test_same_exact_pair_is_counted_as_multi_step_continuation(tmp_path):
    path = tmp_path / "online.jsonl"
    _write(path, [_submit(0), _submit(1), _submit(2)])

    summary, edges = analyze_online_trace(path)

    assert summary["b2_batches"] == 3
    assert summary["continuation_edges"] == 2
    assert summary["next_edge_observations"] == 2
    assert summary["continuation_rate"] == 1.0
    assert summary["max_streak_steps"] == 3
    assert summary["full10_streaks"] == 0
    assert [edge["category"] for edge in edges] == [
        "CONTINUATION_COMPATIBLE",
        "CONTINUATION_COMPATIBLE",
        "BREAK_NO_NEXT_SUBMIT",
    ]


def test_shape_change_is_not_treated_as_a_scheduler_continuation(tmp_path):
    path = tmp_path / "online.jsonl"
    _write(path, [_submit(0, shape=10), _submit(1, shape=11)])

    summary, edges = analyze_online_trace(path)

    assert summary["continuation_edges"] == 0
    assert summary["transition_counts"]["BREAK_SHAPE"] == 1
    assert edges[0]["category"] == "BREAK_SHAPE"


def test_pair_reformation_requires_both_members_and_batch_two(tmp_path):
    path = tmp_path / "online.jsonl"
    a_next = _submit(1, batch_size=1)
    b_next = dict(a_next)
    b_next["request_ids"] = ["b"]
    b_next["timestamp_monotonic_ns"] += 1
    _write(path, [_submit(0), a_next, b_next])

    summary, edges = analyze_online_trace(path)

    assert summary["continuation_edges"] == 0
    assert edges[0]["category"] == "BREAK_PAIR_NOT_REFORMED"
