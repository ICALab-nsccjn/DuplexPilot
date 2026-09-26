from __future__ import annotations

from tools.benchmarks.flow_b2_combination_oracle import (
    evaluate_combination_events,
    write_combination_outputs,
)


def _metadata(request_id: str, *, step: int, shape: list[int], ready: int) -> dict:
    return {
        "request_id": request_id,
        "generation_id": 1,
        "version": 0,
        "ready_at_ns": ready,
        "model_identity": "flow",
        "device": "cuda:1",
        "dtype": "torch.float32",
        "step_index": step,
        "current_chunk_shape": shape,
        "attention_cache_length": 8,
        "cnn_cache_signature": [1, 2],
        "last_chunk": False,
        "n_timesteps": 10,
    }


def _events() -> list[dict]:
    return [
        {
            "event": "FLOW_COMPATIBILITY_WATERFALL",
            "timestamp_monotonic_ns": 100,
            "candidate_metadata": [
                _metadata("a", step=2, shape=[1, 80, 20], ready=100),
            _metadata("b", step=4, shape=[1, 80, 24], ready=100),
            ],
        },
        {
            "event": "FLOW_STEP_START",
            "timestamp_monotonic_ns": 100,
            "request_ids": ["a"],
            "generation_ids": [1],
            "step_indices": [2],
        },
        {
            "event": "FLOW_STEP_END",
            "timestamp_monotonic_ns": 10_000_100,
            "request_ids": ["a"],
            "generation_ids": [1],
            "step_indices": [2],
            "wall_time_ms": 10.0,
        },
        {
            "event": "FLOW_STEP_START",
            "timestamp_monotonic_ns": 200,
            "request_ids": ["b"],
            "generation_ids": [1],
            "step_indices": [4],
        },
        {
            "event": "FLOW_STEP_END",
            "timestamp_monotonic_ns": 10_000_200,
            "request_ids": ["b"],
            "generation_ids": [1],
            "step_indices": [4],
            "wall_time_ms": 10.0,
        },
        {
            "event": "REQUEST_REGISTER",
            "timestamp_monotonic_ns": 100,
            "request_ids": ["a", "b"],
        },
        {
            "event": "REQUEST_FINISH",
            "timestamp_monotonic_ns": 20_000_100,
            "request_ids": ["a", "b"],
        },
    ]


def test_combination_evaluator_exposes_factorial_and_interaction_fields():
    row = evaluate_combination_events(
        _events(),
        step_mode="mixed",
        shape_mode="chunk_padding",
        attention_cache_mode="exact",
        batch_speedup=1.8,
    )

    assert row["policy"].startswith("step_mixed__shape_chunk_padding")
    assert row["b2_batch_count"] == 1
    assert row["b2_work_fraction"] == 1.0
    assert row["fixed_ready_time_digest"]
    assert row["observation_source"] == "fixed_online_timeline"


def test_combination_report_records_broad_oracle_recommendation(tmp_path):
    rows = [
        {
            "policy": "step_mixed__shape_chunk_padding__attention_exact",
            "split": "discovery",
            "wait_ms": 0.0,
            "flow_speedup": 1.2,
            "predicted_e2e_speedup": 1.1,
            "b2_work_fraction": 0.4,
            "b2_batch_count": 2,
            "step_mode": "mixed",
            "shape_mode": "chunk_padding",
            "padding_measurement_available": False,
        }
    ]

    write_combination_outputs(rows, tmp_path)

    report = (tmp_path / "FLOW_B2_COMBINATION_ORACLE_REPORT.md").read_text()
    assert "PROCEED_EXPLORATORY" in report
    assert (tmp_path / "FLOW_B2_COMBINATION_ORACLE.csv").exists()
