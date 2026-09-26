import pytest

from tools.benchmarks.run_graph_b2_fixed_work import (
    compute_interaction_effect,
    derive_graph_chunk_sizes,
    evaluate_fixed_work_gate,
    summarize_comparisons,
)


def test_interaction_effect_is_difference_of_log_effects():
    assert compute_interaction_effect(1.25, 1.75) == pytest.approx(0.5)


def test_fixed_work_gate_requires_b2_graph_hits_and_speedup():
    result = evaluate_fixed_work_gate(
        {
            "correctness": True,
            "b2_graph_steps": 10,
            "b2_graph_eligible": 10,
            "b2_graph_throughput_ratio": 1.21,
            "b2_graph_ci_lower": 1.05,
            "peak_memory_bytes": 30 * 1024**3,
        }
    )
    assert result["status"] == "PASS"
    blocked = evaluate_fixed_work_gate(
        {
            "correctness": True,
            "b2_graph_steps": 9,
            "b2_graph_eligible": 10,
            "b2_graph_throughput_ratio": 1.21,
            "b2_graph_ci_lower": 1.05,
            "peak_memory_bytes": 30 * 1024**3,
        }
    )
    assert blocked["status"] == "FAIL"


def test_comparison_summary_reports_b1_b2_and_interaction_separately():
    summary = summarize_comparisons(
        [
            {"batch": 1, "graph": False, "throughput": 10.0},
            {"batch": 1, "graph": True, "throughput": 12.0},
            {"batch": 2, "graph": False, "throughput": 15.0},
            {"batch": 2, "graph": True, "throughput": 21.0},
        ]
    )
    assert summary["b1_graph_effect_ratio"] == pytest.approx(1.2)
    assert summary["b2_graph_effect_ratio"] == pytest.approx(1.4)
    assert summary["interaction_ratio_difference"] == pytest.approx(0.2)


def test_graph_chunk_sizes_are_derived_from_encoder_output_geometry():
    assert derive_graph_chunk_sizes((30, 48, 96), 3, 2) == (54, 90, 186)


def test_graph_chunk_size_derivation_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        derive_graph_chunk_sizes((3, 4, 5), 3, 2)
