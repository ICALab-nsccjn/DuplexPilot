from __future__ import annotations

from tools.benchmarks.run_joint_acoustic_fixed_work import SHAPES


def test_registered_envelopes_have_unequal_current_lengths_and_bounded_cache():
    assert set(SHAPES) == {"p10", "p50", "p90"}
    for spec in SHAPES.values():
        assert len(spec["prior_lengths"]) == 2
        assert len(spec["current_lengths"]) == 2
        assert spec["current_lengths"][0] != spec["current_lengths"][1]
        assert max(spec["prior_lengths"] + spec["current_lengths"]) < 1000
