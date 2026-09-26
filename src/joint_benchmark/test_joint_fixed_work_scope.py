from __future__ import annotations

import json
from pathlib import Path

from tools.analysis.joint_fixed_work_analysis import build_joint_fixed_work_rows


def test_full_mixed_scope_is_carried_into_factorial_rows(tmp_path: Path):
    model = tmp_path / "model.csv"
    model.write_text(
        "mode,shape,median_elapsed_ms,peak_reserved_bytes\n"
        "serial,p10,100,10\nrow_batch,p10,60,20\n"
        "serial,p50,110,10\nrow_batch,p50,65,20\n"
        "serial,p90,120,10\nrow_batch,p90,70,20\n",
        encoding="utf-8",
    )
    acoustic = tmp_path / "acoustic.json"
    acoustic.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "shape": shape,
                        "warmup": False,
                        "measured_scope": "all_remaining_steps_with_mixed_prefix_and_singleton_tail",
                        "combined_batch_calls": 6,
                        "combined_singleton_tail_calls": 2,
                        "independent_two_b1_wall_ms": 100.0,
                        "mixed_chunk_padding_b2_wall_ms": 70.0,
                        "pcm_contract_pass": True,
                    }
                    for shape in ("p10", "p50", "p90")
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = build_joint_fixed_work_rows(model, acoustic)
    assert rows[0]["acoustic_measured_scope"] == (
        "all_remaining_steps_with_mixed_prefix_and_singleton_tail"
    )
    assert rows[0]["acoustic_combined_batch_calls"] == 6.0
