from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.analysis.joint_fixed_work_analysis import (
    build_joint_fixed_work_rows,
    summarize_factorial_effects,
    write_report,
)


def _write_model(path: Path) -> None:
    path.write_text(
        "mode,shape,median_elapsed_ms,peak_reserved_bytes,token_digest_equal_serial_row_batch\n"
        "serial,p10,100,10,True\n"
        "serial,p50,110,10,True\n"
        "serial,p90,120,10,True\n"
        "row_batch,p10,60,20,True\n"
        "row_batch,p50,65,20,True\n"
        "row_batch,p90,70,20,True\n",
        encoding="utf-8",
    )


def _write_acoustic(path: Path) -> None:
    payload = {
        "configuration": {"dtype": "float32"},
        "records": [
            {
                "scenario": "mixed_chunk_padding",
                "candidate": {"wall_ms": 40.0, "peak_memory_bytes": 200},
                "independent": {"wall_ms": 70.0},
                "comparison": {"all_pcm_contract_pass": True},
                "config": {"current_lengths": [12, 16]},
            },
            {
                "scenario": "mixed_chunk_padding",
                "candidate": {"wall_ms": 42.0, "peak_memory_bytes": 210},
                "independent": {"wall_ms": 72.0},
                "comparison": {"all_pcm_contract_pass": True},
                "config": {"current_lengths": [12, 16]},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_has_all_four_joint_cells_without_multiplying_speedups(tmp_path: Path):
    model = tmp_path / "model.csv"
    acoustic = tmp_path / "acoustic.json"
    _write_model(model)
    _write_acoustic(acoustic)

    rows = build_joint_fixed_work_rows(model, acoustic)

    assert {(row["shape"], row["joint_id"]) for row in rows} == {
        (shape, joint) for shape in ("p10", "p50", "p90")
        for joint in ("J11", "J21", "J12", "J22")
    }
    assert all(row["causal_scope"] == "factorial_component_join" for row in rows)
    assert rows[0]["joint_wall_time_ms"] is None


def test_factorial_effects_use_additive_component_effects(tmp_path: Path):
    model = tmp_path / "model.csv"
    acoustic = tmp_path / "acoustic.json"
    _write_model(model)
    _write_acoustic(acoustic)
    rows = build_joint_fixed_work_rows(model, acoustic)

    summary = summarize_factorial_effects(rows)
    p50 = summary["p50"]
    assert p50["model_effect_b1_ratio"] == 110 / 65
    assert p50["model_effect_b2_ratio"] == 110 / 65
    assert p50["acoustic_effect_b1_ratio"] == 71 / 41
    assert p50["acoustic_effect_b2_ratio"] == 71 / 41
    assert p50["joint_wall_time_claim"] == "not_measured"


def test_actual_shape_keyed_acoustic_runner_is_joined_by_envelope(tmp_path: Path):
    model = tmp_path / "model.csv"
    acoustic = tmp_path / "acoustic_actual.json"
    _write_model(model)
    records = []
    for shape, base in (("p10", 10.0), ("p50", 20.0), ("p90", 30.0)):
        for repeat in range(2):
            records.append(
                {
                    "shape": shape,
                    "warmup": False,
                    "independent_two_b1_wall_ms": base + repeat,
                    "mixed_chunk_padding_b2_wall_ms": base / 2 + repeat,
                    "pcm_contract_pass": True,
                    "gpu_memory": [{"device": 1, "max_reserved_bytes": 123}],
                }
            )
    acoustic.write_text(json.dumps({"records": records}), encoding="utf-8")

    rows = build_joint_fixed_work_rows(model, acoustic)
    by_key = {(row["shape"], row["joint_id"]): row for row in rows}
    assert by_key[("p10", "J12")]["acoustic_component_median_ms"] == 5.5
    assert by_key[("p50", "J22")]["acoustic_component_median_ms"] == 10.5
    assert by_key[("p90", "J12")]["acoustic_component_median_ms"] == 15.5
    assert all(row["acoustic_source_shape"] == "shape-keyed" for row in rows)


def test_fixed_work_report_emits_bounded_trace(tmp_path: Path):
    model = tmp_path / "model.csv"
    acoustic = tmp_path / "acoustic.json"
    out = tmp_path / "out"
    _write_model(model)
    _write_acoustic(acoustic)
    rows = build_joint_fixed_work_rows(model, acoustic)
    write_report(rows, out)
    trace = out / "APR_JOINT_B22_FIXED_WORK_TRACE.jsonl"
    assert trace.exists()
    line = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert line["event_type"] == "FIXED_WORK_ACOUSTIC_COMPONENT"
    assert "pcm_b64" not in line
