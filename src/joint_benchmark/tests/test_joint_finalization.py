from pathlib import Path


from tools.analysis.finalize_joint_b22 import (
    build_pair_rows,
    classify_joint_result,
    expected_case_count,
    summarize_bootstrap_ratios,
)
from tools.analysis.finalize_joint_b22 import _fixed_work_status
from tools.analysis.finalize_joint_b22 import _session_fields


def _row(mode: str, workload: str = "HD-Burst", n: int = 8, repeat: int = 1, value: float = 1.0):
    return {
        "case_id": f"counted_{workload}_N{n}_{mode}_r{repeat}",
        "joint_mode": mode,
        "workload": workload,
        "N": n,
        "repeat": repeat,
        "valid": True,
        "completion_rate": 1.0,
        "ownership_errors": 0,
        "runtime_errors": 0,
        "useful_audio_throughput": value,
        "session_span_s": 10.0 / value,
        "ttfa_p95_ms": 100.0,
        "audio_gap_p95_ms": 20.0,
        "work_fingerprint_digest": "same-work",
        "acoustic_b2_work_fraction": 0.4 if mode in {"J12", "J22"} else 0.0,
        "model_decode_b2_row_fraction": 0.9 if mode in {"J21", "J22"} else 0.0,
    }


def test_pair_rows_keep_four_factor_comparisons_and_work_status():
    rows = [_row(mode, value=(1.2 if mode == "J22" else 1.0)) for mode in ("J11", "J21", "J12", "J22")]
    pairs = build_pair_rows(rows)
    comparisons = {row["comparison"] for row in pairs}
    assert {"J21/J11", "J12/J11", "J22/J11", "J22/J21", "J22/J12"}.issubset(comparisons)
    assert all(row["work_comparability"] == "MATCHED" for row in pairs)


def test_pair_rows_keep_counted_and_heldout_scopes_separate():
    counted = [_row(mode, repeat=1, value=(1.0 if mode != "J22" else 1.2)) for mode in ("J11", "J21", "J12", "J22")]
    heldout = [_row(mode, repeat=1, value=(1.0 if mode != "J22" else 0.8)) for mode in ("J11", "J21", "J12", "J22")]
    heldout = [dict(row, case_id=row["case_id"].replace("counted_", "heldout_")) for row in heldout]

    pairs = build_pair_rows(counted + heldout)
    acoustic = [row for row in pairs if row["comparison"] == "J22/J21"]

    assert {row["scope"] for row in acoustic} == {"counted", "heldout"}
    assert {row["right_case"].split("_")[0] for row in acoustic} == {"counted", "heldout"}


def test_bootstrap_is_deterministic_and_reports_wins():
    summary = summarize_bootstrap_ratios([1.2, 1.1, 0.9], seed=20260901, samples=1000)
    assert summary["n"] == 3
    assert summary["wins"] == 2
    assert summary["median"] == 1.1
    assert summary["ci_low"] <= summary["median"] <= summary["ci_high"]
    assert summary == summarize_bootstrap_ratios([1.2, 1.1, 0.9], seed=20260901, samples=1000)


def test_classification_distinguishes_workload_dependent_positive_signal():
    positive = [
        _row("J21", workload="HD-Burst", value=1.0),
        _row("J22", workload="HD-Burst", value=1.2),
    ]
    neutral = [
        _row("J21", workload="HD-LongTail", value=1.0),
        _row("J22", workload="HD-LongTail", value=1.0),
    ]
    result = classify_joint_result(
        rows=positive + neutral,
        pair_rows=build_pair_rows(positive + neutral),
        expected=expected_case_count(),
        fixed_work_status="PASS",
    )
    assert result["classification"] == "JOINT_B22_WORKLOAD_DEPENDENT"


def test_expected_case_count_is_explicit_and_does_not_depend_on_results(tmp_path: Path):
    assert expected_case_count() == 136


def test_fixed_work_status_accepts_legacy_comparison_contract(tmp_path: Path):
    path = tmp_path / "fixed.json"
    path.write_text(
        '{"records": [{"comparison": {"all_pcm_contract_pass": true}}]}',
        encoding="utf-8",
    )
    status, meta = _fixed_work_status(path)
    assert status == "PASS"
    assert meta["pcm_contract_pass_count"] == 1


def test_session_fields_do_not_double_count_new_client_aggregates():
    session = {
        "pcm_sample_count": 10,
        "pcm_chunk_count": 2,
        "events": [
            {"type": "audio_chunk_pcm", "num_samples": 6, "server_sse_send_epoch_ms": 100},
            {"type": "audio_chunk_pcm", "num_samples": 4, "server_sse_send_epoch_ms": 120},
        ],
    }
    fields = _session_fields(session)
    assert fields["pcm_sample_count"] == 10
    assert fields["pcm_chunk_count"] == 2
