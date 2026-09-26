from pathlib import Path
import importlib.util
import json


MODULE_PATH = Path(__file__).parents[1] / "tools" / "apr" / "finalize_capacity_slo_reports.py"
SPEC = importlib.util.spec_from_file_location("finalize_capacity_slo_reports", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_zero_copy_summary_computes_reduction(tmp_path):
    root = tmp_path
    path = root / "zero_copy_mechanism_v3"
    path.mkdir()
    (path / "APR_ZERO_COPY_HANDOFF_METRICS.csv").write_text(
        "repeat,mode,handoff_latency_p95_ns\n"
        "1,copy,8000000\n2,copy,7000000\n"
        "1,zero_copy,100000\n2,zero_copy,120000\n",
        encoding="utf-8",
    )
    _, summary = MODULE._zero_copy_summary(root)
    assert summary["copy_p95_median_ms"] == 7.5
    assert summary["zero_copy_p95_median_ms"] == 0.11
    assert summary["reduction_fraction"] > 0.98


def test_calibration_report_never_promotes_unknown_memory(tmp_path):
    csv_path = tmp_path / "APR_CORRECTED_CAPACITY_CALIBRATION.csv"
    csv_path.write_text(
        "load_fraction,valid,attempt_status,gpu1_peak_memory_gib,audio_gap_p95_ms\n"
        "0.7,False,MEMORY_PEAK_UNKNOWN,,1200\n"
        "0.7,False,MEMORY_ENVELOPE_EXCEEDED,39.1,1200\n",
        encoding="utf-8",
    )
    rows, report = MODULE._calibration_summary(tmp_path)
    assert len(rows) == 2
    assert "no valid CAP0 calibration point" in report
    assert "unknown" in report.lower()
