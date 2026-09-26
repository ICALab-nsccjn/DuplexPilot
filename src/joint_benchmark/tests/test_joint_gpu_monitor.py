from pathlib import Path

from tools.analysis.analyze_joint_b22 import summarize_gpu_monitor


def test_summarize_runtime_gpu_monitor_without_header(tmp_path: Path):
    path = tmp_path / "gpu_monitor.csv"
    path.write_text(
        "2026/09/01 00:00:00.000, 0, 1200, 40960, 10\n"
        "2026/09/01 00:00:00.000, 1, 24576, 40960, 80\n"
        "2026/09/01 00:00:00.200, 0, 1800, 40960, 20\n"
        "2026/09/01 00:00:00.200, 1, 26000, 40960, 90\n",
        encoding="utf-8",
    )
    summary = summarize_gpu_monitor(path)
    assert summary["gpu_monitor_status"] == "OBSERVED"
    assert summary["gpu_monitor_samples"] == 4
    assert summary["gpu0_peak_memory_mib"] == 1800.0
    assert summary["gpu1_peak_memory_mib"] == 26000.0
    assert summary["gpu1_peak_memory_bytes"] == 26000 * 1024 * 1024


def test_summarize_gpu_monitor_missing_is_explicit(tmp_path: Path):
    summary = summarize_gpu_monitor(tmp_path / "missing.csv")
    assert summary["gpu_monitor_status"] == "MISSING"
    assert summary["gpu1_peak_memory_mib"] is None
    assert summary["gpu1_peak_memory_bytes"] is None
