from __future__ import annotations

from pathlib import Path

from tools.apr.run_acoustic_microbatch_diagnostic import (
    _equivalence_session_count,
    _ensure_repo_root_importable,
    _write_invalid_output,
    run,
)
from profiling.token2wav_flow_profile.layout import real_flow_cache_layout


def test_real_diagnostic_layout_is_explicit_and_complete() -> None:
    assert real_flow_cache_layout().as_tuple() == (
        ("conformer_cnn_cache", 0),
        ("conformer_att_cache", 1),
        ("estimator_cnn_cache", 2),
        ("estimator_att_cache", 2),
    )


def test_diagnostic_runner_does_not_require_production_apr_selector() -> None:
    assert callable(run)


def test_script_entrypoint_adds_repository_root_to_import_path(monkeypatch) -> None:
    import sys

    repo_root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repo_root])

    _ensure_repo_root_importable()

    assert repo_root in sys.path


def test_equivalence_control_does_not_exceed_requested_sessions() -> None:
    assert _equivalence_session_count(1) == 1
    assert _equivalence_session_count(2) == 2


def test_invalid_diagnostic_attempt_is_written_fail_closed(tmp_path) -> None:
    import argparse
    import json

    output = tmp_path / "invalid.json"
    args = argparse.Namespace(
        output=str(output),
        logical_sessions=2,
        repeats=1,
        model_path="model",
        prompt_wav="prompt.wav",
    )

    payload = _write_invalid_output(args, RuntimeError("model batch unsupported"))

    assert payload["status"] == "INVALID"
    assert payload["diagnostic_only"] is True
    assert "model batch unsupported" in payload["failure_signature"]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "INVALID"
