import threading

import life_runner
from life_runner import SessionRunner


def _runner() -> SessionRunner:
    return SessionRunner(
        "http://127.0.0.1:1",
        "native_overlap",
        life_runner.Path("input.wav"),
        life_runner.Path("out"),
        1,
        threading.Barrier(1),
        set(),
        threading.Lock(),
        0,
        30.0,
    )


def test_cleanup_wait_does_not_hard_cut_at_45_seconds() -> None:
    """A slow server worker must get the configured run budget to finish."""
    assert SessionRunner.cleanup_wait_timeout_sec(900.0) == 900.0


def test_stop_request_is_idempotent(monkeypatch) -> None:
    runner = _runner()
    runner.session_id = "session-1"
    calls = []

    def fake_http_json(*args, **kwargs):
        calls.append((args, kwargs))
        return {"stopping": True}

    monkeypatch.setattr(life_runner, "http_json", fake_http_json)

    runner._request_stop()
    runner._request_stop()

    assert len(calls) == 1
    assert runner.stop_requested is True
