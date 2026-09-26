import torch
import pytest

import lychee_fd.app as runtime_app


def test_local_token2wav_load_enables_opt_in_flow_cuda_graph(monkeypatch):
    calls = []

    class FakeFlow:
        def scatter_cuda_graph(self, enabled):
            calls.append(bool(enabled))

    class FakeToken2wav:
        def __init__(self, path):
            self.path = path
            self.flow = FakeFlow()

    monkeypatch.setenv("LYCHEEFD_ENABLE_FLOW_CUDA_GRAPH", "1")
    monkeypatch.setattr(runtime_app, "token2wav_model", None)
    monkeypatch.setattr(runtime_app, "_import_token2wav_class", lambda: FakeToken2wav)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    runtime_app.ensure_local_token2wav_loaded("/tmp/token2wav")

    assert calls == [True]


def test_local_token2wav_load_does_not_enable_graph_by_default(monkeypatch):
    calls = []

    class FakeFlow:
        def scatter_cuda_graph(self, enabled):
            calls.append(bool(enabled))

    class FakeToken2wav:
        def __init__(self, path):
            self.path = path
            self.flow = FakeFlow()

    monkeypatch.delenv("LYCHEEFD_ENABLE_FLOW_CUDA_GRAPH", raising=False)
    monkeypatch.setattr(runtime_app, "token2wav_model", None)
    monkeypatch.setattr(runtime_app, "_import_token2wav_class", lambda: FakeToken2wav)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    runtime_app.ensure_local_token2wav_loaded("/tmp/token2wav")

    assert calls == []


def test_local_token2wav_load_fails_closed_without_public_graph_api(monkeypatch):
    class FakeToken2wav:
        def __init__(self, path):
            self.path = path
            self.flow = object()

    monkeypatch.setenv("LYCHEEFD_ENABLE_FLOW_CUDA_GRAPH", "1")
    monkeypatch.setattr(runtime_app, "token2wav_model", None)
    monkeypatch.setattr(runtime_app, "_import_token2wav_class", lambda: FakeToken2wav)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    with pytest.raises(RuntimeError, match="scatter_cuda_graph"):
        runtime_app.ensure_local_token2wav_loaded("/tmp/token2wav")

    assert runtime_app.token2wav_model is None
