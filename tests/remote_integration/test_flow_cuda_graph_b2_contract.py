import types

import pytest
import torch
import torch.nn as nn

from cosyvoice2.flow.decoder_dit import (
    DiT,
    _cfg_row_order,
    _new_cuda_graph_stats,
    _resolve_cuda_graph_logical_batches,
    _cuda_graph_static_shapes,
)
from cosyvoice2.flow.flow import CausalMaskedDiffWithXvec


def test_logical_batch_config_defaults_to_b1_and_accepts_explicit_b2(monkeypatch):
    monkeypatch.delenv("LYCHEEFD_FLOW_CUDA_GRAPH_LOGICAL_BATCHES", raising=False)
    assert _resolve_cuda_graph_logical_batches() == (1,)
    monkeypatch.setenv("LYCHEEFD_FLOW_CUDA_GRAPH_LOGICAL_BATCHES", "1,2,2")
    assert _resolve_cuda_graph_logical_batches() == (1, 2)


@pytest.mark.parametrize("logical_batch,physical_batch", [(1, 2), (2, 4)])
def test_graph_cfg_rows_and_static_buffers_are_explicit(logical_batch, physical_batch):
    assert _cfg_row_order(logical_batch) == tuple(
        [("conditional", row) for row in range(logical_batch)]
        + [("unconditional", row) for row in range(logical_batch)]
    )
    shapes = _cuda_graph_static_shapes(logical_batch, chunk_size=30, max_cache=1000)
    assert shapes["physical_cfg_batch"] == physical_batch
    assert shapes["x"] == (physical_batch, 320, 30)
    assert shapes["t"] == (physical_batch, 1, 512)
    assert shapes["cnn_cache"] == (16, physical_batch, 1024, 2)
    assert shapes["att_cache"] == (16, physical_batch, 8, 1000, 128)


def test_public_flow_graph_api_forwards_logical_batch_sizes():
    flow = CausalMaskedDiffWithXvec.__new__(CausalMaskedDiffWithXvec)
    calls = []

    class FakeDecoder:
        def scatter_cuda_graph(self, enabled, *, logical_batch_sizes=(1,)):
            calls.append((bool(enabled), tuple(logical_batch_sizes)))

    flow.decoder = FakeDecoder()
    flow.scatter_cuda_graph(True, logical_batch_sizes=(1, 2))
    assert calls == [(True, (1, 2))]


def test_graph_contract_rejects_unsupported_logical_batch(monkeypatch):
    monkeypatch.setenv("LYCHEEFD_FLOW_CUDA_GRAPH_LOGICAL_BATCHES", "1,3")
    with pytest.raises(ValueError, match="logical batch"):
        _resolve_cuda_graph_logical_batches()


def test_b2_graph_can_replay_the_initial_step_without_logical_cache():
    """An empty initial cache must use graph-local zero buffers, not fallback."""

    class FakeTimestepEmbedder(nn.Module):
        def forward(self, values):
            return torch.zeros(
                (values.shape[0], 512), dtype=values.dtype, device=values.device
            )

    model = DiT.__new__(DiT)
    nn.Module.__init__(model)
    model.t_embedder = FakeTimestepEmbedder()
    model.blocks = nn.ModuleList([])
    model.cnn_cache_buffer = torch.zeros((16, 2, 1024, 2))
    model.att_cache_buffer = torch.zeros((16, 2, 8, 1000, 128))
    model.use_cuda_graph = True
    model.cuda_graph_logical_batch_sizes = (2,)
    model.graph_chunk_by_logical_batch = {(2, 3): object()}
    model.max_size_chunk_by_logical_batch = {(2, 3): 1000}
    model._cuda_graph_stats = _new_cuda_graph_stats()
    replayed = []

    def fake_replay(self, **kwargs):
        replayed.append(kwargs["logical_batch_size"])
        x = kwargs["x"]
        batch = int(x.shape[0])
        chunk = int(x.shape[2])
        return (
            x,
            torch.zeros((16, batch, 1024, 2)),
            torch.zeros((16, batch, 8, chunk, 128)),
        )

    def unexpected_dynamic_path(*args, **kwargs):
        raise AssertionError("B=2 initial step unexpectedly used dynamic path")

    model._replay_cuda_graph = types.MethodType(fake_replay, model)
    model.blocks_forward_chunk = unexpected_dynamic_path
    # ``DiT.forward_chunk`` receives the already-expanded physical CFG rows.
    x = torch.zeros((4, 80, 3))
    result = model.forward_chunk(
        x=x,
        mu=torch.zeros_like(x),
        t=torch.zeros((4,)),
        spks=torch.zeros((4, 80)),
        cond=torch.zeros_like(x),
        cnn_cache=None,
        att_cache=None,
    )

    assert replayed == [2]
    assert result[0].shape == (4, 320, 3)
    assert model.cuda_graph_stats()["graph_replay_calls_by_logical_batch"]["2"] == 1
