import pytest

from cosyvoice2.flow.decoder_dit import _resolve_cuda_graph_chunk_sizes
from cosyvoice2.flow.decoder_dit import _resolve_cuda_graph_max_cache


def test_cuda_graph_chunk_sizes_default_is_backward_compatible(monkeypatch):
    monkeypatch.delenv("LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS", raising=False)

    assert _resolve_cuda_graph_chunk_sizes() == (30, 48, 96)


def test_cuda_graph_chunk_sizes_are_deduplicated_and_sorted(monkeypatch):
    monkeypatch.setenv("LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS", "50, 26,50, 96")

    assert _resolve_cuda_graph_chunk_sizes() == (26, 50, 96)


def test_cuda_graph_max_cache_can_cover_the_frozen_flow_capacity(monkeypatch):
    monkeypatch.setenv("LYCHEEFD_FLOW_CUDA_GRAPH_MAX_CACHE", "2048")

    assert _resolve_cuda_graph_max_cache(50, 2048) == 2048


def test_cuda_graph_max_cache_preserves_bounded_defaults(monkeypatch):
    monkeypatch.delenv("LYCHEEFD_FLOW_CUDA_GRAPH_MAX_CACHE", raising=False)

    assert _resolve_cuda_graph_max_cache(30, 2048) == 500
    assert _resolve_cuda_graph_max_cache(50, 2048) == 1000


@pytest.mark.parametrize("value", ["0", "abc", "2049"])
def test_cuda_graph_max_cache_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("LYCHEEFD_FLOW_CUDA_GRAPH_MAX_CACHE", value)

    with pytest.raises(ValueError):
        _resolve_cuda_graph_max_cache(50, 2048)


@pytest.mark.parametrize("value", ["", "0", "-1", "abc", "26,abc", "257"])
def test_cuda_graph_chunk_sizes_reject_invalid_values(monkeypatch, value):
    if value == "":
        value = ","
    monkeypatch.setenv("LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS", value)

    with pytest.raises(ValueError):
        _resolve_cuda_graph_chunk_sizes()
