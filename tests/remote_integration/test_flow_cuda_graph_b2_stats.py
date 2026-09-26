from cosyvoice2.flow.decoder_dit import DiT, _new_cuda_graph_stats


def test_stats_have_per_logical_batch_counters():
    stats = _new_cuda_graph_stats()
    for name in (
        "forward_chunk_calls_by_logical_batch",
        "graph_replay_calls_by_logical_batch",
        "fallback_calls_by_logical_batch",
    ):
        assert stats[name] == {"1": 0, "2": 0}
    assert stats["fallback_reasons_by_logical_batch"] == {"1": {}, "2": {}}
    assert stats["eligible_calls"] == 0


def test_stats_snapshot_and_reset_keep_per_batch_fields():
    model = DiT.__new__(DiT)
    model._cuda_graph_stats = _new_cuda_graph_stats()
    model._cuda_graph_stats["graph_replay_calls_by_logical_batch"]["2"] = 4
    snapshot = model.cuda_graph_stats()
    snapshot["graph_replay_calls_by_logical_batch"]["2"] = 99
    assert model.cuda_graph_stats()["graph_replay_calls_by_logical_batch"]["2"] == 4
    model.reset_cuda_graph_stats()
    assert model.cuda_graph_stats()["graph_replay_calls_by_logical_batch"] == {"1": 0, "2": 0}
