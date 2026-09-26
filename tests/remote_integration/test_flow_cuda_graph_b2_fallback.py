from cosyvoice2.flow.decoder_dit import _graph_fallback_reason


def _base(**overrides):
    values = {
        "logical_batch_size": 2,
        "captured_logical_batch_sizes": (1, 2),
        "chunk_sizes": (30, 48, 96),
        "current_chunk_lengths": (30, 30),
        "cnn_cache_shapes": ((16, 2, 1024, 2), (16, 2, 1024, 2)),
        "last_chunks": (False, False),
        "n_timesteps": (10, 10),
        "generation_valid": True,
        "attention_cache_lengths": (20, 22),
        "max_graph_cache": 1000,
        "step_indices": (0, 3),
    }
    values.update(overrides)
    return values


def test_fallback_reasons_are_specific_and_fail_closed():
    assert _graph_fallback_reason(**_base(current_chunk_lengths=(30, 48))) == "current_chunk_shape"
    assert _graph_fallback_reason(**_base(cnn_cache_shapes=((16, 2, 1024, 2), (16, 2, 1024, 3)))) == "cnn_cache_shape"
    assert _graph_fallback_reason(**_base(last_chunks=(False, True))) == "last_chunk"
    assert _graph_fallback_reason(**_base(n_timesteps=(10, 9))) == "n_timesteps"
    assert _graph_fallback_reason(**_base(generation_valid=False)) == "invalid_generation"


def test_different_cache_lengths_and_step_indices_are_graph_eligible():
    assert _graph_fallback_reason(**_base()) is None
    assert _graph_fallback_reason(**_base(attention_cache_lengths=(20, 22))) is None
