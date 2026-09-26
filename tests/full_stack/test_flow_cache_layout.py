from __future__ import annotations

import pytest
import torch

from profiling.token2wav_flow_profile.batching import FlowStateBatchAdapter
from profiling.token2wav_flow_profile.contracts import FlowStep
from profiling.token2wav_flow_profile.layout import (
    FlowCacheLayoutSpec,
    real_flow_cache_layout,
)


def _tensor(shape: tuple[int, ...], value: float) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def _step(request_id: str, value: float) -> FlowStep:
    return FlowStep(
        request_id=request_id,
        generation_id=0,
        sequence_no=0,
        tokens=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        speaker=_tensor((1, 4), value),
        flow_cache={
            "conformer_cnn_cache": _tensor((1, 2, 3), value),
            "conformer_att_cache": _tensor((5, 1, 2, 3), value),
            "estimator_cnn_cache": _tensor((2, 3, 2, 4), value),
            "estimator_att_cache": _tensor((2, 3, 2, 4, 5), value),
        },
        last_chunk=False,
        n_timesteps=10,
        model_identity="real-flow-diagnostic",
    )


def test_real_layout_declares_each_batch_axis() -> None:
    layout = real_flow_cache_layout()

    assert layout.batch_axis("conformer_cnn_cache") == 0
    assert layout.batch_axis("conformer_att_cache") == 1
    assert layout.batch_axis("estimator_cnn_cache") == 2
    assert layout.batch_axis("estimator_att_cache") == 2
    assert layout.batch_extent("estimator_cnn_cache") == 2
    assert layout.batch_extent("estimator_att_cache") == 2


def test_unknown_key_is_rejected() -> None:
    layout = FlowCacheLayoutSpec.from_mapping({"known": 0})

    with pytest.raises(ValueError, match="missing batch axis"):
        layout.batch_axis("unknown")


def test_layout_aware_pack_and_split_use_declared_axes() -> None:
    adapter = FlowStateBatchAdapter(layout_spec=real_flow_cache_layout())
    first = _step("request-a", 1.0)
    second = _step("request-b", 2.0)

    packed = adapter.pack((first, second))

    assert packed.flow_cache["conformer_cnn_cache"].shape == (2, 2, 3)
    assert packed.flow_cache["conformer_att_cache"].shape == (5, 2, 2, 3)
    assert packed.flow_cache["estimator_cnn_cache"].shape == (2, 3, 4, 4)
    assert packed.flow_cache["estimator_att_cache"].shape == (2, 3, 4, 4, 5)
    assert packed.cache_batch_axes == (
        ("conformer_cnn_cache", 0),
        ("conformer_att_cache", 1),
        ("estimator_cnn_cache", 2),
        ("estimator_att_cache", 2),
    )
    assert packed.cache_batch_extents == (
        ("conformer_cnn_cache", 1),
        ("conformer_att_cache", 1),
        ("estimator_cnn_cache", 2),
        ("estimator_att_cache", 2),
    )

    output_mel = _tensor((2, 6, 7), 9.0)
    output_cache = {
        key: tensor.clone() for key, tensor in packed.flow_cache.items()
    }
    split = adapter.split(packed, output_mel, output_cache)

    assert len(split) == 2
    assert split[0][0].flow_cache["conformer_att_cache"].shape == (5, 1, 2, 3)
    assert split[1][0].flow_cache["estimator_att_cache"].shape == (2, 3, 2, 4, 5)


def test_split_allows_progression_cache_to_grow_non_batch_dimensions() -> None:
    adapter = FlowStateBatchAdapter(layout_spec=real_flow_cache_layout())
    packed = adapter.pack((_step("request-a", 1.0), _step("request-b", 2.0)))
    output_mel = _tensor((2, 6, 7), 9.0)
    output_cache = {
        key: tensor.clone() for key, tensor in packed.flow_cache.items()
    }
    output_cache["conformer_att_cache"] = torch.cat(
        [
            output_cache["conformer_att_cache"],
            torch.zeros((5, 2, 2, 1), dtype=torch.float32),
        ],
        dim=3,
    )

    split = adapter.split(packed, output_mel, output_cache)

    assert split[0][0].flow_cache["conformer_att_cache"].shape == (5, 1, 2, 4)
    assert split[1][0].flow_cache["conformer_att_cache"].shape == (5, 1, 2, 4)
