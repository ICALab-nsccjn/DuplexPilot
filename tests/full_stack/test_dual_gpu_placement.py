import pytest

from lychee_fd.runtime.dual_gpu_placement import (
    DualGpuPlacementError,
    resolve_dual_gpu_placement,
)


def test_resolve_fixed_model_and_acoustic_gpu_roles():
    placement = resolve_dual_gpu_placement(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "LYCHEEFD_TOKEN2WAV_DEVICE": "1",
            "LYCHEEFD_VLLM_TENSOR_PARALLEL_SIZE": "1",
            "LYCHEEFD_VLLM_PIPELINE_PARALLEL_SIZE": "1",
        },
        cuda_device_count=2,
    )

    assert placement.visible_devices == "0,1"
    assert placement.model_device == 0
    assert placement.acoustic_device == 1
    assert placement.tensor_parallel_size == 1
    assert placement.pipeline_parallel_size == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"CUDA_VISIBLE_DEVICES": "1,0"},
        {"LYCHEEFD_TOKEN2WAV_DEVICE": "0"},
        {"LYCHEEFD_VLLM_TENSOR_PARALLEL_SIZE": "2"},
        {"LYCHEEFD_VLLM_PIPELINE_PARALLEL_SIZE": "2"},
    ],
)
def test_rejects_non_fixed_dual_gpu_roles(overrides):
    env = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "LYCHEEFD_TOKEN2WAV_DEVICE": "1",
        "LYCHEEFD_VLLM_TENSOR_PARALLEL_SIZE": "1",
        "LYCHEEFD_VLLM_PIPELINE_PARALLEL_SIZE": "1",
    }
    env.update(overrides)

    with pytest.raises(DualGpuPlacementError):
        resolve_dual_gpu_placement(env, cuda_device_count=2)


def test_rejects_single_visible_cuda_device():
    with pytest.raises(DualGpuPlacementError, match="two CUDA devices"):
        resolve_dual_gpu_placement(
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "LYCHEEFD_TOKEN2WAV_DEVICE": "1",
            },
            cuda_device_count=1,
        )
