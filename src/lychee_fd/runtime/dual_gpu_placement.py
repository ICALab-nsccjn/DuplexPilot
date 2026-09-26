"""Fail-closed two-A100 placement contract for model/acoustic experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


class DualGpuPlacementError(RuntimeError):
    """Raised when the fixed two-GPU experiment contract is not satisfied."""


@dataclass(frozen=True)
class DualGpuPlacement:
    """Logical CUDA roles under the required ``0,1`` visibility order."""

    visible_devices: str
    model_device: int
    acoustic_device: int
    tensor_parallel_size: int
    pipeline_parallel_size: int


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = str(env.get(name, default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise DualGpuPlacementError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise DualGpuPlacementError(f"{name} must be positive, got {value}")
    return value


def resolve_dual_gpu_placement(
    env: Mapping[str, str] | None = None,
    *,
    cuda_device_count: int | None = None,
) -> DualGpuPlacement:
    """Validate and return the fixed model-GPU0/acoustic-GPU1 contract.

    ``CUDA_VISIBLE_DEVICES=0,1`` is intentionally exact.  This keeps logical
    CUDA indices identical to physical GPU indices and prevents an accidental
    ``1,0`` remapping from making telemetry misleading.
    """

    values = os.environ if env is None else env
    visible = str(values.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible != "0,1":
        raise DualGpuPlacementError(
            'CUDA_VISIBLE_DEVICES must be exactly "0,1" for the fixed two-GPU run'
        )
    if cuda_device_count is None:
        import torch

        cuda_device_count = int(torch.cuda.device_count())
    if int(cuda_device_count) != 2:
        raise DualGpuPlacementError(
            f"two CUDA devices are required, detected {cuda_device_count}"
        )

    acoustic_raw = str(values.get("LYCHEEFD_TOKEN2WAV_DEVICE", "")).strip()
    if acoustic_raw != "1":
        raise DualGpuPlacementError(
            'LYCHEEFD_TOKEN2WAV_DEVICE must be "1" so acoustic execution stays on GPU1'
        )
    tensor_parallel = _positive_int(values, "LYCHEEFD_VLLM_TENSOR_PARALLEL_SIZE", 1)
    pipeline_parallel = _positive_int(values, "LYCHEEFD_VLLM_PIPELINE_PARALLEL_SIZE", 1)
    if tensor_parallel != 1 or pipeline_parallel != 1:
        raise DualGpuPlacementError(
            "vLLM tensor/pipeline parallelism must both equal 1 for the fixed role split"
        )

    return DualGpuPlacement(
        visible_devices=visible,
        model_device=0,
        acoustic_device=1,
        tensor_parallel_size=tensor_parallel,
        pipeline_parallel_size=pipeline_parallel,
    )


def dual_gpu_required(env: Mapping[str, str] | None = None) -> bool:
    """Return whether production startup must enforce the dual-GPU contract."""

    values = os.environ if env is None else env
    return str(values.get("LYCHEEFD_REQUIRE_DUAL_GPU_PLACEMENT", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
