"""Benchmark-only acoustic backend selector for the B0-B3 harness."""

from __future__ import annotations

from typing import Any

from lychee_fd.runtime.apr.paper_systems import PaperSystemSpec
from tools.apr.flow_batch_acoustic_lane import APRFlowBatchAcousticLane
from tools.apr.real_acoustic_lanes import FixedAffinityAcousticLane


def runtime_mode_for(spec: PaperSystemSpec) -> str:
    if not isinstance(spec, PaperSystemSpec):
        raise TypeError("spec must be a PaperSystemSpec")
    return spec.model_runtime_mode


def build_acoustic_lane(
    spec: PaperSystemSpec,
    *,
    worker_count: int,
    model: Any,
    prompt_wav: str,
    event_sink: Any | None = None,
) -> Any:
    """Build a lane from the public system spec, without private state access."""
    if not isinstance(spec, PaperSystemSpec):
        raise TypeError("spec must be a PaperSystemSpec")
    common = {
        "worker_count": worker_count,
        "model": model,
        "prompt_wav": prompt_wav,
    }
    if event_sink is not None:
        common["event_sink"] = event_sink
    if spec.acoustic_mode == "fixed_affinity":
        return FixedAffinityAcousticLane(**common)
    return APRFlowBatchAcousticLane(
        **common,
        max_batch_size=spec.max_flow_batch_size,
    )
