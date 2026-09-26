"""Isolated torch.profiler wrapper for APR diagnostic runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def run_torch_profile(
    model_step: Callable[[], Any],
    output_path: str | Path,
    enabled: bool,
    *,
    step_count: int = 1,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Run existing model steps and optionally emit one finite torch trace."""
    if not callable(model_step):
        raise TypeError("model_step must be callable")
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count <= 0:
        raise ValueError("step_count must be a positive integer")
    output = Path(output_path)
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a boolean")

    if not enabled:
        for _ in range(step_count):
            model_step()
        return {
            "enabled": False,
            "model_steps": step_count,
            "profiler_steps": 0,
            "trace_path": None,
            "cuda_synchronize": False,
        }

    torch = torch_module
    if torch is None:
        try:
            import torch as torch  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("TORCH_PROFILER_UNAVAILABLE") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler_steps = 0

    def export_trace(profiler: Any) -> None:
        profiler.export_chrome_trace(str(output))

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=step_count,
            repeat=1,
        ),
        on_trace_ready=export_trace,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(step_count):
            model_step()
            profiler.step()
            profiler_steps += 1
        if not output.exists():
            profiler.export_chrome_trace(str(output))

    return {
        "enabled": True,
        "model_steps": step_count,
        "profiler_steps": profiler_steps,
        "trace_path": str(output),
        "cuda_synchronize": False,
    }
