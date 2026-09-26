"""Fail-closed configuration for real-model/local-acoustic APR evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import sys
from typing import Any

from lychee_fd.runtime.apr.profiling import PROFILE_SCHEMA_VERSION


REAL_MODE = "real_model_local_acoustic"
MEASUREMENT_SCHEMA_VERSION = "apr-e2e-v1"


class RealE2EConfigError(ValueError):
    """Raised when a real E2E configuration cannot be proven valid."""


@dataclass(frozen=True)
class RealE2EConfig:
    model_path: Path
    token2wav_path: Path
    prompt_wav: Path
    worker_count: int = 2
    mode: str = REAL_MODE
    max_model_len: int = 128
    gpu_memory_utilization: float = 0.60
    max_num_seqs: int = 2
    max_num_batched_tokens: int = 128
    acoustic_chunk_size: int = 28

    @classmethod
    def from_env(cls) -> "RealE2EConfig":
        return cls(
            model_path=Path(os.environ.get(
                "APR_E2E_MODEL_PATH",
                "/mnt/DuplexPilot/data/models/lychee_full_duplex",
            )),
            token2wav_path=Path(os.environ.get(
                "APR_E2E_TOKEN2WAV_PATH",
                "/mnt/DuplexPilot/data/models/token2wav",
            )),
            prompt_wav=Path(os.environ.get(
                "APR_E2E_PROMPT_WAV",
                "/mnt/DuplexPilot/data/DuplexPilot/lychee_dsv_closure_worktree/frontend/public/clone_24k_mono/default_male.wav",
            )),
            worker_count=int(os.environ.get("APR_E2E_WORKERS", "2")),
            mode=os.environ.get("APR_E2E_MODE", REAL_MODE),
            max_model_len=int(os.environ.get("APR_E2E_MAX_MODEL_LEN", "128")),
            gpu_memory_utilization=float(os.environ.get("APR_E2E_GPU_MEMORY_UTILIZATION", "0.60")),
            max_num_seqs=int(os.environ.get("APR_E2E_MAX_NUM_SEQS", "2")),
            max_num_batched_tokens=int(os.environ.get("APR_E2E_MAX_NUM_BATCHED_TOKENS", "128")),
            acoustic_chunk_size=int(os.environ.get("APR_E2E_ACOUSTIC_CHUNK_SIZE", "28")),
        )

    def validate_paths(self) -> None:
        errors: list[str] = []
        if self.mode != REAL_MODE:
            errors.append(f"mode must be {REAL_MODE!r}, got {self.mode!r}")
        if isinstance(self.worker_count, bool) or self.worker_count <= 0:
            errors.append("worker_count must be a positive integer")
        if self.max_model_len <= 0:
            errors.append("max_model_len must be positive")
        if self.max_num_seqs <= 0:
            errors.append("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            errors.append("max_num_batched_tokens must be positive")
        if self.acoustic_chunk_size <= 0:
            errors.append("acoustic_chunk_size must be positive")
        if not 0.0 < float(self.gpu_memory_utilization) <= 1.0:
            errors.append("gpu_memory_utilization must be in (0, 1]")
        if not self.model_path.is_dir():
            errors.append(f"model_path is not a directory: {self.model_path}")
        elif not (self.model_path / "config.json").is_file():
            errors.append(f"model_path lacks config.json: {self.model_path}")
        if not self.token2wav_path.is_dir():
            errors.append(f"token2wav_path is not a directory: {self.token2wav_path}")
        else:
            for filename in ("flow.pt", "hift.pt"):
                if not (self.token2wav_path / filename).is_file():
                    errors.append(f"token2wav_path lacks {filename}: {self.token2wav_path}")
        if not self.prompt_wav.is_file() or self.prompt_wav.stat().st_size <= 0:
            errors.append(f"prompt_wav is missing or empty: {self.prompt_wav}")
        if errors:
            raise RealE2EConfigError("; ".join(errors))

    def manifest(self) -> dict[str, Any]:
        """Return configuration metadata without loading model or CUDA state."""
        return {
            "mode": self.mode,
            "model_path": str(self.model_path),
            "token2wav_path": str(self.token2wav_path),
            "prompt_wav": str(self.prompt_wav),
            "worker_count": int(self.worker_count),
            "max_model_len": int(self.max_model_len),
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
            "max_num_seqs": int(self.max_num_seqs),
            "max_num_batched_tokens": int(self.max_num_batched_tokens),
            "acoustic_chunk_size": int(self.acoustic_chunk_size),
            "model_checkpoint_loaded": False,
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        }


def build_profile_manifest(
    config: RealE2EConfig,
    *,
    run_id: str,
    system: str,
    workload: str,
    concurrency: int,
    repeat: int,
    warmup: bool,
    source_commit: str,
    workload_trace_hash: str,
    profiling_enabled: bool,
    torch_profiler_enabled: bool,
    nsight_enabled: bool,
) -> dict[str, Any]:
    """Build a fail-closed identity record for an isolated APR profile run."""
    if not isinstance(config, RealE2EConfig):
        raise RealE2EConfigError("config must be a RealE2EConfig")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RealE2EConfigError("run_id must be non-empty")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise RealE2EConfigError("source_commit must be non-empty")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise RealE2EConfigError("concurrency must be positive")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise RealE2EConfigError("repeat must be positive")
    for name, value in (
        ("profiling_enabled", profiling_enabled),
        ("torch_profiler_enabled", torch_profiler_enabled),
        ("nsight_enabled", nsight_enabled),
        ("warmup", warmup),
    ):
        if not isinstance(value, bool):
            raise RealE2EConfigError(f"{name} must be a boolean")
    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "run_id": run_id,
        "system": system,
        "workload": workload,
        "concurrency": concurrency,
        "repeat": repeat,
        "warmup": warmup,
        "profiling_enabled": profiling_enabled,
        "torch_profiler_enabled": torch_profiler_enabled,
        "nsight_enabled": nsight_enabled,
        "source_commit": source_commit,
        "model_checkpoint": str(config.model_path),
        "token2wav_checkpoint": str(config.token2wav_path),
        "gpu_mapping": {
            "model_execution": "GPU0",
            "local_token2wav": "GPU1",
            "visible_devices": "0,1",
        },
        "worker_count": int(config.worker_count),
        "workload_trace_hash": workload_trace_hash,
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "runtime_config": config.manifest(),
    }
