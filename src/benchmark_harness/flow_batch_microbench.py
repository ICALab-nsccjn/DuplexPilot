"""A100 Flow-step microbenchmark for the public CFG-aware contract.

This benchmark measures only the public Flow Euler-step execution after
request state preparation. It never enables APR or changes serving behavior.
The default three strata use the frozen real token trace; when the available
trace has one observed length, P10/P50/P90 intentionally have the same length
and the report records that limitation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
STEP_AUDIO_ROOT = REPO_ROOT / "third_party" / "Step-Audio2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STEP_AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP_AUDIO_ROOT))

SEED_TOKENS = (1493, 4299, 4218, 2049, 528, 2752, 4850, 4569)
N_TIMESTEPS = 10
SHAPE_LABELS = ("P10", "P50", "P90")


def load_tokens(path: Path | None) -> tuple[list[int], str]:
    if path is None:
        values = list(SEED_TOKENS) * 2
        return values, "embedded_seed_token_trace"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tokens")
    if not isinstance(payload, list) or not payload:
        raise ValueError("token asset must contain a non-empty list")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in payload):
        raise ValueError("token asset values must be integers")
    return [int(value) for value in payload], str(path)


def extend_tokens(source: list[int], length: int) -> list[int]:
    if length <= 0:
        raise ValueError("shape length must be positive")
    repeats = (length + len(source) - 1) // len(source)
    return (source * repeats)[:length]


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_states(model, prompt_wav: str, tokens: list[int], batch_size: int):
    states = []
    for index in range(batch_size):
        stream_state = model.create_stream_state(prompt_wav)
        states.append(
            model.begin_chunk_steps(
                tokens,
                prompt_wav,
                stream_state,
                last_chunk=False,
                n_timesteps=N_TIMESTEPS,
                request_id=f"microbench-{batch_size}-{index}",
                generation_id=31,
                sequence_no=5,
                version=17,
            )
        )
    return tuple(states)


def execute_steps(model, states):
    values = tuple(states)
    for _ in range(N_TIMESTEPS):
        if len(values) == 1:
            values = (model.advance_chunk_step(values[0]),)
        else:
            values = tuple(model.advance_chunk_step_batch(values))
    return values


def finish_states(model, states):
    return tuple(model.finish_chunk_steps(state) for state in states)


class UtilizationSampler:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = False
        self.error: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}: {exc}"

    def _sample(self):
        while not self._stop.is_set():
            try:
                value = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                self.samples.append(float(value))
            except Exception:
                pass
            self._stop.wait(0.005)

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.available and self.samples:
            self.mean_pct = float(statistics.fmean(self.samples))
            self.max_pct = float(max(self.samples))
        else:
            self.mean_pct = None
            self.max_pct = None


def run_timed(model, prompt_wav: str, tokens: list[int], batch_size: int) -> dict[str, Any]:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    states = build_states(model, prompt_wav, tokens, batch_size)
    torch.cuda.synchronize()

    start_wall = time.perf_counter_ns()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with UtilizationSampler() as utilization:
        start_event.record()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
            states = execute_steps(model, states)
        end_event.record()
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - start_wall) / 1_000_000.0
    cuda_ms = float(start_event.elapsed_time(end_event))
    finished = finish_states(model, states)
    torch.cuda.synchronize()

    for index, state in enumerate(states):
        if state.step_index != N_TIMESTEPS:
            raise AssertionError(f"state {index} stopped at step {state.step_index}")
        if state.request_id != f"microbench-{batch_size}-{index}":
            raise AssertionError("logical identity changed during Flow execution")
    state_bytes = sum(state.logical_state_size_bytes() for state in states)
    mel_samples = sum(int(mel.numel()) for mel, _ in finished)
    peak_bytes = int(torch.cuda.max_memory_allocated())
    return {
        "batch_size": batch_size,
        "wall_time_ms": wall_ms,
        "cuda_time_ms": cuda_ms,
        "per_session_wall_ms": wall_ms / batch_size,
        "per_session_cuda_ms": cuda_ms / batch_size,
        "logical_samples_per_s": batch_size * N_TIMESTEPS / max(cuda_ms / 1000.0, 1e-12),
        "packing_splitting_overhead_ms_estimate": max(0.0, wall_ms - cuda_ms),
        "packing_splitting_overhead_ratio": max(0.0, wall_ms - cuda_ms) / max(wall_ms, 1e-12),
        "state_bytes": int(state_bytes),
        "peak_memory_bytes": peak_bytes,
        "gpu_utilization_mean_pct": utilization.mean_pct,
        "gpu_utilization_max_pct": utilization.max_pct,
        "gpu_utilization_status": "available" if utilization.available else "unavailable",
        "gpu_utilization_error": utilization.error,
        "mel_samples": mel_samples,
        "correctness": "PASS",
    }


def kernel_count_once(model, prompt_wav: str, tokens: list[int], batch_size: int) -> dict[str, Any]:
    states = build_states(model, prompt_wav, tokens, batch_size)
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ) as profiler:
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
                states = execute_steps(model, states)
        events = profiler.events()
        cuda_events = sum("cuda" in str(event.device_type).lower() for event in events)
        return {"kernel_count": int(cuda_events), "profile_event_count": len(events), "status": "PASS"}
    except Exception as exc:
        return {
            "kernel_count": None,
            "profile_event_count": None,
            "status": "UNAVAILABLE",
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    finally:
        del states
        gc.collect()
        torch.cuda.empty_cache()


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def ci_lower(values: list[float]) -> float:
    if len(values) < 2:
        return float(values[0]) if values else float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array) - 1.96 * np.std(array, ddof=1) / math.sqrt(len(array)))


def gate_for_shape(results: list[dict[str, Any]], shape_label: str) -> dict[str, Any]:
    by_batch = {
        batch: [
            item for item in results
            if item["batch_size"] == batch and item["shape"] == shape_label
        ]
        for batch in (1, 2, 4)
    }
    b1 = by_batch[1]
    b2 = by_batch[2]
    if not b1 or not b2:
        return {"shape": shape_label, "status": "NOT_RUN", "reason": "B1 or B2 absent"}
    ratios = [
        float(right["logical_samples_per_s"]) / float(left["logical_samples_per_s"])
        for left, right in zip(b1, b2)
    ]
    overhead = [float(item["packing_splitting_overhead_ratio"]) for item in b2]
    payload = {
        "shape": shape_label,
        "b2_median_throughput_ratio": median(ratios),
        "b2_ci95_lower_throughput_ratio": ci_lower(ratios),
        "b2_max_packing_splitting_ratio": max(overhead),
        "b2_correctness": all(item["correctness"] == "PASS" for item in b2),
        "b2_no_oom": all(item.get("oom", False) is False for item in b2),
        "ratios": ratios,
    }
    payload["status"] = (
        "PASS"
        if payload["b2_median_throughput_ratio"] >= 1.30
        and payload["b2_ci95_lower_throughput_ratio"] > 1.10
        and payload["b2_max_packing_splitting_ratio"] <= 0.05
        and payload["b2_correctness"]
        and payload["b2_no_oom"]
        else "FAIL"
    )
    if by_batch[4]:
        b4_ratios = [
            float(right["logical_samples_per_s"]) / float(left["logical_samples_per_s"])
            for left, right in zip(b1, by_batch[4])
        ]
        b4_peak = max(int(item["peak_memory_bytes"]) for item in by_batch[4])
        payload["b4_median_throughput_ratio"] = median(b4_ratios)
        payload["b4_peak_memory_bytes"] = b4_peak
        payload["b4_status"] = (
            "PASS"
            if payload["b4_median_throughput_ratio"] >= 1.60
            and b4_peak <= 32 * 1024**3
            and all(item["correctness"] == "PASS" for item in by_batch[4])
            else "FAIL"
        )
    return payload


def write_report(output_path: Path, payload: dict[str, Any]) -> Path:
    report_path = output_path.with_name("FLOW_BATCH_MICROBENCH_REPORT.md")
    lines = [
        "# Flow Batch Microbenchmark Report",
        "",
        f"- source commit: {payload['source_commit']}",
        f"- model: {payload['model_path']}",
        f"- prompt: {payload['prompt_wav']}",
        f"- token source: {payload['token_source']}",
        f"- token SHA256: {payload.get('token_sha256')}",
        f"- dtype: {payload['dtype']}",
        f"- warmups / measured repeats: {payload['warmups']} / {payload['repeats']}",
        f"- timesteps: {payload['n_timesteps']}",
        "",
        "The available frozen real profile contained one observed token length. "
        "Therefore P10/P50/P90 are three repeated shape strata at that frozen "
        "length unless a richer trace is supplied; this is recorded rather than "
        "presented as an empirical duration distribution.",
        "",
        "| stratum | B | median wall ms | median CUDA ms | median per-session ms | median logical samples/s | peak memory GiB | state bytes | kernel count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["summary"]:
        lines.append(
            f"| {item['shape']} | {item['batch_size']} | {item['wall_median_ms']:.3f} | "
            f"{item['cuda_median_ms']:.3f} | {item['per_session_median_ms']:.3f} | "
            f"{item['throughput_median']:.3f} | {item['peak_memory_gib']:.3f} | "
            f"{item['state_bytes_median']} | {item['kernel_count']} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        "| stratum | B2 median ratio | B2 95% CI lower | pack/split max | B2 result | B4 result |",
        "|---|---:|---:|---:|---|---|",
    ]
    for gate in payload["gates"]:
        lines.append(
            f"| {gate['shape']} | {gate.get('b2_median_throughput_ratio', float('nan')):.3f} | "
            f"{gate.get('b2_ci95_lower_throughput_ratio', float('nan')):.3f} | "
            f"{gate.get('b2_max_packing_splitting_ratio', float('nan')):.4f} | "
            f"{gate.get('status')} | {gate.get('b4_status', 'NOT_RUN')} |"
        )
    lines += [
        "",
        f"**B2 overall gate: {payload['b2_overall_gate']}**",
        "",
        "The packing/splitting value is a conservative host-orchestration estimate "
        "(wall time minus CUDA event time); it is not treated as a kernel attribution.",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_sizes not in ("1,2", "1,2,4"):
        raise ValueError("--batch-sizes must be exactly 1,2 or 1,2,4")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    token_path = Path(args.tokens) if args.tokens else None
    source_tokens, token_source = load_tokens(token_path)
    lengths = [int(value) for value in args.shape_lengths.split(",") if value.strip()]
    if len(lengths) != 3:
        raise ValueError("--shape-lengths must contain exactly three integers")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    from token2wav import Token2wav

    model = Token2wav(args.model_path, float16=False)
    raw_results: list[dict[str, Any]] = []
    for shape, length in zip(SHAPE_LABELS, lengths):
        tokens = extend_tokens(source_tokens, length)
        for batch_size in batch_sizes:
            for _ in range(args.warmups):
                run_timed(model, args.prompt_wav, tokens, batch_size)
            for repeat in range(args.repeats):
                try:
                    measurement = run_timed(model, args.prompt_wav, tokens, batch_size)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        measurement = {
                            "batch_size": batch_size,
                            "correctness": "FAIL",
                            "oom": True,
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    else:
                        raise
                measurement.update({"shape": shape, "shape_length": length, "repeat": repeat})
                raw_results.append(measurement)
            sample = run_timed(model, args.prompt_wav, tokens, batch_size)
            profile = kernel_count_once(model, args.prompt_wav, tokens, batch_size)
            sample.update(profile)
            raw_results.append({
                "shape": shape,
                "shape_length": length,
                "batch_size": batch_size,
                "repeat": "profile",
                **sample,
            })
    measured = [item for item in raw_results if isinstance(item["repeat"], int)]
    summary: list[dict[str, Any]] = []
    for shape in SHAPE_LABELS:
        for batch_size in batch_sizes:
            items = [item for item in measured if item["shape"] == shape and item["batch_size"] == batch_size]
            valid = [item for item in items if item.get("correctness") == "PASS"]
            profiles = [item for item in raw_results if item["shape"] == shape and item["batch_size"] == batch_size and item["repeat"] == "profile"]
            if not valid:
                summary.append({
                    "shape": shape,
                    "batch_size": batch_size,
                    "wall_median_ms": float("nan"),
                    "cuda_median_ms": float("nan"),
                    "per_session_median_ms": float("nan"),
                    "throughput_median": float("nan"),
                    "peak_memory_gib": float("nan"),
                    "state_bytes_median": 0,
                    "kernel_count": None,
                })
                continue
            summary.append({
                "shape": shape,
                "batch_size": batch_size,
                "wall_median_ms": median([float(item["wall_time_ms"]) for item in valid]),
                "cuda_median_ms": median([float(item["cuda_time_ms"]) for item in valid]),
                "per_session_median_ms": median([float(item["per_session_wall_ms"]) for item in valid]),
                "throughput_median": median([float(item["logical_samples_per_s"]) for item in valid]),
                "peak_memory_gib": max(float(item["peak_memory_bytes"]) for item in valid) / 1024**3,
                "state_bytes_median": int(median([float(item["state_bytes"]) for item in valid])),
                "kernel_count": profiles[0].get("kernel_count") if profiles else None,
            })
    gates = [gate_for_shape(measured, shape) for shape in SHAPE_LABELS]
    b2_overall = "PASS" if all(gate["status"] == "PASS" for gate in gates) else "FAIL"
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        source_commit = "unknown"
    output = {
        "schema_version": "flow-batch-microbench-v1",
        "source_commit": source_commit,
        "model_path": args.model_path,
        "prompt_wav": args.prompt_wav,
        "token_source": token_source,
        "token_sha256": sha256_file(token_path),
        "shape_lengths": dict(zip(SHAPE_LABELS, lengths)),
        "dtype": "float32",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "n_timesteps": N_TIMESTEPS,
        "batch_sizes": batch_sizes,
        "raw_results": raw_results,
        "summary": summary,
        "gates": gates,
        "b2_overall_gate": b2_overall,
        "apr_enabled": False,
        "rsv_dsv_modified": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    report_path = write_report(output_path, output)
    output["report_path"] = str(report_path)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if b2_overall == "PASS" else "GATE_FAIL",
        "output": str(output_path),
        "report": str(report_path),
        "gates": gates,
    }, indent=2, sort_keys=True))
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--tokens")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--shape-lengths", default="16,16,16")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
