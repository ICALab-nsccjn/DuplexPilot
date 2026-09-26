"""Run the fixed-work CUDA-Graph x logical-B=2 causal experiment on A100.

The runner intentionally measures only the acoustic Flow contract after token
preparation.  It keeps the work (tokens, chunk shape, ten Euler steps, and
request identities) fixed across the four cells, so a Graph/B=2 interaction is
not confounded with model sampling or request arrival differences.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Step-Audio2") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Step-Audio2"))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


N_TIMESTEPS = 10
SHAPES = ("P10", "P50", "P90")
DEFAULT_SHAPE_LENGTHS = (30, 48, 96)
SEED_ENVELOPES = (
    (1493, 4299, 4218, 2049, 528, 2752, 4850, 4569),
    (2171, 5103, 389, 6001, 1733, 4444, 721, 3099),
    (808, 2777, 5312, 1630, 4021, 944, 6150, 2588),
)

CSV_FIELDS = (
    "variant",
    "graph_enabled",
    "logical_batch",
    "repeat",
    "shape",
    "shape_length",
    "envelope",
    "wall_time_ms",
    "cuda_time_ms",
    "logical_samples_per_s",
    "per_session_wall_ms",
    "per_session_cuda_ms",
    "packing_splitting_overhead_ms",
    "packing_splitting_overhead_ratio",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "logical_state_bytes",
    "graph_forward_calls",
    "graph_replay_calls",
    "graph_fallback_calls",
    "graph_eligible_calls",
    "graph_fallback_reasons",
    "graph_hit",
    "kernel_count",
    "correctness",
    "mel_sha256",
    "state_sha256",
)


def compute_interaction_effect(b1_graph_effect: float, b2_graph_effect: float) -> float:
    """Return the additive interaction on the supplied effect scale."""
    return float(b2_graph_effect) - float(b1_graph_effect)


def evaluate_fixed_work_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered fixed-work Graph x B=2 gate."""
    reasons: list[str] = []
    if not bool(metrics.get("correctness", False)):
        reasons.append("correctness")
    if int(metrics.get("b2_graph_steps", -1)) != N_TIMESTEPS:
        reasons.append("not_all_b2_steps_hit_graph")
    eligible = int(metrics.get("b2_graph_eligible", 0))
    if eligible <= 0 or int(metrics.get("b2_graph_steps", 0)) < eligible:
        reasons.append("invalid_eligibility_accounting")
    if float(metrics.get("b2_graph_throughput_ratio", 0.0)) < 1.10:
        reasons.append("throughput_ratio_below_1.10")
    if float(metrics.get("b2_graph_ci_lower", 0.0)) <= 1.00:
        reasons.append("bootstrap_or_pair_ci_lower_not_above_1")
    if int(metrics.get("peak_memory_bytes", 0)) > 32 * 1024**3:
        reasons.append("peak_memory_over_32_gib")
    return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons}


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else float("nan")


def summarize_comparisons(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize Graph effects separately at logical B=1 and B=2."""
    grouped: dict[tuple[int, bool], list[float]] = defaultdict(list)
    for row in rows:
        # Keep this helper convenient for small audit callers while the raw
        # benchmark uses the more explicit column names below.
        batch = row.get("logical_batch", row.get("batch"))
        graph = row.get("graph_enabled", row.get("graph"))
        throughput = row.get("throughput", row.get("logical_samples_per_s"))
        if batch is None or graph is None or throughput is None:
            raise KeyError("comparison rows need batch/graph/throughput fields")
        grouped[(int(batch), bool(graph))].append(float(throughput))
    b1_off = _median(grouped[(1, False)])
    b1_on = _median(grouped[(1, True)])
    b2_off = _median(grouped[(2, False)])
    b2_on = _median(grouped[(2, True)])
    b1_ratio = b1_on / b1_off
    b2_ratio = b2_on / b2_off
    return {
        "b1_graph_effect_ratio": b1_ratio,
        "b2_graph_effect_ratio": b2_ratio,
        "interaction_ratio_difference": compute_interaction_effect(b1_ratio, b2_ratio),
    }


def _extend_tokens(source: tuple[int, ...], length: int) -> list[int]:
    repeats = (length + len(source) - 1) // len(source)
    return list((source * repeats)[:length])


def derive_graph_chunk_sizes(
    token_lengths: Iterable[int], pre_lookahead_len: int, up_rate: int
) -> tuple[int, ...]:
    """Map token-shape fixtures to the encoder's actual Flow mel geometry."""
    lengths = tuple(int(value) for value in token_lengths)
    pre_lookahead_len = int(pre_lookahead_len)
    up_rate = int(up_rate)
    if not lengths or any(value <= pre_lookahead_len for value in lengths):
        raise ValueError("token lengths must exceed the encoder lookahead length")
    if pre_lookahead_len < 0 or up_rate <= 0:
        raise ValueError("encoder lookahead and up-rate must be valid")
    chunk_sizes = tuple((value - pre_lookahead_len) * up_rate for value in lengths)
    if any(value <= 0 or value > 256 for value in chunk_sizes):
        raise ValueError("derived Flow chunk sizes must be in [1, 256]")
    return chunk_sizes


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_tree(item) for item in value)
    return value


def _tensor_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            cpu = item.detach().contiguous().cpu()
            digest.update(str(tuple(cpu.shape)).encode())
            digest.update(str(cpu.dtype).encode())
            digest.update(cpu.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif item is not None:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _configure() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = int(os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE", "1"))
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _reset_model_seed(seed: int = 0) -> None:
    """Give graph and control models identical checkpoint-local RNG state."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_states(model: Any, prompt: str, tokens: list[int], batch: int, envelope: int):
    states = []
    for row in range(batch):
        stream = model.create_stream_state(prompt)
        states.append(
            model.begin_chunk_steps(
                tokens,
                prompt,
                stream,
                last_chunk=False,
                n_timesteps=N_TIMESTEPS,
                request_id=f"fixed-{envelope}-{batch}-{row}",
                generation_id=100 + envelope,
                sequence_no=3,
                version=7 + row,
            )
        )
    return tuple(states)


def _execute(model: Any, states: tuple[Any, ...]) -> tuple[Any, ...]:
    values = states
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(N_TIMESTEPS):
            if len(values) == 1:
                values = (model.advance_chunk_step(values[0]),)
            else:
                values = tuple(model.advance_chunk_step_batch(values))
    return values


def _profile_kernel_count(model: Any, states: tuple[Any, ...]) -> int | None:
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
            _execute(model, states)
        return int(
            sum(
                1
                for event in profiler.events()
                if str(getattr(event, "device_type", "")).lower().find("cuda") >= 0
            )
        )
    except Exception:
        return None


def _load_model(model_path: str):
    from token2wav import Token2wav

    return Token2wav(model_path, float16=False)


def _configure_graph(model: Any, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "capture_peak_allocated_bytes": 0}
    torch.cuda.reset_peak_memory_stats()
    model.flow.scatter_cuda_graph(True, logical_batch_sizes=(1, 2))
    torch.cuda.synchronize()
    return {
        "enabled": True,
        "capture_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "capture_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _run_cell(
    model: Any,
    *,
    variant: str,
    graph_enabled: bool,
    logical_batch: int,
    shape: str,
    shape_length: int,
    envelope: int,
    tokens: list[int],
    prompt: str,
    profile_kernels: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    states = _build_states(model, prompt, tokens, logical_batch, envelope)
    torch.cuda.synchronize()
    # Exclude state preparation from the execution timing and memory peak.
    torch.cuda.reset_peak_memory_stats()
    if hasattr(model.flow, "reset_cuda_graph_stats"):
        model.flow.reset_cuda_graph_stats()
    start_wall = time.perf_counter_ns()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    values = _execute(model, states)
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - start_wall) / 1_000_000.0
    cuda_ms = float(start_event.elapsed_time(end_event))
    finished = tuple(model.finish_chunk_steps(state) for state in values)
    torch.cuda.synchronize()
    stats = model.flow.cuda_graph_stats() if hasattr(model.flow, "cuda_graph_stats") else {}
    by_batch = str(logical_batch)
    fallback_reasons = stats.get("fallback_reasons_by_logical_batch", {}).get(by_batch, {})
    graph_forward = int(stats.get("forward_chunk_calls_by_logical_batch", {}).get(by_batch, 0))
    graph_replay = int(stats.get("graph_replay_calls_by_logical_batch", {}).get(by_batch, 0))
    graph_fallback = int(stats.get("fallback_calls_by_logical_batch", {}).get(by_batch, 0))
    graph_eligible = int(stats.get("eligible_calls", 0))
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    state_hash = _tensor_sha256(tuple(state.x for state in values))
    mel_hash = _tensor_sha256(tuple(mel for mel, _ in finished))
    kernel_count = _profile_kernel_count(model, _build_states(model, prompt, tokens, logical_batch, envelope)) if profile_kernels else None
    logical_state_bytes = sum(int(state.logical_state_size_bytes()) for state in values)
    row = {
        "variant": variant,
        "graph_enabled": graph_enabled,
        "logical_batch": logical_batch,
        "shape": shape,
        "shape_length": shape_length,
        "envelope": envelope,
        "wall_time_ms": wall_ms,
        "cuda_time_ms": cuda_ms,
        "logical_samples_per_s": logical_batch * N_TIMESTEPS / max(wall_ms / 1000.0, 1e-12),
        "per_session_wall_ms": wall_ms / logical_batch,
        "per_session_cuda_ms": cuda_ms / logical_batch,
        "packing_splitting_overhead_ms": max(0.0, wall_ms - cuda_ms),
        "packing_splitting_overhead_ratio": max(0.0, wall_ms - cuda_ms) / max(wall_ms, 1e-12),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "logical_state_bytes": logical_state_bytes,
        "graph_forward_calls": graph_forward,
        "graph_replay_calls": graph_replay,
        "graph_fallback_calls": graph_fallback,
        "graph_eligible_calls": graph_eligible,
        "graph_fallback_reasons": fallback_reasons,
        "graph_hit": bool(graph_enabled and graph_replay >= N_TIMESTEPS),
        "kernel_count": kernel_count,
        "correctness": "PASS" if all(state.step_index == N_TIMESTEPS for state in values) else "FAIL",
        "mel_sha256": mel_hash,
        "state_sha256": state_hash,
    }
    outputs = {
        "variant": variant,
        "logical_batch": logical_batch,
        "shape": shape,
        "shape_length": shape_length,
        "envelope": envelope,
        "mel": tuple(mel.detach().cpu().clone() for mel, _ in finished),
        "states": tuple(state.x.detach().cpu().clone() for state in values),
    }
    del states, values, finished
    gc.collect()
    return row, outputs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field], sort_keys=True)
                    if isinstance(row.get(field), dict)
                    else row.get(field, "")
                    for field in CSV_FIELDS
                }
            )


def _compare_outputs(outputs: dict[tuple[str, int, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for batch in (1, 2):
        for shape in SHAPES:
            for envelope in range(len(SEED_ENVELOPES)):
                off = outputs[("no_graph", batch, shape, envelope)]
                on = outputs[("graph", batch, shape, envelope)]
                mel_diffs = []
                state_diffs = []
                for left, right in zip(off["mel"], on["mel"]):
                    mel_diffs.append(float((left - right).abs().max().item()))
                for left, right in zip(off["states"], on["states"]):
                    state_diffs.append(float((left - right).abs().max().item()))
                comparisons.append(
                    {
                        "logical_batch": batch,
                        "shape": shape,
                        "envelope": envelope,
                        "mel_max_abs_diff": max(mel_diffs, default=0.0),
                        "state_max_abs_diff": max(state_diffs, default=0.0),
                        "status": "PASS" if max(mel_diffs, default=0.0) <= 1e-4 else "FAIL",
                    }
                )
    return comparisons


def _bootstrap_ci(values: list[float], samples: int = 10000, seed: int = 20260828) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GRAPH_B2 Fixed-Work Report",
        "",
        "This is a fixed-work acoustic-path causal experiment. It is not a full online serving result.",
        "",
        f"- source commit: `{payload['source_commit']}`",
        f"- model: `{payload['model_path']}`",
        f"- prompt: `{payload['prompt']}`",
        f"- logical steps: `{N_TIMESTEPS}`",
        f"- Graph max cache: `{payload['graph_max_cache']}`",
        "",
        "## Median measurements",
        "",
        "| variant | logical B | shape | wall ms | CUDA ms | logical samples/s | Graph replays | fallbacks | peak GiB |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["summary"]:
        lines.append(
            f"| {item['variant']} | {item['logical_batch']} | {item['shape']} | "
            f"{item['wall_ms']:.3f} | {item['cuda_ms']:.3f} | {item['throughput']:.3f} | "
            f"{item['graph_replays']} | {item['fallbacks']} | {item['peak_gib']:.3f} |"
        )
    lines += [
        "",
        "## Graph/B=2 numerical comparison",
        "",
        "| B | shape | max mel abs diff | max state abs diff | result |",
        "|---:|---|---:|---:|---|",
    ]
    for item in payload["comparisons"]:
        lines.append(
            f"| {item['logical_batch']} | {item['shape']} | {item['mel_max_abs_diff']:.6g} | "
            f"{item['state_max_abs_diff']:.6g} | {item['status']} |"
        )
    lines += [
        "",
        "## Causal effects",
        "",
        f"- Graph effect at B=1 (ratio): `{payload['effects']['b1_graph_effect_ratio']:.4f}`",
        f"- Graph effect at B=2 (ratio): `{payload['effects']['b2_graph_effect_ratio']:.4f}`",
        f"- interaction (B2 effect minus B1 effect): `{payload['effects']['interaction_ratio_difference']:.4f}`",
        "",
        "The effects are computed on the same fixed token envelopes and are not added together.",
        "",
        "## Gate",
        "",
        f"- fixed-work gate: **{payload['gate']['status']}**",
        f"- reasons: `{', '.join(payload['gate']['reasons']) or 'none'}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _configure()
    from token2wav import Token2wav

    shape_lengths = tuple(int(value) for value in args.shape_lengths.split(","))
    if len(shape_lengths) != 3 or any(value <= 0 for value in shape_lengths):
        raise ValueError("--shape-lengths must contain three positive integers")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prompt = str(args.prompt_wav)
    all_rows: list[dict[str, Any]] = []
    outputs: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    capture_info: dict[str, Any] = {}

    derived_graph_chunks: tuple[int, ...] | None = None
    for graph_enabled, variant in ((False, "no_graph"), (True, "graph")):
        _reset_model_seed()
        model = _load_model(str(args.model_path))
        if graph_enabled:
            derived_graph_chunks = derive_graph_chunk_sizes(
                shape_lengths,
                int(getattr(model.flow, "pre_lookahead_len")),
                int(getattr(model.flow, "up_rate")),
            )
            # The fixed-work fixture describes token lengths, whereas Graph
            # keys use the post-encoder Flow mel length.  Keep this mapping
            # explicit in the benchmark environment and output manifest.
            os.environ["LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS"] = ",".join(
                str(value) for value in derived_graph_chunks
            )
        capture_info[variant] = _configure_graph(model, graph_enabled)
        for shape, length in zip(SHAPES, shape_lengths):
            for envelope, source in enumerate(SEED_ENVELOPES):
                tokens = _extend_tokens(source, length)
                for batch in (1, 2):
                    for warmup in range(args.warmups):
                        _run_cell(
                            model,
                            variant=variant,
                            graph_enabled=graph_enabled,
                            logical_batch=batch,
                            shape=shape,
                            shape_length=length,
                            envelope=envelope,
                            tokens=tokens,
                            prompt=prompt,
                            profile_kernels=False,
                        )
                    for repeat in range(args.repeats):
                        row, result = _run_cell(
                            model,
                            variant=variant,
                            graph_enabled=graph_enabled,
                            logical_batch=batch,
                            shape=shape,
                            shape_length=length,
                            envelope=envelope,
                            tokens=tokens,
                            prompt=prompt,
                            profile_kernels=args.profile_kernels and repeat == 0,
                        )
                        row["repeat"] = repeat
                        all_rows.append(row)
                        outputs[(variant, batch, shape, envelope)] = result
        del model
        gc.collect()
        torch.cuda.empty_cache()

    comparisons = _compare_outputs(outputs)
    effect_rows = [
        {
            "logical_batch": int(row["logical_batch"]),
            "graph": bool(row["graph_enabled"]),
            "throughput": float(row["logical_samples_per_s"]),
        }
        for row in all_rows
    ]
    effects = summarize_comparisons(effect_rows)
    b2_ratios: list[float] = []
    for shape in SHAPES:
        for envelope in range(len(SEED_ENVELOPES)):
            off = next(
                row for row in all_rows
                if row["variant"] == "no_graph" and row["logical_batch"] == 2
                and row["shape"] == shape and row["envelope"] == envelope
            )
            on = next(
                row for row in all_rows
                if row["variant"] == "graph" and row["logical_batch"] == 2
                and row["shape"] == shape and row["envelope"] == envelope
            )
            b2_ratios.append(float(on["logical_samples_per_s"]) / float(off["logical_samples_per_s"]))
    ci_lower, ci_upper = _bootstrap_ci(b2_ratios)
    b2_graph_rows = [row for row in all_rows if row["variant"] == "graph" and row["logical_batch"] == 2]
    gate = evaluate_fixed_work_gate(
        {
            "correctness": all(row["correctness"] == "PASS" for row in all_rows)
            and all(item["status"] == "PASS" for item in comparisons),
            "b2_graph_steps": min((int(row["graph_replay_calls"]) for row in b2_graph_rows), default=0),
            "b2_graph_eligible": min((int(row["graph_eligible_calls"]) for row in b2_graph_rows), default=0),
            "b2_graph_throughput_ratio": float(np.median(np.asarray(b2_ratios))),
            "b2_graph_ci_lower": ci_lower,
            "peak_memory_bytes": max((int(row["peak_allocated_bytes"]) for row in all_rows), default=0),
        }
    )

    summary: list[dict[str, Any]] = []
    for variant in ("no_graph", "graph"):
        for batch in (1, 2):
            for shape in SHAPES:
                selected = [
                    row for row in all_rows
                    if row["variant"] == variant and row["logical_batch"] == batch and row["shape"] == shape
                ]
                summary.append(
                    {
                        "variant": variant,
                        "logical_batch": batch,
                        "shape": shape,
                        "wall_ms": _median(row["wall_time_ms"] for row in selected),
                        "cuda_ms": _median(row["cuda_time_ms"] for row in selected),
                        "throughput": _median(row["logical_samples_per_s"] for row in selected),
                        "graph_replays": int(max((row["graph_replay_calls"] for row in selected), default=0)),
                        "fallbacks": int(max((row["graph_fallback_calls"] for row in selected), default=0)),
                        "peak_gib": max((int(row["peak_allocated_bytes"]) for row in selected), default=0) / 1024**3,
                    }
                )

    try:
        import subprocess

        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        source_commit = "unknown"
    payload = {
        "schema_version": "graph-b2-fixed-work-v1",
        "source_commit": source_commit,
        "model_path": str(args.model_path),
        "prompt": prompt,
        "shape_lengths": dict(zip(SHAPES, shape_lengths)),
        "graph_chunk_sizes": dict(zip(SHAPES, derived_graph_chunks or ())),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "n_timesteps": N_TIMESTEPS,
        "graph_max_cache": os.environ.get("LYCHEEFD_FLOW_CUDA_GRAPH_MAX_CACHE", "default(500/1000)"),
        "capture_info": capture_info,
        "rows": all_rows,
        "summary": summary,
        "comparisons": comparisons,
        "effects": effects,
        "b2_ratio_samples": b2_ratios,
        "b2_ratio_bootstrap_ci95": [ci_lower, ci_upper],
        "gate": gate,
    }
    json_path = out_root / "GRAPH_B2_FIXED_WORK_RAW.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_csv(out_root / "GRAPH_B2_FIXED_WORK_METRICS.csv", all_rows)
    _write_report(out_root / "GRAPH_B2_FIXED_WORK_REPORT.md", payload)
    print(json.dumps({"status": gate["status"], "output": str(json_path), "gate": gate}, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--shape-lengths", default=",".join(map(str, DEFAULT_SHAPE_LENGTHS)))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile-kernels", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
