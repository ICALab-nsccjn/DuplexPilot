"""Run the frozen mixed-step/current-shape-padded acoustic B=2 path.

This is a fixed-work A100 lane.  It compares two independent B=1 trajectories
with a mixed trajectory that repeatedly calls the exact public
``advance_chunk_step_variable_mixed_chunk_padding_batch`` API until the first
row reaches the terminal Euler step, then finishes the remaining row as a
singleton.  The three envelopes are deliberately small, real-token Flow
envelopes; no online arrival synchronization is inferred from this benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


SHAPES = {
    "p10": {"prior_lengths": (8, 12), "current_lengths": (12, 16)},
    "p50": {"prior_lengths": (12, 20), "current_lengths": (16, 24)},
    "p90": {"prior_lengths": (20, 32), "current_lengths": (24, 40)},
}


def _mixed_schedule_counts(
    start_steps: tuple[int, int], terminal_step: int = 10
) -> tuple[int, int]:
    """Return shared mixed calls and singleton tail calls.

    The mixed public API can only receive rows that still have a Flow step to
    execute.  When rows start at different Euler positions, they therefore
    share the prefix until the earlier row reaches ``terminal_step``; the
    remaining steps are executed as singleton calls.  Keeping this arithmetic
    in a pure helper makes the benchmark scope explicit and testable.
    """
    if len(start_steps) != 2:
        raise ValueError("the fixed-work B=2 schedule requires exactly two rows")
    if terminal_step < 0:
        raise ValueError("terminal_step must be non-negative")
    if any(step < 0 or step > terminal_step for step in start_steps):
        raise ValueError("start steps must lie within the terminal step")
    shared = min(terminal_step - step for step in start_steps)
    after_shared = tuple(step + shared for step in start_steps)
    singleton_tail = sum(terminal_step - step for step in after_shared)
    return shared, singleton_tail


def _configure() -> None:
    device = int(os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE", "1"))
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _max_abs(left, right) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _finish(model, state):
    result = model.finish_chunk_steps(state)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("public Token2Wav finish_chunk_steps contract changed")
    return result


def _run_prior(model, gate, prompt: str, length: int, request_id: str):
    stream = model.create_stream_state(prompt)
    state = model.begin_chunk_steps(
        gate._tokens(length), prompt, stream, last_chunk=False,
        n_timesteps=gate.N_TIMESTEPS, request_id=request_id,
        generation_id=3, sequence_no=0, version=0,
    )
    values = state
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(gate.N_TIMESTEPS):
            values = model.advance_chunk_step(values)
    _, cache = _finish(model, values)
    return gate._clone_tree(cache)


def _begin(model, gate, prompt: str, prior_cache, length: int, request_id: str, generation: int):
    stream = model.create_stream_state(prompt)
    stream["flow_cache"].update(gate._clone_tree(prior_cache))
    return model.begin_chunk_steps(
        gate._tokens(length), prompt, stream, last_chunk=False,
        n_timesteps=gate.N_TIMESTEPS, request_id=request_id,
        generation_id=generation, sequence_no=0, version=0,
    )


def _advance_to(model, state, count: int):
    value = state
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(count):
            value = model.advance_chunk_step(value)
    return value


def _complete(model, states):
    values = tuple(states)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(20):
            if all(value.step_index >= 10 for value in values):
                break
            values = tuple(
                model.advance_chunk_step(value)
                if value.step_index < 10 else value
                for value in values
            )
    if any(value.step_index < 10 for value in values):
        raise RuntimeError("fixed-work state did not reach terminal step")
    return values


def _complete_mixed(model, states, terminal_step: int = 10):
    """Complete a pair with mixed B=2 calls followed by a singleton tail."""
    values = tuple(states)
    shared_calls, _ = _mixed_schedule_counts(
        tuple(int(value.step_index) for value in values), terminal_step
    )
    singleton_tail_calls = 0
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(shared_calls):
            values = tuple(
                model.advance_chunk_step_variable_mixed_chunk_padding_batch(values)
            )
        while any(value.step_index < terminal_step for value in values):
            next_values = []
            for value in values:
                if value.step_index < terminal_step:
                    next_values.append(model.advance_chunk_step(value))
                    singleton_tail_calls += 1
                else:
                    next_values.append(value)
            values = tuple(next_values)
    if any(value.step_index < terminal_step for value in values):
        raise RuntimeError("mixed fixed-work state did not reach terminal step")
    return values, shared_calls, singleton_tail_calls


def _run_one(model, gate, shape: str, repeat: int) -> dict[str, object]:
    prompt = str(gate.PROMPT_WAV)
    spec = SHAPES[shape]
    first_cache = _run_prior(model, gate, prompt, spec["prior_lengths"][0], f"prior-{shape}-{repeat}-a")
    second_cache = _run_prior(model, gate, prompt, spec["prior_lengths"][1], f"prior-{shape}-{repeat}-b")
    first = _begin(model, gate, prompt, first_cache, spec["current_lengths"][0], f"fixed-{shape}-{repeat}-a", 41)
    second = _begin(model, gate, prompt, second_cache, spec["current_lengths"][1], f"fixed-{shape}-{repeat}-b", 73)
    first = _advance_to(model, first, 2)
    second = _advance_to(model, second, 4)
    independent_inputs = (gate._clone_state(first), gate._clone_state(second))
    combined_inputs = (gate._clone_state(first), gate._clone_state(second))

    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    independent = _complete(model, independent_inputs)
    torch.cuda.synchronize()
    independent_ms = (time.perf_counter_ns() - start) / 1e6

    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    combined, combined_batch_calls, combined_singleton_tail_calls = _complete_mixed(
        model, combined_inputs
    )
    torch.cuda.synchronize()
    combined_ms = (time.perf_counter_ns() - start) / 1e6

    transition = []
    for expected, actual in zip(independent, combined):
        transition.append({
            "request_id": actual.request_id,
            "step_index": int(actual.step_index),
            "x_max_abs_diff": _max_abs(actual.x, expected.x),
            "attention_cache_max_abs_diff": _max_abs(actual.completed_att_cache, expected.completed_att_cache),
            "cache_length": int(actual.completed_att_cache.shape[3]),
        })
    final = []
    for expected, actual in zip(independent, combined):
        expected_mel, _ = _finish(model, expected)
        actual_mel, _ = _finish(model, actual)
        expected_pcm = gate._pcm_from_mel(model, prompt, expected_mel, last_chunk=False)
        actual_pcm = gate._pcm_from_mel(model, prompt, actual_mel, last_chunk=False)
        pcm = gate._pcm_metrics(expected_pcm, actual_pcm)
        final.append({
            "request_id": actual.request_id,
            "mel_max_abs_diff": _max_abs(actual_mel, expected_mel),
            "pcm": pcm,
            "sample_count": int(len(actual_pcm) // 2),
        })
    return {
        "shape": shape,
        "repeat": repeat,
        "prior_lengths": list(spec["prior_lengths"]),
        "current_lengths": list(spec["current_lengths"]),
        "start_step_indices": [2, 4],
        "measured_scope": "all_remaining_steps_with_mixed_prefix_and_singleton_tail",
        "independent_singleton_calls": sum(10 - step for step in (2, 4)),
        "combined_batch_calls": combined_batch_calls,
        "combined_singleton_tail_calls": combined_singleton_tail_calls,
        "independent_two_b1_wall_ms": independent_ms,
        "mixed_chunk_padding_b2_wall_ms": combined_ms,
        "speedup_independent_over_combined": independent_ms / combined_ms if combined_ms else None,
        "transition_rows": transition,
        "final_rows": final,
        "pcm_contract_pass": all(
            bool(row["pcm"].get("same_length"))
            and float(row["pcm"].get("normalized_rmse", float("inf"))) <= 0.02
            and float(row["pcm"].get("correlation", -1.0)) >= 0.99
            and float(row["pcm"].get("snr_db", float("-inf"))) >= 34.0
            for row in final
        ),
        "gpu_memory": [
            {
                "device": index,
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        raise SystemExit("warmups must be non-negative and repeats must be positive")
    _configure()
    tests_dir = Path(__file__).resolve().parents[2] / "tests"
    sys.path.insert(0, str(tests_dir))
    import test_flow_batch_a100_equivalence as gate
    from token2wav import Token2wav

    model = Token2wav(str(gate.MODEL_PATH), float16=False)
    records: list[dict[str, object]] = []
    try:
        for shape in SHAPES:
            for index in range(args.warmups + args.repeats):
                torch.cuda.reset_peak_memory_stats()
                row = _run_one(model, gate, shape, index + 1)
                row["warmup"] = index < args.warmups
                records.append(row)
                print(json.dumps({
                    "shape": shape,
                    "repeat": index + 1,
                    "warmup": index < args.warmups,
                    "independent_ms": row["independent_two_b1_wall_ms"],
                    "combined_ms": row["mixed_chunk_padding_b2_wall_ms"],
                    "pcm_contract_pass": row["pcm_contract_pass"],
                }, sort_keys=True), flush=True)
    finally:
        del model
        torch.cuda.empty_cache()
    payload = {
        "schema": "apr-joint-acoustic-fixed-work-v1",
        "configuration": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "dtype": "float32",
            "flow_steps": 10,
            "method": "advance_chunk_step_variable_mixed_chunk_padding_batch",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "token2wav_device": os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE"),
            "flow_attention_cache_capacity": os.environ.get("LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY"),
            "model": str(gate.MODEL_PATH),
            "prompt": str(gate.PROMPT_WAV),
        },
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
