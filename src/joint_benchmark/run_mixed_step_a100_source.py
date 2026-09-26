"""Measure real A100 mixed-step B=2 equivalence against independent B=1.

The runner is deliberately separate from the frozen exact-shape and same-step
benchmarks.  It exercises two real states at Euler positions 2 and 4 and
records both the one-transition comparison and the final PCM contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


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


def _run_one(model, gate, repeat: int) -> dict[str, object]:
    prompt = str(gate.PROMPT_WAV)
    first_cache = gate._run_prior_chunk(model, prompt, 12, f"mixed-run-{repeat}-a")
    second_cache = gate._run_prior_chunk(model, prompt, 20, f"mixed-run-{repeat}-b")
    first_stream = model.create_stream_state(prompt)
    second_stream = model.create_stream_state(prompt)
    first_stream["flow_cache"].update(gate._clone_tree(first_cache))
    second_stream["flow_cache"].update(gate._clone_tree(second_cache))
    first = model.begin_chunk_steps(
        gate._tokens(16), prompt, first_stream, last_chunk=False,
        n_timesteps=gate.N_TIMESTEPS, request_id=f"mixed-{repeat}-a", generation_id=41,
    )
    second = model.begin_chunk_steps(
        gate._tokens(16), prompt, second_stream, last_chunk=False,
        n_timesteps=gate.N_TIMESTEPS, request_id=f"mixed-{repeat}-b", generation_id=73,
    )
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(2):
            first = model.advance_chunk_step(first)
        for _ in range(4):
            second = model.advance_chunk_step(second)

    start_steps = (first.step_index, second.step_index)
    initial_lengths = (
        int(first.input_att_cache[first.step_index].shape[3]),
        int(second.input_att_cache[second.step_index].shape[3]),
    )
    independent = (gate._clone_state(first), gate._clone_state(second))
    mixed_inputs = (gate._clone_state(first), gate._clone_state(second))
    torch.cuda.synchronize()
    independent_start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        independent = tuple(model.advance_chunk_step(value) for value in independent)
    torch.cuda.synchronize()
    independent_ms = (time.perf_counter() - independent_start) * 1000.0

    torch.cuda.synchronize()
    mixed_start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        mixed = tuple(model.advance_chunk_step_variable_mixed_batch(mixed_inputs))
    torch.cuda.synchronize()
    mixed_ms = (time.perf_counter() - mixed_start) * 1000.0

    transition_rows = []
    for expected, actual in zip(independent, mixed):
        transition_rows.append(
            {
                "request_id": actual.request_id,
                "step_index": int(actual.step_index),
                "x_max_abs_diff": _max_abs(actual.x, expected.x),
                "attention_cache_max_abs_diff": _max_abs(
                    actual.completed_att_cache, expected.completed_att_cache
                ),
                "cache_length": int(actual.completed_att_cache.shape[3]),
            }
        )

    # Continue both branches with the frozen B=1 operation.  This checks that
    # a mixed transition leaves a valid resumable state, not just a matching
    # one-step tensor.
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        while any(value.step_index < gate.N_TIMESTEPS for value in independent):
            independent = tuple(
                model.advance_chunk_step(value)
                if value.step_index < gate.N_TIMESTEPS else value
                for value in independent
            )
        while any(value.step_index < gate.N_TIMESTEPS for value in mixed):
            mixed = tuple(
                model.advance_chunk_step(value)
                if value.step_index < gate.N_TIMESTEPS else value
                for value in mixed
            )

    final_rows = []
    for expected, actual in zip(independent, mixed):
        expected_mel, _ = model.finish_chunk_steps(expected)
        actual_mel, _ = model.finish_chunk_steps(actual)
        expected_pcm = gate._pcm_from_mel(model, prompt, expected_mel, last_chunk=False)
        actual_pcm = gate._pcm_from_mel(model, prompt, actual_mel, last_chunk=False)
        pcm = gate._pcm_metrics(expected_pcm, actual_pcm)
        final_rows.append(
            {
                "request_id": actual.request_id,
                "final_step_index": int(actual.step_index),
                "mel_max_abs_diff": _max_abs(actual_mel, expected_mel),
                "pcm": pcm,
                "sample_count": int(len(actual_pcm) // 2),
            }
        )

    return {
        "repeat": repeat,
        "start_step_indices": list(start_steps),
        "initial_attention_cache_lengths": list(initial_lengths),
        "independent_two_b1_wall_ms": independent_ms,
        "mixed_b2_wall_ms": mixed_ms,
        "transition_rows": transition_rows,
        "final_rows": final_rows,
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
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    _configure()
    tests_dir = Path(__file__).resolve().parents[2] / "tests"
    sys.path.insert(0, str(tests_dir))
    import test_flow_batch_a100_equivalence as gate
    from token2wav import Token2wav

    model = Token2wav(str(gate.MODEL_PATH), float16=False)
    records = []
    for repeat in range(1, args.repeats + 1):
        torch.cuda.reset_peak_memory_stats()
        records.append(_run_one(model, gate, repeat))
    output = {
        "configuration": {
            "repeats": args.repeats,
            "dtype": "float32",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "token2wav_device": os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE"),
            "flow_attention_cache_capacity": os.environ.get(
                "LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY"
            ),
            "model": str(gate.MODEL_PATH),
            "prompt": str(gate.PROMPT_WAV),
            "mixed_step_indices": [2, 4],
        },
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n")
    del model
    torch.cuda.empty_cache()
    print(json.dumps({"out": str(out_path), "repeats": args.repeats}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
