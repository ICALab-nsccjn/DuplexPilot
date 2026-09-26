#!/usr/bin/env python3
"""Validate the opt-in mixed-step variable Flow path at logical B=4.

This is a mechanism/equivalence experiment only.  It compares four
independent B=1 transitions with one explicit B=4-capable transition while
keeping the current chunk shape and timestep schedule equal and varying only
the per-request Euler position and attention-cache history.
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


def _clone_state(state):
    from dataclasses import replace

    values = {}
    for key, value in state.__dict__.items():
        if isinstance(value, torch.Tensor):
            values[key] = value.detach().clone()
        elif isinstance(value, tuple):
            values[key] = tuple(
                item.detach().clone() if isinstance(item, torch.Tensor) else item
                for item in value
            )
    return replace(
        state,
        **{
            key: value
            for key, value in values.items()
            if key in state.__dataclass_fields__
        },
    )


def _run_one(model, gate, repeat: int) -> dict[str, object]:
    prompt = str(gate.PROMPT_WAV)
    start_steps = (0, 2, 4, 6)
    prior_lengths = (12, 16, 20, 24)
    states = []
    for row, (prior_length, start_step) in enumerate(
        zip(prior_lengths, start_steps)
    ):
        prior_cache = gate._run_prior_chunk(
            model, prompt, prior_length, f"b4-run-{repeat}-prior-{row}"
        )
        stream = model.create_stream_state(prompt)
        stream["flow_cache"].update(gate._clone_tree(prior_cache))
        state = model.begin_chunk_steps(
            gate._tokens(16),
            prompt,
            stream,
            last_chunk=False,
            n_timesteps=gate.N_TIMESTEPS,
            request_id=f"b4-{repeat}-{row}",
            generation_id=100 + row,
            sequence_no=3,
            version=11,
        )
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
            for _ in range(start_step):
                state = model.advance_chunk_step(state)
        states.append(state)

    start_cache_lengths = [
        int(state.input_att_cache[state.step_index].shape[3]) for state in states
    ]
    independent = tuple(_clone_state(state) for state in states)
    mixed_inputs = tuple(_clone_state(state) for state in states)

    torch.cuda.synchronize()
    independent_start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        independent_next = tuple(
            model.advance_chunk_step(state) for state in independent
        )
    torch.cuda.synchronize()
    independent_ms = (time.perf_counter() - independent_start) * 1000.0

    torch.cuda.synchronize()
    mixed_start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        mixed_next = tuple(
            model.advance_chunk_step_variable_mixed_batch_b4(mixed_inputs)
        )
    torch.cuda.synchronize()
    mixed_ms = (time.perf_counter() - mixed_start) * 1000.0

    transition_rows = []
    for expected, actual in zip(independent_next, mixed_next):
        transition_rows.append(
            {
                "request_id": actual.request_id,
                "start_step_index": int(expected.step_index - 1),
                "next_step_index": int(actual.step_index),
                "x_max_abs_diff": _max_abs(actual.x, expected.x),
                "attention_cache_max_abs_diff": _max_abs(
                    actual.completed_att_cache, expected.completed_att_cache
                ),
                "cache_length": int(actual.completed_att_cache.shape[3]),
            }
        )

    # Continue both branches with the frozen B=1 operation.  This verifies
    # that the B=4 transition leaves four independently resumable states.
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        while any(state.step_index < gate.N_TIMESTEPS for state in independent_next):
            independent_next = tuple(
                model.advance_chunk_step(state)
                if state.step_index < gate.N_TIMESTEPS
                else state
                for state in independent_next
            )
        while any(state.step_index < gate.N_TIMESTEPS for state in mixed_next):
            mixed_next = tuple(
                model.advance_chunk_step(state)
                if state.step_index < gate.N_TIMESTEPS
                else state
                for state in mixed_next
            )

    final_rows = []
    for expected, actual in zip(independent_next, mixed_next):
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
        "initial_attention_cache_lengths": start_cache_lengths,
        "independent_four_b1_wall_ms": independent_ms,
        "mixed_b4_wall_ms": mixed_ms,
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
            "logical_batch_size": 4,
            "dtype": "float32",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "token2wav_device": os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE"),
            "flow_attention_cache_capacity": os.environ.get(
                "LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY"
            ),
            "model": str(gate.MODEL_PATH),
            "prompt": str(gate.PROMPT_WAV),
            "mixed_step_indices": [0, 2, 4, 6],
        },
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    del model
    torch.cuda.empty_cache()
    print(json.dumps({"out": str(out_path), "repeats": args.repeats}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
