"""Repeat the real A100 B=1 and unequal-attention-cache B=2 gates.

This benchmark is opt-in and intentionally uses the same helpers and model
configuration as ``test_flow_batch_a100_equivalence.py``.  It emits numeric
evidence instead of relying only on pytest's pass/fail result.
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
    torch.cuda.set_device(int(os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE", "1")))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _max_abs(left, right) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _run_one(model, gate):
    prompt_wav = str(gate.PROMPT_WAV)
    torch.cuda.synchronize()
    b1_start = time.perf_counter()
    frozen_mel, frozen_cache = gate._frozen_control(model, prompt_wav)
    state = gate._begin(model, prompt_wav, "metric-variable-single")
    variable_states, variable_finished = gate._finish_variable_single(model, state)
    variable_mel, variable_cache = variable_finished[0]
    torch.cuda.synchronize()
    b1_wall_ms = (time.perf_counter() - b1_start) * 1000.0
    b1_pcm = gate._pcm_from_mel(model, prompt_wav, variable_mel, last_chunk=False)
    b1_control_stream = model.create_stream_state(prompt_wav)
    b1_control_pcm = model.stream_with_state(
        gate._tokens(), prompt_wav, b1_control_stream, last_chunk=False
    )
    b1_pcm_metrics = gate._pcm_metrics(b1_control_pcm, b1_pcm)

    torch.cuda.synchronize()
    b2_start = time.perf_counter()
    first_cache = gate._run_prior_chunk(model, prompt_wav, 12, "metric-prior-a")
    second_cache = gate._run_prior_chunk(model, prompt_wav, 20, "metric-prior-b")
    first_stream = model.create_stream_state(prompt_wav)
    second_stream = model.create_stream_state(prompt_wav)
    first_stream["flow_cache"].update(gate._clone_tree(first_cache))
    second_stream["flow_cache"].update(gate._clone_tree(second_cache))
    first = model.begin_chunk_steps(
        gate._tokens(16), prompt_wav, first_stream, last_chunk=False,
        n_timesteps=gate.N_TIMESTEPS, request_id="metric-var-a", generation_id=11,
    )
    second = model.begin_chunk_steps(
        gate._tokens(16), prompt_wav, second_stream, last_chunk=False,
        n_timesteps=gate.N_TIMESTEPS, request_id="metric-var-b", generation_id=29,
    )
    initial_lengths = [
        int(first.input_att_cache[0].shape[3]),
        int(second.input_att_cache[0].shape[3]),
    ]
    independent = (gate._clone_state(first), gate._clone_state(second))
    variable = (first, second)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(gate.N_TIMESTEPS):
            independent = tuple(model.advance_chunk_step(value) for value in independent)
            variable = tuple(model.advance_chunk_step_variable_batch(variable))
    torch.cuda.synchronize()
    b2_wall_ms = (time.perf_counter() - b2_start) * 1000.0
    b2_rows = []
    for expected, actual in zip(independent, variable):
        expected_mel, _ = model.finish_chunk_steps(expected)
        actual_mel, _ = model.finish_chunk_steps(actual)
        expected_pcm = gate._pcm_from_mel(
            model, prompt_wav, expected_mel, last_chunk=False
        )
        actual_pcm = gate._pcm_from_mel(
            model, prompt_wav, actual_mel, last_chunk=False
        )
        pcm_metrics = gate._pcm_metrics(expected_pcm, actual_pcm)
        b2_rows.append(
            {
                "request_id": actual.request_id,
                "initial_attention_cache_length": initial_lengths[len(b2_rows)],
                "final_attention_cache_length": int(actual.completed_att_cache.shape[3]),
                "x_max_abs_diff": _max_abs(actual.x, expected.x),
                "attention_cache_max_abs_diff": _max_abs(
                    actual.completed_att_cache, expected.completed_att_cache
                ),
                "pcm": pcm_metrics,
            }
        )

    allocated = []
    for index in range(torch.cuda.device_count()):
        allocated.append(
            {
                "device": index,
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    return {
        "b1": {
            "wall_ms": b1_wall_ms,
            "mel_max_abs_diff": _max_abs(variable_mel, frozen_mel),
            "pcm": b1_pcm_metrics,
            "step_index": int(variable_states[0].step_index),
        },
        "b2": {"wall_ms": b2_wall_ms, "rows": b2_rows},
        "gpu_memory": allocated,
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
        record = _run_one(model, gate)
        record["repeat"] = repeat
        records.append(record)
    output = {
        "configuration": {
            "repeats": args.repeats,
            "dtype": "float32",
            "tf32": False,
            "matmul_precision": "highest",
            "deterministic_algorithms": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "flow_attention_cache_capacity": os.environ.get(
                "LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY"
            ),
            "model": str(gate.MODEL_PATH),
            "prompt": str(gate.PROMPT_WAV),
        },
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n")
    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
