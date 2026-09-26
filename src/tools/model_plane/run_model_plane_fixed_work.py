#!/usr/bin/env python3
"""Deterministic fixed-work probe for the row-aware model execution plane.

This is an analysis-only runner.  It loads the real Lychee vLLM model inside
the approved container, uses greedy multi-head sampling, and compares either
two serialized logical requests or two requests dispatched together through
the row-aware driver's actual batch callback.  It intentionally does not
touch the public realtime protocol or the acoustic Flow implementation.

The ``acoustic_batch_label`` is retained in the output so the same model
workload can be joined with a frozen acoustic B=1/B=2 envelope in later
analysis.  No acoustic work is fabricated by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("serial", "row_batch", "row_single"), required=True)
    parser.add_argument("--shape", choices=("p10", "p50", "p90"), required=True)
    parser.add_argument("--acoustic-batch-label", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _shape_prompt(shape: str) -> list[int]:
    # Fixed prompt lengths stand in for P10/P50/P90 model input envelopes.
    # The output explicitly records these as shape proxies; no claim is made
    # that they reproduce a particular acoustic tensor shape.
    lengths = {"p10": 8, "p50": 32, "p90": 96}
    length = lengths[shape]
    return [11 + (index % 17) for index in range(length)]


def _state(request_id: str):
    from vllm.sequence import MultiHeadRequestState

    return MultiHeadRequestState(
        text_input_ids=[11],
        stoken_input_ids=[152418],
        control_input_ids=[153228],
        session_id=request_id,
        phase="speaking",
        row_aware_enabled=True,
        keep_alive=False,
        text_sampling={"do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
        stoken_sampling={"do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
        control_sampling={"do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
    )


def _output_tokens(output: Any) -> tuple[int, ...]:
    values = []
    completions = getattr(output, "outputs", None) or ()
    if completions:
        token_ids = getattr(completions[0], "token_ids", None) or ()
        # vLLM may expose the cumulative completion prefix on every output.
        # The fixed-work contract compares the newly emitted token per engine
        # turn, matching the online fingerprint definition.
        if token_ids:
            values.append(int(token_ids[-1]))
    return tuple(values)


def _output_id(output: Any) -> str:
    return str(getattr(output, "request_id", ""))


def _digest(values: list[tuple[str, tuple[int, ...]]]) -> str:
    h = hashlib.sha256()
    for request_id, tokens in values:
        h.update(request_id.encode("utf-8"))
        h.update(b"\0")
        for token in tokens:
            h.update(int(token).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def _token_only_digest(values: list[tuple[str, tuple[int, ...]]]) -> str:
    h = hashlib.sha256()
    for _, tokens in values:
        for token in tokens:
            h.update(int(token).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def _make_engine(root: str, model: str, row_driver: bool):
    if row_driver:
        os.environ["LYCHEEFD_ROW_AWARE_MODEL_EXECUTION_PLANE"] = "1"
    else:
        os.environ.pop("LYCHEEFD_ROW_AWARE_MODEL_EXECUTION_PLANE", None)
    from lychee_fd.vllm_integration.engine import LycheeVLLMEngine

    return LycheeVLLMEngine(
        model_path=model,
        device="cuda",
        dtype="bfloat16",
        gpu_memory_utilization=0.70,
        max_model_len=8192,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        disable_log_stats=True,
    )


def _add(engine, request_id: str, prompt: list[int], params, state) -> None:
    add = getattr(engine, "_add_request_compat_direct")
    add(request_id, prompt, params, multihead_request_state=state)


def _run_serial(engine, request_ids: tuple[str, str], prompt: list[int], params, decode_steps: int):
    output_tokens: list[tuple[str, tuple[int, ...]]] = []
    physical_batch_sizes: list[int] = []
    for request_id in request_ids:
        _add(engine, request_id, prompt, params, _state(request_id))
        produced = 0
        while engine.engine.has_unfinished_requests() and produced < decode_steps:
            outputs = list(engine.engine.step() or ())
            physical_batch_sizes.append(len(outputs))
            for output in outputs:
                if _output_id(output) == request_id:
                    output_tokens.append((request_id, _output_tokens(output)))
                    produced += 1
        if produced != decode_steps:
            raise RuntimeError(f"serialized request {request_id} produced {produced}/{decode_steps} steps")
    return output_tokens, physical_batch_sizes


def _run_row(engine, request_ids: tuple[str, str], prompt: list[int], params, decode_steps: int, batch: int):
    plane = engine._row_model_execution_plane
    if plane is None:
        raise RuntimeError("row_batch mode requires the row-aware execution plane")
    from lychee_fd.runtime.row_aware_model_execution_plane import ModelRound

    output_tokens: list[tuple[str, tuple[int, ...]]] = []
    physical_batch_sizes: list[int] = []
    request_groups = (request_ids,) if batch == 2 else tuple((item,) for item in request_ids)
    for group in request_groups:
        for request_id in group:
            state = _state(request_id)
            engine._row_request_states[request_id] = state
            engine._row_model_generation[request_id] = 0
            plane.register({"request_id": request_id, "generation_id": 0})
            # The add operation shares the driver's engine thread with step().
            plane.call_on_driver(
                lambda request_id=request_id, state=state: _add(
                    engine, request_id, prompt, params, state
                ),
                description="fixed_work_add_request",
            ).result(timeout=30)
        for step in range(decode_steps):
            rounds = tuple(
                ModelRound(request_id, 0, step, {"fixed_work": True})
                for request_id in group
            )
            outputs = engine._run_on_row_model_driver(
                lambda rounds=rounds: engine._execute_row_model_batch(rounds),
                description="fixed_work_model_step",
            )
            physical_batch_sizes.append(len(outputs))
            for output in outputs:
                output_tokens.append((_output_id(output), _output_tokens(output)))

    # The engine requests normally finish at the requested max_tokens.  Make
    # the logical maps disposable even if a future vLLM version keeps a row
    # alive for one extra bookkeeping turn.
    for request_id in request_ids:
        engine._abort_request_compat_direct(request_id)
        engine._row_request_states.pop(request_id, None)
        engine._row_model_generation.pop(request_id, None)
        plane.finish(request_id, 0)
    return output_tokens, physical_batch_sizes


def main() -> int:
    args = _args()
    root = str(Path(args.root).resolve())
    for path in (root, str(Path(root) / "third_party" / "vllm"), str(Path(root) / "third_party" / "Step-Audio2")):
        if path not in sys.path:
            sys.path.insert(0, path)

    # The runtime integration expects these explicit library/backend choices
    # in the approved container; they are also recorded in the output.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        "/root/anaconda3/envs/sglang/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64",
    )
    random.seed(12345)
    try:
        import numpy as np
        np.random.seed(12345)
    except Exception:
        pass
    import torch
    torch.manual_seed(12345)

    row_driver = args.mode != "serial"
    engine = _make_engine(root, args.model, row_driver)
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=int(args.decode_steps),
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    prompt = _shape_prompt(args.shape)
    rows: list[dict[str, Any]] = []
    total = int(args.warmups) + int(args.repeats)
    try:
        for repeat in range(total):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            request_ids = (f"fixed-{args.mode}-{repeat}-0", f"fixed-{args.mode}-{repeat}-1")
            started = time.perf_counter_ns()
            if args.mode == "serial":
                outputs, batch_sizes = _run_serial(
                    engine, request_ids, prompt, params, int(args.decode_steps)
                )
            else:
                outputs, batch_sizes = _run_row(
                    engine,
                    request_ids,
                    prompt,
                    params,
                    int(args.decode_steps),
                    2 if args.mode == "row_batch" else 1,
                )
            torch.cuda.synchronize()
            elapsed_ns = time.perf_counter_ns() - started
            ordered = sorted(outputs, key=lambda item: item[0])
            rows.append(
                {
                    "repeat": repeat,
                    "warmup": repeat < int(args.warmups),
                    "mode": args.mode,
                    "shape": args.shape,
                    "shape_prompt_length": len(prompt),
                    "acoustic_batch_label": int(args.acoustic_batch_label),
                    "logical_request_count": 2,
                    "decode_steps": int(args.decode_steps),
                    "physical_batch_sizes": batch_sizes,
                    "physical_batch_distribution": dict(Counter(batch_sizes)),
                    "output_step_count": len(outputs),
                    "output_token_count": sum(len(tokens) for _, tokens in outputs),
                    "output_sha256": _digest(ordered),
                    "output_token_only_sha256": _token_only_digest(ordered),
                    "elapsed_ns": elapsed_ns,
                    "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
    finally:
        plane = getattr(engine, "_row_model_execution_plane", None)
        if plane is not None:
            plane.close()

    measured = [row for row in rows if not row["warmup"]]
    payload = {
        "schema": "apr-model-plane-fixed-work-v1",
        "mode": args.mode,
        "shape": args.shape,
        "shape_prompt_length": len(prompt),
        "acoustic_batch_label": int(args.acoustic_batch_label),
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "decode_steps": int(args.decode_steps),
        "rows": rows,
        "measured_elapsed_ns": [row["elapsed_ns"] for row in measured],
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": args.mode,
        "shape": args.shape,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "median_ms": sorted(payload["measured_elapsed_ns"])[len(measured) // 2] / 1e6 if measured else None,
        "batch_distribution": dict(Counter(size for row in measured for size in row["physical_batch_sizes"])),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
