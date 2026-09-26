"""Bounded real-vLLM model service probe for the row-aware execution audit.

This is an analysis-only entry point.  It submits a small number of explicit
row-aware requests to the existing Lychee engine and measures one prefill plus
decode steps.  It does not change the serving runtime or public protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


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
        text_sampling={"do_sample": False, "temperature": 1.0,
                       "top_k": 0, "top_p": 1.0},
        stoken_sampling={"do_sample": False, "temperature": 1.0,
                         "top_k": 0, "top_p": 1.0},
        control_sampling={"do_sample": False, "temperature": 1.0,
                          "top_k": 0, "top_p": 1.0},
    )


def main() -> int:
    args = _parse_args()
    root = str(Path(args.root).resolve())
    sys.path.insert(0, root)
    sys.path.insert(0, str(Path(root) / "third_party" / "vllm"))
    sys.path.insert(0, str(Path(root) / "third_party" / "Step-Audio2"))

    from lychee_fd.vllm_integration.engine import LycheeVLLMEngine
    from vllm import SamplingParams

    batch_size = int(args.batch_size)
    decode_steps = int(args.decode_steps)
    engine = LycheeVLLMEngine(
        model_path=args.model,
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
    params = SamplingParams(
        max_tokens=decode_steps,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    request_ids = tuple(f"probe-{index}" for index in range(batch_size))
    for request_id in request_ids:
        engine.engine.add_request(
            request_id=request_id,
            inputs={"prompt_token_ids": [11]},
            params=params,
            multihead_request_state=_state(request_id),
        )

    rows = []
    step = 0
    started_ns = time.perf_counter_ns()
    while engine.engine.has_unfinished_requests() and step < decode_steps + 1:
        step_start = time.perf_counter_ns()
        outputs = engine.engine.step()
        # The engine's output path may enqueue CUDA work.  Synchronize only in
        # this standalone probe so service-time measurements are comparable.
        torch.cuda.synchronize()
        rows.append({
            "step": step,
            "batch_size": len(outputs),
            "request_ids": [str(getattr(item, "request_id", ""))
                            for item in outputs],
            "elapsed_ns": time.perf_counter_ns() - step_start,
        })
        step += 1

    payload = {
        "schema": "lychee-model-service-probe-v1",
        "batch_size_requested": batch_size,
        "decode_steps_requested": decode_steps,
        "request_ids": list(request_ids),
        "steps": rows,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "batch_size_requested": batch_size,
        "steps": len(rows),
        "observed_batch_sizes": [row["batch_size"] for row in rows],
        "elapsed_s": payload["elapsed_ns"] / 1e9,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
