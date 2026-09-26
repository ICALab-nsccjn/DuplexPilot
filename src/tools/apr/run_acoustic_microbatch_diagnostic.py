"""Diagnostic real-local Flow micro-batch benchmark.

This script calls the existing Flow.inference_chunk directly through the public
diagnostic adapter. It does not enable APR or modify the production acoustic
serving path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_tokens(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tokens")
    if not isinstance(payload, list) or not payload:
        raise ValueError("token asset must contain a non-empty list")
    if any(not isinstance(token, int) or isinstance(token, bool) for token in payload):
        raise ValueError("token asset values must be integers")
    return payload


def _ensure_repo_root_importable() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return repo_root


def _equivalence_session_count(logical_sessions: int) -> int:
    if not isinstance(logical_sessions, int) or logical_sessions <= 0:
        raise ValueError("logical_sessions must be a positive integer")
    return min(2, logical_sessions)


def _write_invalid_output(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    failure_signature = f"{exc.__class__.__name__}: {exc}"
    payload = {
        "diagnostic_only": True,
        "status": "INVALID",
        "failure_signature": failure_signature,
        "error": failure_signature,
        "model_path": getattr(args, "model_path", None),
        "prompt_wav": getattr(args, "prompt_wav", None),
        "logical_sessions": getattr(args, "logical_sessions", None),
        "repeats": getattr(args, "repeats", None),
        "measurements": [],
        "equivalence": {
            "state_integrity": False,
            "output_order": False,
            "cache_equivalence": False,
            "numerical_equivalence": False,
            "mismatches": (failure_signature,),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


class RealFlowBackend:
    def __init__(self, model: Any, *, float16: bool):
        self.model = model
        self.float16 = float16

    def execute(self, packed_step):
        import torch

        with torch.amp.autocast(
            "cuda", dtype=torch.float16 if self.float16 else torch.float32
        ):
            return self.model.flow.inference_chunk(
                token=packed_step.tokens,
                spk=packed_step.speaker,
                cache=packed_step.flow_cache,
                last_chunk=packed_step.last_chunk,
                n_timesteps=packed_step.n_timesteps,
            )


def _make_steps(model: Any, prompt_wav: str, tokens: list[int], count: int):
    import torch

    from profiling.token2wav_flow_profile.contracts import FlowStep

    state_steps = []
    model.create_stream_state(prompt_wav)
    speaker = model.cache[prompt_wav][2]
    for index in range(count):
        stream_state = model.create_stream_state(prompt_wav)
        state_steps.append(
            FlowStep(
                request_id=f"microbatch-request-{index}",
                generation_id=0,
                sequence_no=0,
                tokens=torch.tensor([tokens], dtype=torch.int32, device="cuda"),
                speaker=speaker.clone(),
                flow_cache={
                    key: value.clone()
                    for key, value in stream_state["flow_cache"].items()
                },
                last_chunk=False,
                n_timesteps=10,
                model_identity="token2wav-flow-real-local",
            )
        )
    return tuple(state_steps)


def _measure(model, prompt_wav, tokens, logical_sessions, batch_size, float16):
    import torch

    from profiling.token2wav_flow_profile.batching import (
        AcousticBatchScheduler,
        FlowStateBatchAdapter,
    )
    from profiling.token2wav_flow_profile.layout import real_flow_cache_layout

    steps = _make_steps(model, prompt_wav, tokens, logical_sessions)
    backend = RealFlowBackend(model, float16=float16)
    adapter = FlowStateBatchAdapter(layout_spec=real_flow_cache_layout())
    scheduler = AcousticBatchScheduler(
        backend=backend, max_batch_size=batch_size, adapter=adapter
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    results = scheduler.submit(steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if any(result.status != "COMMITTED" for result in results):
        raise RuntimeError(f"invalid micro-batch attempt: {results!r}")
    return {
        "logical_sessions": logical_sessions,
        "configured_batch_size": batch_size,
        "physical_attempts": len(results),
        "wall_time_ms": elapsed * 1000.0,
        "per_session_ms": elapsed * 1000.0 / logical_sessions,
        "trace": list(scheduler.trace()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _ensure_repo_root_importable()
    sys.path.insert(0, str(repo_root / "third_party" / "Step-Audio2"))
    from token2wav import Token2wav
    from profiling.token2wav_flow_profile.equivalence import FlowBatchEquivalenceRunner
    from profiling.token2wav_flow_profile.layout import real_flow_cache_layout
    from profiling.token2wav_flow_profile.batching import FlowStateBatchAdapter

    model = Token2wav(args.model_path, float16=args.float16)
    prompt_wav = str(Path(args.prompt_wav))
    tokens = _load_tokens(Path(args.tokens))
    if args.logical_sessions <= 0 or args.logical_sessions > args.max_sessions:
        raise ValueError("logical_sessions exceeds diagnostic bound")
    if args.repeats <= 0 or args.repeats > args.max_repeats:
        raise ValueError("repeats exceeds diagnostic bound")

    backend = RealFlowBackend(model, float16=args.float16)
    adapter = FlowStateBatchAdapter(layout_spec=real_flow_cache_layout())
    equivalence = FlowBatchEquivalenceRunner(adapter=adapter).compare(
        _make_steps(
            model,
            prompt_wav,
            tokens,
            _equivalence_session_count(args.logical_sessions),
        ),
        backend,
    )
    measurements = []
    for repeat in range(args.repeats):
        for batch_size in (1, 2, 4):
            if batch_size > args.logical_sessions:
                continue
            item = _measure(
                model,
                prompt_wav,
                tokens,
                args.logical_sessions,
                batch_size,
                args.float16,
            )
            item["repeat"] = repeat
            measurements.append(item)
    output = {
        "diagnostic_only": True,
        "model_path": args.model_path,
        "prompt_wav": prompt_wav,
        "logical_sessions": args.logical_sessions,
        "repeats": args.repeats,
        "cuda_attribution": "available",
        "gpu_utilization": "not_collected_by_this_runner",
        "equivalence": {
            "state_integrity": equivalence.state_integrity,
            "output_order": equivalence.output_order,
            "cache_equivalence": equivalence.cache_equivalence,
            "numerical_equivalence": equivalence.numerical_equivalence,
            "mismatches": list(equivalence.mismatches),
        },
        "measurements": measurements,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--logical-sessions", type=int, default=2)
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-repeats", type=int, default=3)
    parser.add_argument("--float16", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = run(args)
    except Exception as exc:
        _write_invalid_output(args, exc)
        print(f"INVALID_MICROBATCH_DIAGNOSTIC: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["equivalence"]["numerical_equivalence"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
