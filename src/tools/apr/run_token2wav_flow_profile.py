"""Run a bounded real-local Token2Wav Flow operator profile.

This runner is diagnostic-only. It instruments the existing local model call
without enabling APR or changing any serving path.
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
        raise ValueError("token asset must contain a non-empty list of tokens")
    if any(not isinstance(token, int) or isinstance(token, bool) for token in payload):
        raise ValueError("token asset values must be integers")
    return payload


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "third_party" / "Step-Audio2"))
    import torch

    from profiling.token2wav_flow_profile.profiler import FlowProfiler
    from tools.apr.analyze_token2wav_flow_profile import generate_flow_reports
    from token2wav import Token2wav

    model_path = Path(args.model_path)
    prompt_wav = Path(args.prompt_wav)
    token_asset = Path(args.tokens)
    output_dir = Path(args.output_dir)
    if not model_path.is_dir():
        raise ValueError(f"model path does not exist: {model_path}")
    if not prompt_wav.is_file():
        raise ValueError(f"prompt wav does not exist: {prompt_wav}")
    tokens = _load_tokens(token_asset)
    if args.steps <= 0 or args.steps > args.max_steps:
        raise ValueError("steps must be within the bounded diagnostic limit")
    output_dir.mkdir(parents=True, exist_ok=True)

    profiler = FlowProfiler(
        enabled=True,
        output_path=output_dir / "flow_operator_profile.jsonl",
    )
    model = Token2wav(str(model_path), float16=args.float16)
    active = {"value": False}
    original_inference_chunk = model.flow.inference_chunk

    def profiled_inference_chunk(*call_args: Any, **call_kwargs: Any):
        if not active["value"]:
            return original_inference_chunk(*call_args, **call_kwargs)
        token = call_kwargs.get("token", call_args[0] if call_args else None)
        speaker = call_kwargs.get("spk", call_args[1] if len(call_args) > 1 else None)
        cache = call_kwargs.get("cache", call_args[2] if len(call_args) > 2 else None)
        if token is None or speaker is None or not isinstance(cache, dict):
            raise ValueError("Flow profiling wrapper could not identify call tensors")
        cache_tensors = tuple(value for value in cache.values() if isinstance(value, torch.Tensor))
        shapes = {
            "input_shapes": (_shape(token), _shape(speaker)) + tuple(_shape(value) for value in cache_tensors),
            "output_shapes": (),
        }
        identity = {
            "request_id": args.request_id,
            "generation_id": args.generation_id,
            "batch_signature": (
                f"token={_shape(token)}:{token.dtype}:{token.device};"
                f"speaker={_shape(speaker)}:{speaker.dtype}:{speaker.device}"
            ),
        }
        return profiler.profile_call(
            lambda: original_inference_chunk(*call_args, **call_kwargs),
            identity=identity,
            shapes=shapes,
        )

    model.flow.inference_chunk = profiled_inference_chunk
    manifest = {
        "schema_version": "token2wav-flow-real-run-v1",
        "model_path": str(model_path),
        "prompt_wav": str(prompt_wav),
        "token_asset": str(token_asset),
        "request_id": args.request_id,
        "generation_id": args.generation_id,
        "steps": args.steps,
        "float16": bool(args.float16),
        "diagnostic_only": True,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    state = model.create_stream_state(str(prompt_wav))
    model.stream_with_state(tokens, str(prompt_wav), state, last_chunk=False)
    state = model.create_stream_state(str(prompt_wav))
    active["value"] = True
    pcm_bytes = 0
    start_ns = time.monotonic_ns()
    try:
        for sequence_no in range(args.steps):
            model.set_profile_context(
                session_id=args.request_id,
                generation_id=args.generation_id,
                sequence_no=sequence_no,
            )
            pcm = model.stream_with_state(
                tokens,
                str(prompt_wav),
                state,
                last_chunk=sequence_no == args.steps - 1,
            )
            if not isinstance(pcm, (bytes, bytearray)) or len(pcm) == 0:
                raise ValueError("Flow run produced empty or non-bytes PCM")
            pcm_bytes += len(pcm)
        status = "VALID"
        error = None
    except Exception as exc:
        status = "INVALID"
        error = f"{exc.__class__.__name__}: {exc}"
    finally:
        active["value"] = False
        profiler.flush()
    attempt = {
        "status": status,
        "error": error,
        "pcm_bytes": pcm_bytes,
        "wall_time_ms": (time.monotonic_ns() - start_ns) / 1_000_000.0,
    }
    (output_dir / "attempt.json").write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    invalid_attempts = () if status == "VALID" else (attempt,)
    report_paths = generate_flow_reports(
        profiler.records,
        output_dir,
        invalid_attempts=invalid_attempts,
    )
    return {
        "status": status,
        "attempt": attempt,
        "profile_status": dict(profiler.profile_status),
        "reports": {key: str(path) for key, path in report_paths.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-id", default="flow-profile-request")
    parser.add_argument("--generation-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--float16", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_profile(args)
    except Exception as exc:
        print(f"INVALID_RUN: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
