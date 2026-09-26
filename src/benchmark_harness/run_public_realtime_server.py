"""Start the API-only realtime server used by public B0--B3 runs."""

from __future__ import annotations

import argparse

from fastapi import FastAPI
import uvicorn

import lychee_fd.app as runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--token2wav-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--attn-impl", default="eager")
    args = parser.parse_args()
    runtime.load_models(args.model_path, args.token2wav_path, attn_impl=args.attn_impl)
    api = FastAPI(title="DuplexPilot public B0-B3 benchmark API")
    runtime.register_realtime_session_routes(api)
    uvicorn.run(api, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

