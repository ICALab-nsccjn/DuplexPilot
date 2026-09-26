"""Public-trace runner for the realtime HTTP B0--B3 harness."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

from lychee_fd.runtime.apr.paper_systems import get_system_spec
from tools.benchmarks.realtime_public_client import (
    RealtimeHTTPClient,
    wav_chunk_from_file,
)
from workloads.apr_public.schema import PublicTrace, read_trace_jsonl


def _runtime_event_errors(events: list[dict[str, Any]]) -> tuple[str, ...]:
    failures: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "").strip().lower()
        detail = ""
        if event_type == "error":
            detail = str(event.get("error") or event.get("detail") or "error")
        elif event_type == "status":
            status = str(event.get("status") or "").strip()
            if "inference error" in status.casefold():
                detail = f"status={status}"
        if not detail:
            continue
        round_id = event.get("round_id")
        prefix = f"round={round_id} " if round_id is not None else ""
        failures.append(f"{prefix}{detail}")
    return tuple(failures)



def _run_one_session(
    client: RealtimeHTTPClient,
    spec: Any,
    trace: PublicTrace,
    session_id: str,
    *,
    wall_start: float,
    sleep: bool,
) -> dict[str, Any]:
    events = [event for event in trace.events if event.session_id == session_id]
    arrival = events[0].timestamp_s
    if sleep:
        time.sleep(max(0.0, arrival - (time.monotonic() - wall_start)))
    local_start = time.monotonic()
    response = client.start(spec, run_id=f"public-{trace.manifest.trace_id}-{session_id}")
    remote_id = str(response.get("session_id") or "")
    if not remote_id:
        raise RuntimeError(f"realtime start returned no session_id for {session_id}")

    received_events: list[dict[str, Any]] = []
    event_errors: list[BaseException] = []
    event_thread = threading.Thread(
        target=lambda: _collect_events(client, remote_id, received_events, event_errors),
        name=f"public-events-{session_id}",
        daemon=True,
    )
    event_thread.start()
    previous = arrival
    cancelled = False
    sent_chunks = 0
    try:
        for event in events:
            if sleep:
                time.sleep(max(0.0, event.timestamp_s - previous))
            previous = event.timestamp_s
            if event.event_type == "audio_chunk" and event.audio_path:
                chunk = wav_chunk_from_file(
                    event.audio_path,
                    offset_s=event.audio_offset_s,
                    duration_s=event.audio_duration_s,
                )
                client.send_chunk(remote_id, chunk, sent_epoch_ms=int(time.time() * 1000))
                sent_chunks += 1
            elif event.event_type == "cancel":
                client.stop(remote_id)
                cancelled = True
                break
        if not cancelled:
            client.stop(remote_id)
    finally:
        event_thread.join(timeout=client.timeout_s + 2.0)

    pcm_events = [event for event in received_events if event.get("type") == "audio_chunk_pcm"]
    errors = [event for event in received_events if event.get("type") == "error"]
    runtime_errors = _runtime_event_errors(received_events)
    done = any(event.get("type") == "done" for event in received_events)
    return {
        "session_id": session_id,
        "accepted": True,
        "done": done,
        "cancelled": cancelled,
        "sent_chunks": sent_chunks,
        "pcm_chunks": len(pcm_events),
        "pcm_bytes": sum(len(str(event.get("pcm_b64") or "")) for event in pcm_events),
        "ownership_errors": len(errors),
        "runtime_errors": list(runtime_errors),
        "event_count": len(received_events),
        "events": received_events,
        "elapsed_s": time.monotonic() - local_start,
        "server_start": response,
        "event_errors": [f"{type(error).__name__}: {error}" for error in event_errors],
    }


def _collect_events(
    client: RealtimeHTTPClient,
    session_id: str,
    target: list[dict[str, Any]],
    errors: list[BaseException],
) -> None:
    try:
        target.extend(client.events(session_id))
    except BaseException as exc:  # surfaced in the session result
        errors.append(exc)


def run_online_public_attempt(
    *,
    base_url: str,
    spec: Any,
    trace: PublicTrace,
    repeat: int,
    out_root: Path,
    sleep: bool = True,
    timeout_s: float = 120.0,
    client_factory: Callable[[str, float], RealtimeHTTPClient] = RealtimeHTTPClient,
) -> dict[str, Any]:
    """Run all trace sessions concurrently through the public realtime API."""
    if not isinstance(trace, PublicTrace):
        raise TypeError("trace must be a PublicTrace")
    session_ids = tuple(f"session-{index}" for index in range(trace.manifest.session_count))
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    wall_start = time.monotonic()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(32, len(session_ids))) as executor:
        futures = {
            executor.submit(
                _run_one_session,
                client_factory(base_url, timeout_s=timeout_s),
                spec,
                trace,
                session_id,
                wall_start=wall_start,
                sleep=sleep,
            ): session_id
            for session_id in session_ids
        }
        for future in as_completed(futures):
            session_id = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append({
                    "session_id": session_id,
                    "accepted": False,
                    "done": False,
                    "pcm_chunks": 0,
                    "pcm_bytes": 0,
                    "ownership_errors": 0,
                    "failure": f"{type(exc).__name__}: {exc}",
                })
    results.sort(key=lambda row: row["session_id"])
    valid = bool(results) and len(results) == len(session_ids) and all(
        row.get("accepted")
        and row.get("done")
        and int(row.get("pcm_chunks", 0)) > 0
        and int(row.get("ownership_errors", 0)) == 0
        and not row.get("event_errors")
        and not row.get("runtime_errors")
        for row in results
    )
    result = {
        "system": spec.system_id,
        "trace_id": trace.manifest.trace_id,
        "repeat": int(repeat),
        "valid": bool(valid),
        "session_count": len(session_ids),
        "completed_sessions": sum(bool(row.get("done")) for row in results),
        "ownership_errors": sum(int(row.get("ownership_errors", 0)) for row in results),
        "pcm_chunks": sum(int(row.get("pcm_chunks", 0)) for row in results),
        "runtime_errors": sum(len(row.get("runtime_errors", ())) for row in results),
        "sessions": results,
    }
    (out_root / "online_attempt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args()
    trace = read_trace_jsonl(args.trace)
    result = run_online_public_attempt(
        base_url=args.base_url,
        spec=get_system_spec(args.system),
        trace=trace,
        repeat=args.repeat,
        out_root=args.out_root,
        sleep=not args.no_sleep,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

