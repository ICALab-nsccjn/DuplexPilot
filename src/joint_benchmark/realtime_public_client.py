"""Small standard-library client for the public realtime HTTP session API."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import time
import urllib.request
import wave
from typing import Any, Iterable, Iterator

from lychee_fd.runtime.apr.paper_systems import PaperSystemSpec
from lychee_fd.runtime.apr.joint_execution import get_joint_execution_spec
from workloads.apr_public.schema import PublicTrace


def wav_chunks_from_file(path: str | Path, *, chunk_ms: int = 400) -> tuple[bytes, ...]:
    if isinstance(chunk_ms, bool) or chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16000:
            raise ValueError("public client requires 16-kHz mono 16-bit WAV")
        frames_per_chunk = int(source.getframerate() * chunk_ms / 1000)
        if frames_per_chunk <= 0:
            raise ValueError("chunk_ms is smaller than one frame")
        chunks = []
        while True:
            frames = source.readframes(frames_per_chunk)
            if not frames:
                break
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(frames)
            chunks.append(buffer.getvalue())
    if not chunks:
        raise ValueError("WAV contains no audio frames")
    return tuple(chunks)


def wav_chunk_from_file(
    path: str | Path, *, offset_s: float, duration_s: float
) -> bytes:
    """Read exactly one canonical trace interval as a standalone WAV."""
    if offset_s < 0 or duration_s <= 0:
        raise ValueError("audio offset must be non-negative and duration positive")
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16000:
            raise ValueError("public client requires 16-kHz mono 16-bit WAV")
        start_frame = int(round(float(offset_s) * source.getframerate()))
        frame_count = max(1, int(round(float(duration_s) * source.getframerate())))
        if start_frame >= source.getnframes():
            raise ValueError("audio offset is beyond the WAV")
        source.setpos(start_frame)
        frames = source.readframes(frame_count)
        if not frames:
            raise ValueError("canonical audio chunk contains no frames")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(frames)
        return buffer.getvalue()


def build_start_payload(spec: PaperSystemSpec, *, run_id: str) -> dict[str, Any]:
    if not isinstance(spec, PaperSystemSpec) or not run_id:
        raise ValueError("spec and run_id are required")
    payload = {
        "run_id": str(run_id),
        "runtime_mode": spec.model_runtime_mode,
        "paper_system_id": spec.system_id,
        "acoustic_mode": spec.acoustic_mode,
        "max_flow_batch_size": spec.max_flow_batch_size,
        "strict_infer_window": True,
    }
    # Joint cells are explicit benchmark contracts.  Carry both dimensions
    # over the public boundary so a server cannot mistake a B_model=2 label
    # for a merely row-aware singleton run.
    try:
        joint = get_joint_execution_spec(spec.system_id)
    except ValueError:
        joint = None
    if joint is not None:
        payload.update(
            {
                "joint_id": joint.joint_id,
                "model_execution_mode": joint.model_execution_mode,
                "acoustic_execution_mode": joint.acoustic_execution_mode,
                "max_model_batch_size": joint.max_model_batch_size,
                "max_acoustic_batch_size": joint.max_acoustic_batch_size,
            }
        )
    return payload


class RealtimeHTTPClient:
    """Transport used by the benchmark harness; no server state is inspected."""

    def __init__(self, base_url: str, *, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _request(self, method: str, path: str, body: bytes = b"", content_type: str = "application/json") -> Any:
        request = urllib.request.Request(
            self.base_url + path,
            data=body if method != "GET" else None,
            method=method,
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = response.read()
        return json.loads(payload.decode("utf-8")) if payload else {}

    def start(self, spec: PaperSystemSpec, *, run_id: str) -> dict[str, Any]:
        payload = json.dumps(build_start_payload(spec, run_id=run_id)).encode("utf-8")
        return dict(self._request("POST", "/api/realtime/session/start", payload))

    def send_chunk(self, session_id: str, wav_bytes: bytes, *, sent_epoch_ms: int | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "audio/wav"}
        if sent_epoch_ms is not None:
            headers["X-Client-Chunk-Sent-Epoch-Ms"] = str(int(sent_epoch_ms))
        request = urllib.request.Request(
            self.base_url + f"/api/realtime/session/{session_id}/chunk",
            data=wav_bytes,
            method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return dict(json.loads(response.read().decode("utf-8")))

    def stop(self, session_id: str) -> dict[str, Any]:
        return dict(self._request("POST", f"/api/realtime/session/{session_id}/stop"))

    def events(self, session_id: str) -> Iterator[dict[str, Any]]:
        request = urllib.request.Request(
            self.base_url + f"/api/realtime/session/{session_id}/events",
            method="GET",
            headers={"Accept": "text/event-stream"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            data_line = None
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data:"):
                    data_line = line[5:].strip()
                elif not line and data_line is not None:
                    try:
                        yield dict(json.loads(data_line))
                    finally:
                        data_line = None


def replay_public_session(
    client: RealtimeHTTPClient,
    spec: PaperSystemSpec,
    trace: PublicTrace,
    *,
    session_id: str,
    sleep: bool = False,
) -> dict[str, Any]:
    """Replay one trace session through start/chunk/stop without private access."""
    events = [event for event in trace.events if event.session_id == session_id]
    if not events or events[0].event_type != "arrival":
        raise ValueError(f"trace has no arrival for {session_id}")
    started = time.monotonic()
    response = client.start(spec, run_id=f"public-{trace.manifest.trace_id}-{session_id}")
    remote_id = str(response.get("session_id") or "")
    if not remote_id:
        raise RuntimeError("realtime start returned no session_id")
    previous = events[0].timestamp_s
    sent = 0
    for event in events:
        if sleep and event.timestamp_s > previous:
            time.sleep(event.timestamp_s - previous)
        previous = event.timestamp_s
        if event.event_type == "audio_chunk" and event.audio_path:
            chunk = wav_chunk_from_file(
                event.audio_path,
                offset_s=event.audio_offset_s,
                duration_s=event.audio_duration_s,
            )
            client.send_chunk(remote_id, chunk, sent_epoch_ms=int(time.time() * 1000))
            sent += 1
        elif event.event_type == "cancel":
            client.stop(remote_id)
    if not any(event.event_type == "cancel" for event in events):
        client.stop(remote_id)
    return {
        "session_id": session_id,
        "remote_session_id": remote_id,
        "sent_chunks": sent,
        "elapsed_s": time.monotonic() - started,
    }
