#!/usr/bin/env python3
"""Experiment-only realtime replay and telemetry runner for Lychee-FD.

This runner talks to the official realtime HTTP API and never imports or
modifies Lychee-FD runtime code.  All local clocks use perf_counter_ns();
server epoch timestamps are retained only for correlating the runtime's
existing stage/timeline logs.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import shutil
import statistics
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Optional


INPUT_RATE = 16_000
CHUNK_MS = 400
CHUNK_SAMPLES = INPUT_RATE * CHUNK_MS // 1000
TTS_RATE_DEFAULT = 24_000
SCHEMA = "lychee-fd-life-v1"


def perf_ns() -> int:
    return time.perf_counter_ns()


def epoch_ms() -> int:
    return int(time.time() * 1000)


def build_session_start_payload(runtime_mode: Optional[str] = None, run_id: Optional[str] = None) -> dict:
    """Build the official realtime session-start payload."""
    payload = {
        "infer_window_ms": CHUNK_MS,
        "strict_infer_window": True,
        "stage_timing_log": True,
        "control_prob_trace_log": False,
        "tts_chunk_size": 1,
        "start_speak_factor": 1.2,
        "start_listen_factor": 1.2,
        "end_speak_factor": 1.0,
    }
    if runtime_mode is not None:
        payload["runtime_mode"] = str(runtime_mode)
    if run_id is not None:
        payload["run_id"] = str(run_id)
    return payload


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def jsonl_append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def http_json(url: str, method: str = "GET", payload: Any = None,
             body: Optional[bytes] = None, headers: Optional[dict] = None,
             timeout: float = 120.0) -> Any:
    data = body
    h = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return None
        return json.loads(raw.decode("utf-8", "replace"))


def read_pcm16_mono(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected PCM16 WAV, sample_width={width}")
    vals = list(struct.unpack("<%dh" % (len(frames) // 2), frames))
    if channels > 1:
        vals = [int(sum(vals[i:i + channels]) / channels) for i in range(0, len(vals), channels)]
    if rate != INPUT_RATE:
        # The formal32 inputs are already 16 kHz.  Keep a deterministic linear
        # resampler for validation inputs without adding a dependency.
        target = max(1, round(len(vals) * INPUT_RATE / rate))
        out: list[int] = []
        for i in range(target):
            pos = i * (len(vals) - 1) / max(1, target - 1)
            left = int(pos)
            right = min(left + 1, len(vals) - 1)
            frac = pos - left
            out.append(int(round(vals[left] * (1.0 - frac) + vals[right] * frac)))
        vals = out
    return vals


def wav_bytes(samples: list[int]) -> bytes:
    raw = b"".join(struct.pack("<h", max(-32768, min(32767, int(v)))) for v in samples)
    import io
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(INPUT_RATE)
        wf.writeframes(raw)
    return out.getvalue()


def split_chunks(samples: list[int]) -> list[bytes]:
    chunks = []
    for start in range(0, len(samples), CHUNK_SAMPLES):
        part = list(samples[start:start + CHUNK_SAMPLES])
        if len(part) < CHUNK_SAMPLES:
            part.extend([0] * (CHUNK_SAMPLES - len(part)))
        chunks.append(wav_bytes(part))
    return chunks


def parse_sse(lines: list[str], recv_ns: int, recv_epoch: int) -> Optional[dict]:
    name = "message"
    data: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if not data:
        return None
    raw = "\n".join(data)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"type": "invalid_json", "raw": raw}
    if not isinstance(payload, dict):
        payload = {"type": name, "value": payload}
    payload.setdefault("type", name)
    return {
        "received_perf_ns": recv_ns,
        "received_epoch_ms": recv_epoch,
        "sse_event": name,
        "payload": payload,
    }


class PlaybackClock:
    """A 1x logical PCM player started at first PCM arrival."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_ns: Optional[int] = None
        self.last_ns: Optional[int] = None
        self.buffer_ms = 0.0
        self.played_ms = 0.0
        self.generated_ms = 0.0
        self.gap_count = 0
        self.gap_duration_ms = 0.0
        self.records: list[dict] = []

    def _advance(self, now_ns: int) -> None:
        if self.started_ns is None or self.last_ns is None:
            return
        elapsed = max(0.0, (now_ns - self.last_ns) / 1_000_000.0)
        if elapsed <= 0:
            return
        consumed = min(self.buffer_ms, elapsed)
        self.buffer_ms -= consumed
        self.played_ms += consumed
        gap = max(0.0, elapsed - consumed)
        if gap > 0:
            if self.gap_count == 0 or self.records and self.records[-1].get("kind") != "gap":
                self.gap_count += 1
            self.gap_duration_ms += gap
            self.records.append({"kind": "gap", "perf_ns": now_ns, "gap_ms": gap})
        self.last_ns = now_ns

    def add_pcm(self, now_ns: int, samples: int, sample_rate: int, channels: int) -> dict:
        playable = (float(samples) / max(1, sample_rate)) * 1000.0
        with self.lock:
            if self.started_ns is None:
                self.started_ns = now_ns
                self.last_ns = now_ns
            self._advance(now_ns)
            self.generated_ms += playable
            self.buffer_ms += playable
            rec = {
                "kind": "pcm",
                "perf_ns": now_ns,
                "samples": int(samples),
                "sample_rate": int(sample_rate),
                "channels": int(channels),
                "playable_ms": round(playable, 3),
                "generated_audio_ms": round(self.generated_ms, 3),
                "played_audio_ms": round(self.played_ms, 3),
                "playback_buffer_ms": round(self.buffer_ms, 3),
            }
            self.records.append(rec)
            return dict(rec)

    def snapshot(self, now_ns: Optional[int] = None) -> dict:
        with self.lock:
            self._advance(now_ns or perf_ns())
            return {
                "generated_audio_ms": round(self.generated_ms, 3),
                "played_audio_ms": round(self.played_ms, 3),
                "playback_buffer_ms": round(max(0.0, self.buffer_ms), 3),
                "gap_count": int(self.gap_count),
                "gap_duration_ms": round(self.gap_duration_ms, 3),
            }


class GpuSampler(threading.Thread):
    def __init__(self, active: set[str], active_lock: threading.Lock, out_path: Path, interval: float = 0.2):
        super().__init__(daemon=True)
        self.active = active
        self.active_lock = active_lock
        self.out_path = out_path
        self.interval = interval
        self.stop_event = threading.Event()
        self.rows: list[dict] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            t_ns = perf_ns()
            t_epoch = epoch_ms()
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
                    stderr=subprocess.STDOUT, text=True, timeout=5,
                )
                with self.active_lock:
                    active = sorted(self.active)
                for line in raw.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) < 7:
                        continue
                    self.rows.append({
                        "timestamp_perf_ns": t_ns,
                        "timestamp_epoch_ms": t_epoch,
                        "gpu_index": int(parts[0]),
                        "gpu_name": parts[1],
                        "gpu_uuid": parts[2],
                        "utilization_gpu_pct": float(parts[3]),
                        "memory_used_mib": float(parts[4]),
                        "memory_total_mib": float(parts[5]),
                        "power_w": float(parts[6]),
                        "active_session_ids": active,
                    })
            except Exception as exc:
                self.rows.append({"timestamp_perf_ns": t_ns, "timestamp_epoch_ms": t_epoch, "error": f"{type(exc).__name__}: {exc}"})
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()

    def flush(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SessionRunner:
    def __init__(self, base_url: str, workload: str, input_path: Path, out_dir: Path,
                 index: int, barrier: threading.Barrier, active: set[str], active_lock: threading.Lock,
                 seed: int, timeout_sec: float, runtime_mode: Optional[str] = None,
                 run_id: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.workload = workload
        self.input_path = input_path
        self.out_dir = out_dir
        self.index = index
        self.barrier = barrier
        self.active = active
        self.active_lock = active_lock
        self.seed = seed
        self.timeout_sec = timeout_sec
        self.runtime_mode = runtime_mode
        self.run_id = run_id
        self.session_id: Optional[str] = None
        self.start_response: dict = {}
        self.events: list[dict] = []
        self.sse_lines: list[str] = []
        self.send_records: list[dict] = []
        self.telemetry: list[dict] = []
        self.sse_error: Optional[str] = None
        self.stop_requested = False
        self.stop_error: Optional[str] = None
        self.server_cleanup_timed_out = False
        self.done = threading.Event()
        self.first_pcm = threading.Event()
        self.sse_thread: Optional[threading.Thread] = None
        self.playback = PlaybackClock()
        self.started_perf_ns = perf_ns()
        self.started_epoch_ms = epoch_ms()
        self.epoch_to_perf_ns = self.started_perf_ns - self.started_epoch_ms * 1_000_000
        self.first_planned_perf_ns: Optional[int] = None
        self.latest_received_input_ms = 0.0
        self.latest_model_consumed_input_ms = 0.0
        self.engine_step_id: Optional[str] = None
        self.round_ids: list[int] = []

    def _server_epoch_to_perf(self, server_epoch_ms: Any) -> int:
        return int(float(server_epoch_ms) * 1_000_000 + self.epoch_to_perf_ns)

    @staticmethod
    def cleanup_wait_timeout_sec(timeout_sec: float) -> float:
        """Use the configured run budget for server-side cleanup waiting."""
        return max(0.0, float(timeout_sec))

    def _request_stop(self) -> None:
        """Request server-side session stop at most once."""
        if not self.session_id or self.stop_requested:
            return
        self.stop_requested = True
        self._write_telemetry("server_stop_requested")
        try:
            response = http_json(
                f"{self.base_url}/api/realtime/session/{self.session_id}/stop",
                method="POST",
                timeout=self.timeout_sec,
            )
            self._write_telemetry("server_stop_ack", stop_response=response)
        except Exception as exc:
            self.stop_error = f"{type(exc).__name__}: {exc}"
            self.sse_error = self.sse_error or f"stop: {self.stop_error}"
            self._write_telemetry("server_stop_error", error=self.stop_error)

    def _write_telemetry(self, kind: str, **fields: Any) -> None:
        now = perf_ns()
        snap = self.playback.snapshot(now)
        rec = {
            "schema_version": SCHEMA,
            "kind": kind,
            "session_id": self.session_id,
            "session_index": self.index,
            "timestamp_perf_ns": now,
            "timestamp_epoch_ms": epoch_ms(),
            "input_chunk_seq": fields.pop("input_chunk_seq", None),
            "latest_received_input_ms": round(self.latest_received_input_ms, 3),
            "latest_model_consumed_input_ms": round(self.latest_model_consumed_input_ms, 3),
            "generated_audio_ms": snap["generated_audio_ms"],
            "played_audio_ms": snap["played_audio_ms"],
            "playback_buffer_ms": snap["playback_buffer_ms"],
            "engine_step_id": self.engine_step_id,
            "selected_session_ids": None,
            "waiting_session_ids": None,
            "batch_size": fields.pop("batch_size", None),
            **fields,
        }
        self.telemetry.append(rec)

    def _event_reader(self) -> None:
        assert self.session_id
        url = f"{self.base_url}/api/realtime/session/{self.session_id}/events"
        req = urllib.request.Request(url, method="GET")
        block: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                while True:
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    self.sse_lines.append(line)
                    if line:
                        block.append(line)
                        continue
                    event = parse_sse(block, perf_ns(), epoch_ms())
                    block = []
                    if event is None:
                        continue
                    self.events.append(event)
                    payload = event.get("payload") or {}
                    etype = str(payload.get("type") or event.get("sse_event") or "")
                    if etype == "stage_timing":
                        q = payload.get("queue") or {}
                        rid = int(payload.get("round_id") or 0)
                        if rid > 0:
                            self.round_ids.append(rid)
                        consumed = float(q.get("consumed_ms") or 0.0)
                        self.latest_model_consumed_input_ms = max(
                            self.latest_model_consumed_input_ms,
                            float(max(rid * CHUNK_MS, consumed)),
                        )
                        self._write_telemetry(
                            "model_window_complete",
                            input_window_id=rid,
                            pending_window_ms=q.get("pending_before_ms"),
                            queue_delay_ms=(payload.get("stages") or {}).get("input_queue_wait"),
                            round_completed_epoch_ms=payload.get("round_completed_at_epoch_ms"),
                        )
                    elif etype == "audio_chunk_pcm":
                        pcm_samples = int(payload.get("num_samples") or 0)
                        sr = int(payload.get("sample_rate") or TTS_RATE_DEFAULT)
                        ch = int(payload.get("num_channels") or 1)
                        playback_rec = self.playback.add_pcm(event["received_perf_ns"], pcm_samples, sr, ch)
                        self.first_pcm.set()
                        playback_fields = dict(playback_rec)
                        playback_fields.pop("kind", None)
                        playback_fields.pop("sample_rate", None)
                        playback_fields.pop("channels", None)
                        playback_fields["pcm_event_perf_ns"] = playback_fields.pop("perf_ns", None)
                        self._write_telemetry(
                            "pcm_emit",
                            pcm_samples=pcm_samples,
                            sample_rate=sr,
                            channels=ch,
                            server_audio_emit_epoch_ms=payload.get("server_audio_emit_epoch_ms"),
                            t2w_synth_duration_ms=payload.get("t2w_synth_duration_ms"),
                            **playback_fields,
                        )
                    elif etype == "error":
                        self.sse_error = str(payload.get("error") or payload)
                    elif etype == "done":
                        self.done.set()
                    self._write_telemetry("sse_event", event_type=etype)
                    if etype == "done":
                        break
        except Exception as exc:
            if not self.done.is_set():
                self.sse_error = f"SSE reader: {type(exc).__name__}: {exc}"
        finally:
            if not self.done.is_set() and not self.sse_error:
                self.sse_error = "SSE reader ended before done event"
            self.done.set()
            self._write_telemetry("sse_reader_end", sse_error=self.sse_error)

    def _send_one(self, index: int, chunk: bytes, planned_ns: int) -> None:
        sent_ns = perf_ns()
        sent_epoch = epoch_ms()
        try:
            payload = http_json(
                f"{self.base_url}/api/realtime/session/{self.session_id}/chunk",
                method="POST", body=chunk,
                headers={
                    "Content-Type": "audio/wav",
                    "Content-Length": str(len(chunk)),
                    "X-Client-Chunk-Sent-Epoch-Ms": str(sent_epoch),
                }, timeout=self.timeout_sec,
            )
            self.latest_received_input_ms = max(
                self.latest_received_input_ms,
                float(index * CHUNK_MS),
            )
            record = {
                "input_chunk_seq": index,
                "planned_perf_ns": planned_ns,
                "client_send_perf_ns": sent_ns,
                "client_send_epoch_ms": sent_epoch,
                "client_return_perf_ns": perf_ns(),
                "chunk_duration_ms": CHUNK_MS,
                "status": "ok",
                "response": payload,
            }
            self.send_records.append(record)
            self._write_telemetry(
                "input_chunk_received_by_server",
                input_chunk_seq=index,
                client_send_perf_ns=sent_ns,
                client_send_epoch_ms=sent_epoch,
                client_return_perf_ns=record["client_return_perf_ns"],
                server_queued_ms=(payload or {}).get("queued_ms") if isinstance(payload, dict) else None,
            )
        except Exception as exc:
            record = {
                "input_chunk_seq": index,
                "planned_perf_ns": planned_ns,
                "client_send_perf_ns": sent_ns,
                "client_send_epoch_ms": sent_epoch,
                "client_return_perf_ns": perf_ns(),
                "chunk_duration_ms": CHUNK_MS,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.send_records.append(record)
            self._write_telemetry("input_send_error", input_chunk_seq=index, error=record["error"])
            raise

    def _send_workload(self, chunks: list[bytes]) -> None:
        self.barrier.wait(timeout=120)
        start_ns = perf_ns()
        self.first_planned_perf_ns = start_ns
        prefix = len(chunks)
        if self.workload == "barge_in":
            prefix = min(len(chunks), max(4, min(8, len(chunks) // 3 or 1)))
        next_deadline = start_ns
        for idx, chunk in enumerate(chunks, 1):
            next_deadline += CHUNK_MS * 1_000_000
            self._send_one(idx, chunk, next_deadline - CHUNK_MS * 1_000_000)
            if self.workload == "barge_in" and idx == prefix:
                self.first_pcm.wait(timeout=3.0)
            delay_ns = next_deadline - perf_ns()
            if delay_ns > 0:
                time.sleep(delay_ns / 1_000_000_000.0)
        # The official worker drains the final window after stop_requested.
        self._request_stop()

    def _copy_runtime_log(self, source: Any, name: str) -> Optional[str]:
        if not source:
            return None
        p = Path(str(source))
        if not p.exists():
            return None
        dest = self.out_dir / name
        try:
            shutil.copy2(p, dest)
            return str(dest)
        except Exception:
            return None

    def _postprocess_timeline(self) -> dict:
        stage_path = self.start_response.get("stage_timing_log_path")
        copied_stage = self._copy_runtime_log(stage_path, "stage_timing.txt")
        timeline_path = None
        if stage_path:
            p = Path(str(stage_path))
            timeline_path = str(p.with_name(p.name.replace("realtime_stage_timing_", "realtime_timeline_").replace(".txt", ".jsonl")))
        copied_timeline = self._copy_runtime_log(timeline_path, "timeline.jsonl")
        spans: list[dict] = []
        if copied_timeline:
            for line in Path(copied_timeline).read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for span in entry.get("spans") or []:
                    if isinstance(span, dict):
                        span = dict(span)
                        span.setdefault("session_id", self.session_id)
                        span.setdefault("round_id", entry.get("round_id"))
                        spans.append(span)
        decode_steps = [s for s in spans if s.get("name") == "decode_step"]
        engine_steps = [s for s in spans if s.get("name") in {"engine_step", "engine.step"}]
        self.engine_step_id = str(decode_steps[0].get("step")) if decode_steps else None
        return {
            "stage_timing_path": copied_stage,
            "timeline_path": copied_timeline,
            "span_count": len(spans),
            "decode_step_count": len(decode_steps),
            "engine_step_count": len(engine_steps),
            "span_names": sorted({str(s.get("name")) for s in spans}),
            "decode_steps": decode_steps,
        }

    def run(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        chunks = split_chunks(read_pcm16_mono(self.input_path))
        try:
            self.start_response = http_json(
                f"{self.base_url}/api/realtime/session/start", method="POST",
                payload=build_session_start_payload(self.runtime_mode, self.run_id),
                timeout=self.timeout_sec,
            )
            self.session_id = str(self.start_response["session_id"])
            json_dump(self.out_dir / "start_response.json", self.start_response)
            with self.active_lock:
                self.active.add(self.session_id)
            self.sse_thread = threading.Thread(target=self._event_reader, daemon=True)
            self.sse_thread.start()
            self._write_telemetry("session_start", input_window_count=len(chunks), seed=self.seed)
            self._send_workload(chunks)
        except Exception as exc:
            self.sse_error = self.sse_error or f"session: {type(exc).__name__}: {exc}"
        finally:
            # A chunk/send/SSE failure must not leave the server-side worker
            # occupying a scheduler slot while this attempt is recorded.
            self._request_stop()
            if self.sse_thread:
                # A server-side worker can finish even if an SSE proxy/client
                # misses the terminal event.  Honor the configured run
                # budget so cleanup is not abandoned after a fixed 45 sec.
                cleanup_timeout = self.cleanup_wait_timeout_sec(self.timeout_sec)
                self._write_telemetry(
                    "server_cleanup_wait",
                    cleanup_wait_timeout_sec=cleanup_timeout,
                )
                self.done.wait(timeout=cleanup_timeout)
                if not self.done.is_set():
                    self.server_cleanup_timed_out = True
                    self.sse_error = self.sse_error or "timeout waiting for SSE done event"
                    self._write_telemetry(
                        "server_cleanup_timeout",
                        cleanup_wait_timeout_sec=cleanup_timeout,
                    )
                    self.done.set()
                self.sse_thread.join(timeout=5)
            with self.active_lock:
                if self.session_id:
                    self.active.discard(self.session_id)
            self._write_telemetry("session_end", done=self.done.is_set(), sse_error=self.sse_error)

        timeline = self._postprocess_timeline()
        self.playback.snapshot(perf_ns())
        json_dump(self.out_dir / "playback_summary.json", self.playback.snapshot(perf_ns()))
        json_dump(self.out_dir / "timeline_summary.json", timeline)
        with (self.out_dir / "events.sse").open("w", encoding="utf-8") as f:
            f.write("\n".join(self.sse_lines) + "\n")
        with (self.out_dir / "events.jsonl").open("w", encoding="utf-8") as f:
            for item in self.events:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        with (self.out_dir / "chunk_send.jsonl").open("w", encoding="utf-8") as f:
            for item in sorted(self.send_records, key=lambda x: x.get("input_chunk_seq", 0)):
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        with (self.out_dir / "client_telemetry.jsonl").open("w", encoding="utf-8") as f:
            for item in self.telemetry:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        result = {
            "schema_version": SCHEMA,
            "workload": self.workload,
            "session_index": self.index,
            "session_id": self.session_id,
            "input_path": str(self.input_path),
            "input_chunks": len(chunks),
            "sent_chunks": len([x for x in self.send_records if x.get("status") == "ok"]),
            "chunk_ms": CHUNK_MS,
            "seed": self.seed,
            "sse_error": self.sse_error,
            "stop_requested": self.stop_requested,
            "stop_error": self.stop_error,
            "cleanup_wait_timeout_sec": self.cleanup_wait_timeout_sec(self.timeout_sec),
            "server_cleanup_timed_out": self.server_cleanup_timed_out,
            "done_event": self.done.is_set(),
            "pcm_events": sum(1 for x in self.events if (x.get("payload") or {}).get("type") == "audio_chunk_pcm"),
            "windows_observed": len(self.round_ids),
            "latest_model_consumed_input_ms": self.latest_model_consumed_input_ms,
            "playback": self.playback.snapshot(perf_ns()),
            "timeline": timeline,
            "status": "ok" if self.session_id and self.done.is_set() and not self.sse_error else "error",
        }
        json_dump(self.out_dir / "session_result.json", result)
        return result


def run_batch(args: argparse.Namespace) -> dict:
    root = Path(args.out_dir) / args.workload / f"c{args.concurrency}" / f"r{args.repeat_index}"
    root.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(x) for x in (args.inputs or [args.input])]
    if not input_paths:
        raise ValueError("at least one input is required")
    active: set[str] = set()
    active_lock = threading.Lock()
    gpu = GpuSampler(active, active_lock, root / "gpu_samples.jsonl", interval=0.2)
    gpu.start()
    barrier = threading.Barrier(args.concurrency)
    runners = [
        SessionRunner(
            args.base_url, args.workload, input_paths[i % len(input_paths)], root / f"session_{i + 1:02d}",
            i + 1, barrier, active, active_lock, args.seed, args.timeout_sec,
            args.runtime_mode,
            f"{args.workload}-r{args.repeat_index}",
        ) for i in range(args.concurrency)
    ]
    threads = [threading.Thread(target=r.run, daemon=True) for r in runners]
    for t in threads:
        t.start()
    deadline = time.time() + args.timeout_sec
    for t in threads:
        remaining = max(1.0, deadline - time.time())
        t.join(timeout=remaining)
    gpu.stop()
    gpu.join(timeout=5)
    gpu.flush()
    results = []
    for r in runners:
        p = r.out_dir / "session_result.json"
        if p.exists():
            results.append(json.loads(p.read_text(encoding="utf-8")))
    manifest = {
        "schema_version": SCHEMA,
        "workload": args.workload,
        "concurrency": args.concurrency,
        "repeat": args.repeat_index,
        "seed": args.seed,
        "inputs": [str(x) for x in input_paths],
        "base_url": args.base_url,
        "runtime_mode": args.runtime_mode or "native",
        "run_id": f"{args.workload}-r{args.repeat_index}",
        "chunk_ms": CHUNK_MS,
        "session_count": len(results),
        "session_results": results,
        "gpu_sample_count": len(gpu.rows),
        "model_sampling_seed_note": "official realtime API does not expose sampling seed; seed fixes runner/workload only",
    }
    json_dump(root / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:17860")
    p.add_argument("--workload", choices=["turn_taking", "barge_in", "native_overlap"], required=True)
    p.add_argument("--input", default=None)
    p.add_argument("--inputs", nargs="+", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--repeat-index", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runtime-mode", default=None)
    p.add_argument("--timeout-sec", type=float, default=900.0)
    run_batch(p.parse_args())


if __name__ == "__main__":
    main()
