import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from lychee_fd.runtime.apr.contracts import AcousticPcmRecord
from lychee_fd.runtime.apr.acoustic_backend import AcousticBackend
from lychee_fd.runtime.apr.online_router import (
    OnlineAcousticRouter,
    OnlineAcousticRouterError,
    OnlineAcousticRouterRegistry,
    resolve_paper_system_payload,
)
from lychee_fd.runtime.apr.paper_systems import get_system_spec
from tools.apr.real_acoustic_lanes import FixedAffinityAcousticLane
from lychee_fd.runtime.apr.online_step_coordinator import PreparedFlowChunk
from lychee_fd.runtime.apr.deadline_bounded_flow_scheduler import FlowStepItem
from tools.benchmarks.paper_backend_selector import build_acoustic_lane


class FakeLane:
    def __init__(self, **_kwargs):
        self.pending = []
        self.submitted = []
        self.started = []
        self.cancelled = []
        self.cleanup_ok = False

    def start(self, request_ids):
        self.started.extend(request_ids)

    def submit(self, batch):
        self.submitted.append(batch)
        self.pending.append(batch)

    def process_one(self):
        batch = self.pending.pop(0)
        return SimpleNamespace(
            request_id=batch.request_id,
            worker_id=0,
            pcm_records=(AcousticPcmRecord(
                request_id=batch.request_id,
                stream_id=batch.stream_id,
                generation_id=batch.generation_id,
                sequence_no=batch.sequence_no,
                pcm_bytes=b"pcm-" + batch.request_id.encode(),
                sample_rate=24000,
                pcm_seq=batch.sequence_no,
            ),),
            state_version=batch.state_version + 1,
        )

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        self.pending = [batch for batch in self.pending if batch.request_id != request_id]

    def close(self):
        self.cleanup_ok = True


class FakeBackend(AcousticBackend):
    def capture_state(self, request_id):
        return {"request_id": request_id}

    def restore_state(self, state):
        return None

    def resume(self):
        return None

    def process(self, tokens):
        return None

    def commit_pcm(self):
        return ()

    def cancel(self):
        return None


def _factory(holder):
    def build(**kwargs):
        holder["lane"] = FakeLane(**kwargs)
        return holder["lane"]
    return build


@dataclass(frozen=True)
class RouterFakeState:
    request_id: str
    generation_id: int
    version: int
    step_index: int = 0


class RouterOnlineAdapter:
    def __init__(self):
        self.batch_sizes = []
        self.cancelled = []

    def register(self, request_id, *, stream_id, generation_id):
        return None

    def prepare_chunk(self, batch):
        return PreparedFlowChunk(
            batch=batch,
            state=RouterFakeState(batch.request_id, batch.generation_id, batch.state_version),
            n_timesteps=1,
            worker_id=0,
        )

    def make_step_item(self, chunk):
        return FlowStepItem(
            request_id=chunk.batch.request_id,
            generation_id=chunk.batch.generation_id,
            version=chunk.batch.state_version,
            state=chunk.state,
            ready_at_ns=0,
            model_identity="router-fake-flow",
            device="cpu",
            dtype="float32",
            step_index=chunk.state.step_index,
            shape_signature=(3,),
            last_chunk=chunk.batch.last_chunk,
            n_timesteps=chunk.n_timesteps,
        )

    def advance_step(self, state):
        return replace(state, step_index=state.step_index + 1)

    def advance_step_batch(self, states):
        self.batch_sizes.append(len(states))
        return tuple(replace(state, step_index=state.step_index + 1) for state in states)

    def update_step(self, chunk, next_state):
        chunk.state = next_state

    def finalize_chunk(self, chunk):
        return SimpleNamespace(
            request_id=chunk.batch.request_id,
            state_version=chunk.batch.state_version + 1,
            worker_id=0,
            pcm_records=(),
        )

    def cancel(self, request_id):
        self.cancelled.append(request_id)


class OnlineFakeLane(FakeLane):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adapter = RouterOnlineAdapter()

    def online_step_adapter(self):
        return self.adapter


def test_router_uses_online_step_coordinator_for_public_adapter():
    holder = {}

    def factory(**kwargs):
        holder["lane"] = OnlineFakeLane(**kwargs)
        return holder["lane"]

    router = OnlineAcousticRouter(
        get_system_spec("rsv_dsv_apr_step_batch_b2"),
        model=object(),
        prompt_wav="prompt.wav",
        worker_count=2,
        max_batch_wait_ms=10.0,
        lane_factory=factory,
    )
    router.register("a", stream_id="stream-a", generation_id=0)
    router.register("b", stream_id="stream-b", generation_id=9)
    barrier = threading.Barrier(2)
    results = {}
    errors = []

    def submit(request_id, generation_id):
        try:
            barrier.wait(timeout=1.0)
            results[request_id] = router.submit(
                request_id,
                stream_id=f"stream-{request_id}",
                generation_id=generation_id,
                tokens=(1, 2, 3),
                last_chunk=True,
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=submit, args=("a", 0)),
        threading.Thread(target=submit, args=("b", 9)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
    assert not errors
    assert set(results) == {"a", "b"}
    assert holder["lane"].adapter.batch_sizes == [2]
    router.unregister("a")
    router.unregister("b")
    assert set(holder["lane"].adapter.cancelled) == {"a", "b"}
    router.close()



def test_router_forms_one_batch_for_compatible_concurrent_requests():
    holder = {}
    router = OnlineAcousticRouter(
        get_system_spec("rsv_dsv_apr_step_batch_b2"),
        model=object(),
        prompt_wav="prompt.wav",
        worker_count=2,
        max_batch_wait_ms=10.0,
        lane_factory=_factory(holder),
    )
    router.register("a", stream_id="stream-a", generation_id=0)
    router.register("b", stream_id="stream-b", generation_id=0)
    barrier = threading.Barrier(2)
    results = {}
    errors = []

    def submit(request_id):
        try:
            barrier.wait(timeout=1.0)
            results[request_id] = router.submit(
                request_id,
                stream_id=f"stream-{request_id}",
                generation_id=0,
                tokens=(1, 2),
                last_chunk=True,
            )
        except Exception as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(request_id,)) for request_id in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
    assert not errors
    assert set(results) == {"a", "b"}
    assert router.events[-1]["event"] == "FLOW_BATCH_COMPLETE"
    formed = [event for event in router.events if event["event"] == "FLOW_BATCH_FORMED"]
    assert formed and formed[-1]["batch_size"] == 2
    assert len(holder["lane"].submitted) == 2
    router.close()


def test_router_drops_completion_after_unregister():
    holder = {}
    router = OnlineAcousticRouter(
        get_system_spec("rsv_dsv_apr_step_b1"),
        model=object(),
        prompt_wav="prompt.wav",
        worker_count=1,
        lane_factory=_factory(holder),
    )
    router.register("a", stream_id="stream-a", generation_id=0)
    result = {}

    def submit():
        try:
            result["value"] = router.submit(
                "a", stream_id="stream-a", generation_id=0, tokens=(1,), last_chunk=True
            )
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=submit)
    thread.start()
    thread.join(timeout=2.0)
    assert "value" in result
    router.unregister("a")
    router.close()


def test_router_rejects_identity_mismatch():
    router = OnlineAcousticRouter(
        get_system_spec("rsv_dsv_apr_step_b1"),
        model=object(),
        prompt_wav="prompt.wav",
        worker_count=1,
        lane_factory=_factory({}),
    )
    router.register("a", stream_id="stream-a", generation_id=0)
    with pytest.raises(OnlineAcousticRouterError, match="identity"):
        router.submit("a", stream_id="wrong", generation_id=0, tokens=(1,), last_chunk=True)
    router.close()


def test_paper_payload_is_fail_closed_against_spec_mismatch():
    payload = {
        "paper_system_id": "rsv_dsv_apr_step_batch_b2",
        "acoustic_mode": "fixed_affinity",
        "max_flow_batch_size": 1,
    }
    with pytest.raises(ValueError, match="does not match"):
        resolve_paper_system_payload(payload, runtime_mode="dynamic_virtualized")


def test_registry_shares_router_for_same_system_and_prompt():
    holder = {}
    registry = OnlineAcousticRouterRegistry()
    spec = get_system_spec("rsv_dsv_apr_step_b1")
    model = object()
    first = registry.acquire(
        spec, model=model, prompt_wav="prompt.wav", worker_count=1,
        lane_factory=_factory(holder),
    )
    second = registry.acquire(
        spec, model=model, prompt_wav="prompt.wav", worker_count=1,
        lane_factory=_factory(holder),
    )
    assert first is second
    registry.release(spec, prompt_wav="prompt.wav")
    assert not holder["lane"].cleanup_ok
    registry.release(spec, prompt_wav="prompt.wav")
    assert holder["lane"].cleanup_ok


def test_fixed_affinity_admission_distributes_later_sessions():
    lane = FixedAffinityAcousticLane(
        worker_count=2,
        backend_factory=lambda _request_id: FakeBackend(),
    )
    lane.start(("a",))
    lane.start(("b",))
    assert lane.worker_for["a"] == 0
    assert lane.worker_for["b"] == 1
    lane.close()

import asyncio
import json

import lychee_fd.app as runtime_app


def test_realtime_start_resolves_paper_spec_before_router_use(monkeypatch, tmp_path):
    from fastapi import FastAPI

    api = FastAPI()
    runtime_app.register_realtime_session_routes(api)
    endpoint = next(
        route.endpoint
        for route in api.routes
        if getattr(route, "path", None) == "/api/realtime/session/start"
    )

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"RIFF")
    acquired = {}

    class FakeRegistry:
        def acquire(self, spec, **kwargs):
            acquired["spec"] = spec
            return object()

    class FakeRequest:
        async def body(self):
            return json.dumps(
                {
                    "run_id": "route-contract",
                    "runtime_mode": "native",
                    "paper_system_id": "lychee_native_affinity",
                    "acoustic_mode": "fixed_affinity",
                    "max_flow_batch_size": 1,
                }
            ).encode("utf-8")

    monkeypatch.setattr(runtime_app, "generator", object())
    monkeypatch.setattr(runtime_app, "token2wav_model", object())
    monkeypatch.setattr(runtime_app, "is_token2wav_available", lambda: True)
    monkeypatch.setattr(runtime_app, "ensure_local_token2wav_loaded", lambda: None)
    monkeypatch.setattr(runtime_app, "_resolve_prompt_wav_path", lambda _: str(prompt))
    monkeypatch.setattr(runtime_app, "_paper_acoustic_router_registry", FakeRegistry())
    monkeypatch.setattr(runtime_app, "_run_realtime_session_worker", lambda _: None)
    monkeypatch.setattr(runtime_app, "register_request_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_app, "REMOTE_TOKEN2WAV_ENABLED", False)

    response = asyncio.run(endpoint(FakeRequest()))
    body = json.loads(response.body.decode("utf-8"))
    assert response.status_code == 200
    assert body["paper_system_id"] == "lychee_native_affinity"
    assert acquired["spec"].system_id == "lychee_native_affinity"

    with runtime_app._realtime_sessions_lock:
        runtime_app._realtime_sessions.pop(body["session_id"], None)

def test_fixed_lane_accepts_event_sink_from_selector():
    events = []
    lane = build_acoustic_lane(
        get_system_spec("lychee_native_affinity"),
        worker_count=1,
        model=object(),
        prompt_wav="prompt.wav",
        event_sink=events.append,
    )
    lane.close()
