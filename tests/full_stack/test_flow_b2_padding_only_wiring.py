from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import threading

from lychee_fd.runtime.apr.flow_batch_runtime import FlowBatchRuntime, FlowStepItem
from lychee_fd.runtime.apr.online_router import OnlineAcousticRouter
from lychee_fd.runtime.apr.online_step_coordinator import PreparedFlowChunk
from lychee_fd.runtime.apr.paper_systems import get_system_spec
from tools.benchmarks.paper_backend_selector import build_acoustic_lane


class _Model:
    def create_stream_state(self, prompt_wav):
        return {"prompt_wav": prompt_wav, "flow_cache": {}, "hift_cache": {}}

    def begin_chunk_steps(self, *args, **kwargs):
        raise AssertionError("wiring test must not execute begin")

    def render_chunk_pcm(self, *args, **kwargs):
        raise AssertionError("wiring test must not render")

    def advance_chunk_step(self, state):
        return state

    def advance_chunk_step_batch(self, states):
        return tuple(states)

    def advance_chunk_step_variable_batch(self, states):
        return tuple(states)

    def advance_chunk_step_variable_chunk_padding_batch(self, states):
        return tuple(states)

    def finish_chunk_steps(self, state):
        return None, {}


def _item(request_id: str, *, shape: tuple[int, ...], step: int = 2) -> FlowStepItem:
    state = SimpleNamespace(
        request_id=request_id,
        generation_id=1,
        version=0,
        step_index=step,
    )
    return FlowStepItem(
        request_id=request_id,
        generation_id=1,
        version=0,
        state=state,
        ready_at_ns=0,
        model_identity="flow",
        device="cuda:1",
        dtype="torch.float32",
        step_index=step,
        shape_signature=shape,
        last_chunk=False,
        n_timesteps=10,
    )


def test_padding_only_system_spec_and_policy_are_explicit():
    spec = get_system_spec("rsv_dsv_apr_step_chunk_padding_b2")
    assert spec.flow_batch_variant == "chunk_padding"
    assert spec.max_flow_batch_size == 2


class _PaddingBackend:
    variable_length = True
    mixed_step = False
    chunk_padding = True
    mixed_step_max_batch_size = 2

    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def advance_step(self, state):
        return replace(state, step_index=state.step_index + 1)

    def advance_step_batch(self, states):
        raise AssertionError("padding-only path must use its explicit API")

    def advance_step_chunk_padding_batch(self, states):
        self.calls.append(tuple(state.request_id for state in states))
        return tuple(
            SimpleNamespace(**{**vars(state), "step_index": state.step_index + 1})
            for state in states
        )

    def finish(self, state):
        return None


def test_runtime_dispatches_padding_only_public_backend_method():
    backend = _PaddingBackend()
    runtime = FlowBatchRuntime(
        backend=backend,
        max_batch_size=2,
        max_batch_wait_ms=0.0,
        compatibility_policy="chunk_padding",
    )
    runtime.submit(_item("a", shape=(1, 80, -1)))
    runtime.submit(_item("b", shape=(1, 80, -1)))
    result = runtime.run_next(now_ns=0)
    assert result is not None and result.committed
    assert backend.calls == [("a", "b")]


def test_selector_exposes_padding_only_public_adapter():
    spec = get_system_spec("rsv_dsv_apr_step_chunk_padding_b2")
    lane = build_acoustic_lane(
        spec, worker_count=2, model=_Model(), prompt_wav="prompt.wav"
    )
    adapter = lane.chunk_padding_online_step_adapter()
    assert adapter.variable_length is True
    assert adapter.mixed_step is False
    assert adapter.chunk_padding is True
    assert callable(adapter.advance_step_chunk_padding_batch)


@dataclass(frozen=True)
class _State:
    request_id: str
    generation_id: int
    version: int
    step_index: int = 0


class _Adapter:
    variable_length = True
    mixed_step = False
    chunk_padding = True
    mixed_step_max_batch_size = 2

    def register(self, request_id, *, stream_id, generation_id):
        return None

    def prepare_chunk(self, batch):
        return PreparedFlowChunk(
            batch=batch,
            state=_State(batch.request_id, batch.generation_id, batch.state_version),
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
            model_identity="padding-test",
            device="cpu",
            dtype="float32",
            step_index=0,
            shape_signature=(1, 80, -1),
            last_chunk=chunk.batch.last_chunk,
            n_timesteps=1,
        )

    def advance_step(self, state):
        return replace(state, step_index=state.step_index + 1)

    def advance_step_batch(self, states):
        raise AssertionError("padding-only path must not use exact batch API")

    def advance_step_chunk_padding_batch(self, states):
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
        return None


class _Lane:
    def __init__(self, **_kwargs):
        self.adapter = _Adapter()

    def start(self, request_ids, **_kwargs):
        return tuple(request_ids)

    def chunk_padding_online_step_adapter(self):
        return self.adapter

    def close(self):
        return None


def test_router_selects_padding_only_adapter_and_forms_b2():
    holder = {}

    def factory(**kwargs):
        holder["lane"] = _Lane(**kwargs)
        return holder["lane"]

    router = OnlineAcousticRouter(
        get_system_spec("rsv_dsv_apr_step_chunk_padding_b2"),
        model=object(),
        prompt_wav="prompt.wav",
        worker_count=2,
        max_batch_wait_ms=10.0,
        lane_factory=factory,
    )
    router.register("a", stream_id="stream-a", generation_id=1)
    router.register("b", stream_id="stream-b", generation_id=2)
    barrier = threading.Barrier(2)
    errors = []

    def run(request_id, generation_id):
        try:
            barrier.wait(timeout=1.0)
            barrier.wait(timeout=1.0)
            router.submit(
                request_id,
                stream_id=f"stream-{request_id}",
                generation_id=generation_id,
                tokens=(1, 2, 3),
                last_chunk=False,
            )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("a", 1)),
        threading.Thread(target=run, args=("b", 2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)
    assert not errors
    router.close()
