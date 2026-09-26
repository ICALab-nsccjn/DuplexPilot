from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import threading

from lychee_fd.runtime.apr.contracts import AcousticTokenBatch
from lychee_fd.runtime.apr.online_router import OnlineAcousticRouter
from lychee_fd.runtime.apr.online_step_coordinator import PreparedFlowChunk
from lychee_fd.runtime.apr.paper_systems import get_system_spec
from lychee_fd.runtime.apr.deadline_bounded_flow_scheduler import FlowStepItem
from tools.benchmarks.paper_backend_selector import build_acoustic_lane


class _Model:
    def create_stream_state(self, prompt_wav):
        return {"prompt_wav": prompt_wav, "flow_cache": {}, "hift_cache": {}}

    def begin_chunk_steps(self, *args, **kwargs):  # pragma: no cover - wiring only
        raise AssertionError("not executed by this wiring test")

    def render_chunk_pcm(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not executed by this wiring test")

    def advance_chunk_step(self, state):
        return state

    def advance_chunk_step_batch(self, states):
        return tuple(states)

    def advance_chunk_step_variable_batch(self, states):
        return tuple(states)

    def finish_chunk_steps(self, state):
        return None, {}

    def advance_chunk_step_variable_mixed_chunk_padding_batch(self, states):
        return tuple(states)


def test_combined_online_system_spec_is_explicit_and_bounded():
    spec = get_system_spec("rsv_dsv_apr_step_mixed_chunk_padding_b2")
    assert spec.acoustic_mode == "apr_step"
    assert spec.max_flow_batch_size == 2
    assert spec.flow_batch_variant == "mixed_step_chunk_padding"
    assert spec.flow_execution_mode == "sync"


def test_selector_exposes_combined_public_adapter():
    spec = get_system_spec("rsv_dsv_apr_step_mixed_chunk_padding_b2")
    lane = build_acoustic_lane(
        spec, worker_count=2, model=_Model(), prompt_wav="prompt.wav"
    )
    adapter = lane.mixed_chunk_padding_online_step_adapter()
    assert adapter.variable_length is True
    assert adapter.mixed_step is True
    assert adapter.chunk_padding is True
    # The coordinator validates the generic mixed-step public contract before
    # it dispatches the policy-specific combined method.
    assert callable(adapter.advance_step_mixed_batch)
    assert callable(adapter.advance_step_mixed_chunk_padding_batch)


@dataclass(frozen=True)
class _State:
    request_id: str
    generation_id: int
    version: int
    step_index: int = 0


class _Adapter:
    variable_length = True
    mixed_step = True
    chunk_padding = True
    mixed_step_max_batch_size = 2

    def __init__(self):
        self.combined_calls = []

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
            model_identity="combined-test",
            device="cpu",
            dtype="float32",
            step_index=chunk.state.step_index,
            shape_signature=(1, 2, -1),
            last_chunk=chunk.batch.last_chunk,
            n_timesteps=chunk.n_timesteps,
        )

    def advance_step(self, state):
        return replace(state, step_index=state.step_index + 1)

    def advance_step_batch(self, states):  # pragma: no cover
        raise AssertionError("combined path must not use exact batch API")

    def advance_step_mixed_batch(self, states):  # pragma: no cover
        raise AssertionError("combined path must not use mixed-only API")

    def advance_step_mixed_chunk_padding_batch(self, states):
        self.combined_calls.append(tuple(state.request_id for state in states))
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

    def mixed_chunk_padding_online_step_adapter(self):
        return self.adapter

    def close(self):
        return None


def _submit(router, request_id, generation_id):
    return router.submit(
        request_id,
        stream_id=f"stream-{request_id}",
        generation_id=generation_id,
        tokens=(1, 2, 3),
        last_chunk=False,
    )


def test_router_selects_combined_adapter_and_forms_b2():
    holder = {}

    def factory(**kwargs):
        holder["lane"] = _Lane(**kwargs)
        return holder["lane"]

    router = OnlineAcousticRouter(
        get_system_spec("rsv_dsv_apr_step_mixed_chunk_padding_b2"),
        model=object(),
        prompt_wav="prompt.wav",
        worker_count=2,
        max_batch_wait_ms=10.0,
        lane_factory=factory,
    )
    router.register("a", stream_id="stream-a", generation_id=1)
    router.register("b", stream_id="stream-b", generation_id=99)
    barrier = threading.Barrier(2)
    errors = []

    def run(request_id, generation_id):
        try:
            barrier.wait(timeout=1.0)
            barrier.wait(timeout=1.0)
            _submit(router, request_id, generation_id)
        except Exception as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("a", 1)),
        threading.Thread(target=run, args=("b", 99)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)
    assert not errors
    assert len(holder["lane"].adapter.combined_calls) == 1
    assert set(holder["lane"].adapter.combined_calls[0]) == {"a", "b"}
    router.close()
