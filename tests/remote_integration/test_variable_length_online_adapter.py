from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import threading
import time

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "Step-Audio2"))

from lychee_fd.runtime.apr.contracts import AcousticPcmRecord, AcousticTokenBatch
from lychee_fd.runtime.apr.online_router import OnlineAcousticRouter
from lychee_fd.runtime.apr.online_step_coordinator import PreparedFlowChunk
from lychee_fd.runtime.apr.paper_systems import get_system_spec
from lychee_fd.runtime.apr.flow_batch_runtime import FlowBatchRuntime
from lychee_fd.runtime.apr.deadline_bounded_flow_scheduler import FlowStepItem
from tools.apr.flow_batch_acoustic_lane import (
    APRFlowBatchAcousticLane,
    VariableLengthToken2WavFlowChunkExecutionAdapter,
)


def _state(request_id: str, *, att_len: int, generation_id: int = 1):
    from cosyvoice2.flow.flow_matching import CausalCFMStepState

    t_span = torch.tensor([0.0, 0.5, 1.0])
    att = torch.zeros(1, 2, 1, att_len, 2)
    cnn = torch.zeros(1, 2, 1, 1)
    return CausalCFMStepState(
        x=torch.zeros(1, 2, 3),
        t=t_span[:1].clone(),
        dt=torch.tensor(0.5),
        step_index=0,
        t_span=t_span,
        mu=torch.zeros(1, 2, 3),
        speaker=torch.zeros(1, 4),
        condition=torch.zeros(1, 2, 3),
        input_cnn_cache=(cnn,),
        input_att_cache=(att,),
        request_id=request_id,
        generation_id=generation_id,
        sequence_no=0,
        version=0,
        _cnn_history=(cnn,),
        _att_history=(att,),
    )


class _Model:
    def __init__(self):
        self.single_calls = 0
        self.exact_batch_calls = []
        self.variable_batch_calls = []

    def create_stream_state(self, _prompt):
        return {"flow_cache": {}, "hift_cache": {}}

    def begin_chunk_steps(self, *_args, **kwargs):
        return _state(kwargs["request_id"], att_len=2, generation_id=kwargs["generation_id"])

    def advance_chunk_step(self, state):
        self.single_calls += 1
        return replace(state, step_index=state.step_index + 1)

    def advance_chunk_step_batch(self, states):
        self.exact_batch_calls.append(len(states))
        return tuple(replace(state, step_index=state.step_index + 1) for state in states)

    def advance_chunk_step_variable_batch(self, states):
        self.variable_batch_calls.append(tuple(state.request_id for state in states))
        return tuple(replace(state, step_index=state.step_index + 1) for state in states)

    def finish_chunk_steps(self, state):
        return torch.zeros(1, 2, 3), {}

    def render_chunk_pcm(self, *_args, **_kwargs):
        return b"pcm"


class _Lane:
    def __init__(self, model):
        self.model = model
        self.prompt_wav = "prompt.wav"
        self.worker_count = 2
        self._worker_cursor = 0

    @staticmethod
    def _state_shape(state, fallback, *, wildcard_attention_time=False):
        current_length = int(state.x.shape[-1])
        return (
            (1, 2, current_length, -1)
            if wildcard_attention_time
            else (1, 2, current_length, int(state.input_att_cache[0].shape[3]))
        )

    @staticmethod
    def _state_device(state):
        return "cpu"

    @staticmethod
    def _state_dtype(state):
        return "float32"

    def _request(self, request_id):
        return type("Request", (), {
            "stream_id": f"stream-{request_id}",
            "generation_id": 0,
            "stream_state": {"flow_cache": {}, "hift_cache": {}},
        })()

    def _begin_state(self, batch, _request):
        return _state(batch.request_id, att_len=2, generation_id=batch.generation_id)


def test_variable_adapter_uses_wildcard_attention_time_and_variable_api():
    model = _Model()
    lane = _Lane(model)
    adapter = VariableLengthToken2WavFlowChunkExecutionAdapter(lane)
    first = _state("a", att_len=5, generation_id=3)
    second = _state("b", att_len=9, generation_id=77)
    first_chunk = PreparedFlowChunk(
        batch=AcousticTokenBatch("a", "sa", 3, 0, (1, 2), "src-a", 0, 1),
        state=first,
        n_timesteps=2,
        worker_id=0,
    )
    second_chunk = PreparedFlowChunk(
        batch=AcousticTokenBatch("b", "sb", 77, 0, (1, 2), "src-b", 0, 2),
        state=second,
        n_timesteps=2,
        worker_id=1,
    )
    left = adapter.make_step_item(first_chunk)
    right = adapter.make_step_item(second_chunk)
    assert left.shape_signature == right.shape_signature
    updated = adapter.advance_step_batch((first, second))
    assert [state.request_id for state in updated] == ["a", "b"]
    assert [state.step_index for state in updated] == [1, 1]
    assert model.variable_batch_calls == [("a", "b")]
    assert model.exact_batch_calls == []


def test_variable_adapter_does_not_accept_different_current_shape_in_signature():
    model = _Model()
    lane = _Lane(model)
    adapter = VariableLengthToken2WavFlowChunkExecutionAdapter(lane)
    first = _state("a", att_len=5)
    second = replace(
        _state("b", att_len=9),
        x=torch.zeros(1, 2, 4),
        mu=torch.zeros(1, 2, 4),
        condition=torch.zeros(1, 2, 4),
    )
    chunks = (
        PreparedFlowChunk(
            batch=AcousticTokenBatch("a", "sa", 1, 0, (1,), "src-a", 0, 1),
            state=first,
            n_timesteps=2,
            worker_id=0,
        ),
        PreparedFlowChunk(
            batch=AcousticTokenBatch("b", "sb", 1, 0, (1,), "src-b", 0, 2),
            state=second,
            n_timesteps=2,
            worker_id=1,
        ),
    )
    assert adapter.make_step_item(chunks[0]).shape_signature != adapter.make_step_item(chunks[1]).shape_signature


def test_variable_system_spec_is_explicit_and_bounded():
    spec = get_system_spec("rsv_dsv_apr_step_variable_b2")
    assert spec.max_flow_batch_size == 2
    assert spec.flow_batch_variant == "variable_length"
    assert spec.flow_execution_mode == "sync"


class _RouterAdapter:
    def __init__(self, variable=False):
        self.variable = variable
        self.calls = []

    def register(self, *_args, **_kwargs):
        return None

    def prepare_chunk(self, batch):
        return PreparedFlowChunk(
            batch=batch,
            state=_RouterState(
                batch.request_id, batch.generation_id, batch.state_version
            ),
            n_timesteps=1,
            worker_id=0,
        )

    def make_step_item(self, chunk):
        return FlowStepItem(
            request_id=chunk.batch.request_id,
            generation_id=chunk.batch.generation_id,
            version=chunk.batch.state_version,
            state=chunk.state,
            ready_at_ns=time.monotonic_ns(),
            model_identity="fake",
            device="cpu",
            dtype="float32",
            step_index=0,
            shape_signature=(1, 2, -1) if self.variable else (1, 2, 3),
            last_chunk=chunk.batch.last_chunk,
            n_timesteps=1,
        )

    def advance_step(self, state):
        return replace(state, step_index=1)

    def advance_step_batch(self, states):
        self.calls.append(len(states))
        return tuple(replace(state, step_index=1) for state in states)

    def update_step(self, chunk, state):
        chunk.state = state

    def finalize_chunk(self, chunk):
        return type("Progress", (), {
            "request_id": chunk.batch.request_id,
            "state_version": chunk.batch.state_version + 1,
            "worker_id": 0,
            "pcm_records": (),
        })()

    def cancel(self, _request_id):
        return None


@dataclass(frozen=True)
class _RouterState:
    request_id: str
    generation_id: int
    version: int
    step_index: int = 0


class _RouterLane:
    def __init__(self, **_kwargs):
        self.exact = _RouterAdapter(variable=False)
        self.variable = _RouterAdapter(variable=True)

    def start(self, request_ids, **_kwargs):
        return tuple(request_ids)

    def online_step_adapter(self):
        return self.exact

    def variable_length_online_step_adapter(self):
        return self.variable

    def close(self):
        return None


def test_router_selects_variable_adapter_for_variable_system():
    holder = {}

    def factory(**kwargs):
        holder["lane"] = _RouterLane(**kwargs)
        return holder["lane"]

    router = OnlineAcousticRouter(
        get_system_spec("rsv_dsv_apr_step_variable_b2"),
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

    def submit(request_id, generation_id):
        try:
            barrier.wait(timeout=1)
            router.submit(
                request_id,
                stream_id=f"stream-{request_id}",
                generation_id=generation_id,
                tokens=(1, 2, 3),
                last_chunk=True,
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=submit, args=("a", 1)),
        threading.Thread(target=submit, args=("b", 99)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert not errors
    assert holder["lane"].variable.calls == [2]
    assert holder["lane"].exact.calls == []
    router.close()


def test_runtime_emits_variable_batch_lifecycle_events():
    class Backend:
        variable_length = True

        def advance_step(self, state):
            return replace(state, step_index=state.step_index + 1)

        def advance_step_batch(self, states):
            return tuple(replace(state, step_index=state.step_index + 1) for state in states)

        def finish(self, state):
            return state

    @dataclass(frozen=True)
    class State:
        request_id: str
        generation_id: int = 1
        version: int = 0
        step_index: int = 0

    events = []
    runtime = FlowBatchRuntime(
        backend=Backend(), max_batch_size=2, max_batch_wait_ms=0, event_sink=events.append
    )
    for request_id in ("a", "b"):
        runtime.submit(
            FlowStepItem(
                request_id=request_id,
                generation_id=1,
                version=0,
                state=State(request_id),
                ready_at_ns=0,
                model_identity="flow",
                device="cpu",
                dtype="float32",
                step_index=0,
                shape_signature=(1, 2, -1),
                last_chunk=False,
                n_timesteps=2,
            )
        )
    result = runtime.run_next(now_ns=0)
    assert result is not None and result.committed
    names = [event["event"] for event in events]
    assert "FLOW_VARIABLE_BATCH_ATTEMPT" in names
    assert "FLOW_VARIABLE_BATCH_COMPLETE" in names
