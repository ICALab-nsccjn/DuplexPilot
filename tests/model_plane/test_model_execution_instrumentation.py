import json
from collections import defaultdict, deque
import threading

from lychee_fd.runtime.model_execution_trace import ModelExecutionTraceRecorder
from lychee_fd.runtime.vllm_generation import _VLLMModelAdapter
from lychee_fd.vllm_integration.engine import _PatchedLycheeVLLMEngine


class _Output:
    def __init__(self, request_id):
        self.request_id = request_id


class _Engine:
    def __init__(self):
        self.calls = 0

    def has_unfinished_requests(self):
        return True

    def step(self):
        self.calls += 1
        return [_Output("r0"), _Output("r1")]


class _StreamEngine:
    def step_generate_stream(self, **kwargs):
        yield {"round": 0}
        yield {"round": 1}


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_engine_step_trace_records_all_rows_and_duration(tmp_path):
    path = tmp_path / "engine.jsonl"
    wrapped = object.__new__(_PatchedLycheeVLLMEngine)
    wrapped.engine = _Engine()
    wrapped._pending_request_outputs = defaultdict(deque)
    wrapped._row_request_states = {}
    wrapped._model_execution_trace = ModelExecutionTraceRecorder(path)

    output = wrapped._next_request_output("r0")

    assert output.request_id == "r0"
    rows = _records(path)
    assert rows[0]["event_type"] == "MODEL_ENGINE_STEP_START"
    end = next(row for row in rows if row["event_type"] == "MODEL_ENGINE_STEP_END")
    assert end["output_request_ids"] == ["r0", "r1"]
    assert end["model_batch_size"] == 2
    assert end["duration_ns"] >= 0
    assert [row["request_id"] for row in rows if row["event_type"] == "MODEL_ROW_OUTPUT"] == [
        "r0", "r1"
    ]


def test_adapter_automatically_records_lock_scope_when_enabled(tmp_path):
    path = tmp_path / "lock.jsonl"
    adapter = object.__new__(_VLLMModelAdapter)
    adapter._stream_lock = threading.RLock()
    adapter._lock_event_recorder = None
    adapter._model_execution_trace = ModelExecutionTraceRecorder(path)

    with adapter._stream_lock_scope(("r0",)):
        pass

    rows = _records(path)
    assert len(rows) == 1
    assert rows[0]["event_type"] == "MODEL_LOCK"
    assert rows[0]["request_ids"] == ["r0"]


def test_model_trace_can_be_disabled_without_runtime_side_effect(tmp_path):
    path = tmp_path / "absent.jsonl"
    adapter = object.__new__(_VLLMModelAdapter)
    adapter._stream_lock = threading.RLock()
    adapter._lock_event_recorder = None
    adapter._model_execution_trace = ModelExecutionTraceRecorder("")

    with adapter._stream_lock_scope(("r0",)):
        pass

    assert not path.exists()


def test_adapter_can_reenter_lock_for_each_stream_item(tmp_path):
    """The production generator acquires the stream lock once per output."""
    path = tmp_path / "reentrant.jsonl"
    adapter = object.__new__(_VLLMModelAdapter)
    adapter._stream_lock = threading.RLock()
    adapter._lock_event_recorder = None
    adapter._model_execution_trace = ModelExecutionTraceRecorder(path)
    adapter._engine = _StreamEngine()
    adapter.runtime_mode = "dynamic_virtualized"
    adapter.config = object()

    outputs = list(adapter.multi_head_generate_stream(
        input_ids=[1],
        stoken_ids=[2],
        control_input_ids=[3],
        prefix_input_ids=[4],
        audio_input_ids=[5],
        audio_embeds=object(),
        request_id="r0",
    ))

    assert [item["round"] for item in outputs] == [0, 1]
