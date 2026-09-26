import json
import threading
import time

from lychee_fd.runtime.model_execution_trace import ModelExecutionTraceRecorder


def test_recorder_writes_schema_and_sanitizes_tensor_like_values(tmp_path):
    path = tmp_path / "model.jsonl"
    recorder = ModelExecutionTraceRecorder(path)

    class TensorLike:
        shape = (2, 3)
        dtype = "torch.float32"
        device = "cuda:0"

        def numel(self):
            return 6

        def __repr__(self):  # pragma: no cover - must never be used
            raise AssertionError("tensor repr must not be serialized")

    assert recorder.record(
        "MODEL_REQUEST_REGISTER",
        request_id="r0",
        model_batch_size=1,
        tensor=TensorLike(),
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["schema"] == "lychee-model-execution-v1"
    assert row["event_type"] == "MODEL_REQUEST_REGISTER"
    assert row["request_id"] == "r0"
    assert row["tensor"] == {
        "container": "tensor",
        "shape": [2, 3],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "numel": 6,
    }


def test_lock_scope_records_wait_and_hold_without_changing_ownership(tmp_path):
    path = tmp_path / "lock.jsonl"
    recorder = ModelExecutionTraceRecorder(path)
    lock = threading.RLock()
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with recorder.lock_scope(lock, request_ids=("r0",)):
            entered.set()
            release.wait(timeout=1)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(timeout=1)
    with recorder.lock_scope(lock, request_ids=("r1",)):
        pass
    release.set()
    thread.join(timeout=1)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["event_type"] == "MODEL_LOCK" for row in rows)
    assert any(row["request_ids"] == ["r1"] and row["wait_ns"] >= 0 for row in rows)
    assert all(row["hold_ns"] >= 0 for row in rows)


def test_recorder_is_fail_open_when_path_is_empty():
    recorder = ModelExecutionTraceRecorder("")
    assert recorder.enabled is False
    assert recorder.record("MODEL_ENGINE_STEP_START", request_id="r0") is False
