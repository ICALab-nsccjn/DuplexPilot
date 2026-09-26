from __future__ import annotations

import pytest

from lychee_fd.runtime.apr.contracts import AcousticTokenBatch
from tools.apr.capacity_slo_lanes import ElasticAcousticLaneV2
from tools.benchmarks.paper_backend_selector import build_acoustic_lane
from lychee_fd.runtime.apr.paper_systems import get_system_spec


class _DeterministicModel:
    def create_stream_state(self, prompt_wav):
        return {"prompt_wav": str(prompt_wav), "position": 0}

    def stream_with_state(self, tokens, prompt_wav, state, last_chunk=False):
        state["position"] += len(tokens)
        return bytes([state["position"] & 0xFF])


def _batch(request_id: str, sequence_no: int = 0, state_version: int = 0):
    return AcousticTokenBatch(
        request_id=request_id,
        stream_id=f"stream-{request_id}",
        generation_id=0,
        sequence_no=sequence_no,
        stoken_ids=(1, 2),
        source_execution_id=f"source-{request_id}-{sequence_no}",
        state_version=state_version,
        created_monotonic_ns=sequence_no + 1,
    )


def test_zero_copy_lane_uses_transfer_without_checkpoint_clone():
    lane = ElasticAcousticLaneV2(
        worker_count=2,
        model=_DeterministicModel(),
        prompt_wav="prompt.wav",
        device="cpu",
        assignment_policy="contention_triggered",
        handoff_mode="zero_copy",
    )
    lane.start(("a", "b"))
    lane.submit(_batch("a"))
    first = lane.process_one(ready_request_ids=("a",))
    lane.submit(_batch("a", sequence_no=1, state_version=1))
    second = lane.process_one(ready_request_ids=("a", "b"))

    assert first is not None and second is not None
    assert second.worker_switch is True
    assert second.checkpoint_count == 0
    assert second.restore_count == 0
    assert second.handoff_mode == "zero_copy"
    assert second.copied_state_bytes == 0
    assert any(
        event.get("event_type") == "APR_ZERO_COPY_HANDOFF_COMMIT"
        for event in lane.events
    )
    lane.close()


def test_zero_copy_lane_does_not_handoff_without_a_ready_competitor():
    lane = ElasticAcousticLaneV2(
        worker_count=2,
        model=_DeterministicModel(),
        prompt_wav="prompt.wav",
        device="cpu",
        assignment_policy="contention_triggered",
        handoff_mode="zero_copy",
    )
    lane.start(("a", "b"))
    lane.submit(_batch("a"))
    first = lane.process_one(ready_request_ids=("a",))
    lane.submit(_batch("a", sequence_no=1, state_version=1))
    second = lane.process_one(ready_request_ids=("a",))

    assert first is not None and second is not None
    assert second.worker_switch is False
    assert second.checkpoint_count == 0
    assert not any(
        event.get("event_type") == "APR_ZERO_COPY_HANDOFF_COMMIT"
        for event in lane.events
    )
    lane.close()


def test_zero_copy_system_spec_is_explicit_and_selector_opt_in():
    spec = get_system_spec("rsv_dsv_apr_elastic_zero_copy_v2")
    assert spec.handoff_mode == "zero_copy"
    lane = build_acoustic_lane(
        spec,
        worker_count=2,
        model=_DeterministicModel(),
        prompt_wav="prompt.wav",
    )
    assert isinstance(lane, ElasticAcousticLaneV2)
    assert lane.handoff_mode == "zero_copy"
    lane.close()
