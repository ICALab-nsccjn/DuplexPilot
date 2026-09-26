from types import SimpleNamespace

from lychee_fd.runtime.apr.contracts import AcousticPcmRecord
from lychee_fd.runtime.apr.online_router import OnlineAcousticRouter
from lychee_fd.runtime.apr.paper_systems import get_system_spec


class _MetricLane:
    def __init__(self, **_kwargs):
        self.pending = []
        self.cleanup_ok = False

    def start(self, request_ids):
        return None

    def submit(self, batch):
        self.pending.append(batch)

    def process_one(self):
        batch = self.pending.pop(0)
        return SimpleNamespace(
            request_id=batch.request_id,
            worker_id=1,
            pcm_records=(
                AcousticPcmRecord(
                    request_id=batch.request_id,
                    stream_id=batch.stream_id,
                    generation_id=batch.generation_id,
                    sequence_no=batch.sequence_no,
                    pcm_bytes=b"pcm",
                    sample_rate=24000,
                    pcm_seq=0,
                ),
            ),
            state_version=batch.state_version + 1,
            worker_switch=True,
            handoff_mode="zero_copy",
            copied_state_bytes=0,
            checkpoint_bytes=0,
            handoff_latency_ns=123,
            capture_latency_ns=11,
            restore_latency_ns=0,
        )

    def cancel(self, request_id):
        self.pending = [item for item in self.pending if item.request_id != request_id]

    def close(self):
        self.cleanup_ok = True


def test_router_propagates_zero_copy_handoff_metrics():
    holder = {}

    def factory(**kwargs):
        holder["lane"] = _MetricLane(**kwargs)
        return holder["lane"]

    router = OnlineAcousticRouter(
        get_system_spec("rsv_dsv_apr_elastic_zero_copy_v2"),
        model=object(),
        prompt_wav="prompt.wav",
        worker_count=2,
        lane_factory=factory,
    )
    router.register("request", stream_id="stream", generation_id=0)
    result = router.submit(
        "request",
        stream_id="stream",
        generation_id=0,
        tokens=(1,),
        last_chunk=True,
    )

    assert result["handoff_mode"] == "zero_copy"
    assert result["copied_state_bytes"] == 0
    assert result["checkpoint_bytes"] == 0
    assert result["handoff_latency_ns"] == 123
    assert result["worker_switch"] is True
    router.close()
