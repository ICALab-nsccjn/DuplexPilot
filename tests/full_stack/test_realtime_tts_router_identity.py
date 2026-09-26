from lychee_fd.app import RealtimeTTSPool


class _FakeRouter:
    def __init__(self):
        self.registered = []
        self.resets = []
        self.unregistered = []

    def register(self, request_id, *, stream_id, generation_id):
        self.registered.append((request_id, stream_id, generation_id))

    def reset(self, request_id, *, stream_id, generation_id):
        self.resets.append((request_id, stream_id, generation_id))

    def unregister(self, request_id):
        self.unregistered.append(request_id)


def test_event_start_resets_acoustic_router_generation():
    router = _FakeRouter()
    pool = RealtimeTTSPool(
        prompt_wav_path="prompt.wav",
        vocoder_hop_size=25,
        pre_lookahead_len=3,
        ownership_context={"logical_request_id": "request-1"},
        acoustic_router=router,
    )
    try:
        pool.submit_event_start()
        assert router.resets == [("request-1", pool._get_stream_id(), 1)]
    finally:
        pool.stop()

