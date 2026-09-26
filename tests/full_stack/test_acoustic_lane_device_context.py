import torch

from tools.apr.real_acoustic_lanes import FixedAffinityAcousticLane


def test_lane_sets_acoustic_device_before_creating_stream_state(monkeypatch):
    selected = []

    monkeypatch.setenv("LYCHEEFD_TOKEN2WAV_DEVICE", "1")
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: selected.append(device))

    class FakeModel:
        def create_stream_state(self, prompt_wav):
            assert selected == [1]
            return {"flow_cache": {}, "hift_cache": {}}

    lane = FixedAffinityAcousticLane(
        worker_count=1,
        model=FakeModel(),
        prompt_wav="prompt.wav",
    )
    lane.start(("request-1",))
    assert selected == [1, 0]
