import torch

import lychee_fd.app as runtime_app


def test_required_dual_gpu_role_selects_acoustic_gpu_one(monkeypatch):
    selected = []
    monkeypatch.setenv("LYCHEEFD_REQUIRE_DUAL_GPU_PLACEMENT", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("LYCHEEFD_TOKEN2WAV_DEVICE", "1")
    monkeypatch.setenv("LYCHEEFD_VLLM_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("LYCHEEFD_VLLM_PIPELINE_PARALLEL_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: selected.append(device))

    runtime_app._configure_local_token2wav_device()

    assert selected == [1]
