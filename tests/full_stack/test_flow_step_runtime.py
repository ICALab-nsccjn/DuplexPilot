import torch

from profiling.token2wav_flow_profile.step_runtime import (
    FlowStepExecutor,
    FlowStepObservation,
    FlowStepProfiler,
)


class FakeEstimator:
    def forward_chunk(self, *, x, mu, t, spks, cond, cnn_cache, att_cache):
        del mu, t, spks, cond, cnn_cache, att_cache
        derivative = torch.ones_like(x)
        return derivative, torch.ones((1, 1, 1, 1), dtype=x.dtype), torch.ones(
            (1, 1, 1, 2, 1), dtype=x.dtype
        )


def _executor():
    return FlowStepExecutor(
        FakeEstimator(),
        inference_cfg_rate=0.7,
        cnn_cache_buffer=torch.zeros((2, 1, 1, 1)),
        att_cache_buffer=torch.zeros((2, 1, 1, 1, 4, 1)),
    )


def _state(executor):
    return executor.begin(
        x=torch.zeros((1, 1, 2)),
        t_span=torch.tensor([0.0, 1.0, 2.0]),
        mu=torch.zeros((1, 1, 2)),
        spks=torch.zeros((1, 1)),
        cond=torch.zeros((1, 1, 2)),
    )


def test_pause_after_step_zero_restore_and_step_one_matches_continuous():
    continuous_executor = _executor()
    continuous = _state(continuous_executor)
    continuous = continuous_executor.advance(continuous)
    continuous = continuous_executor.advance(continuous)
    continuous_output = continuous_executor.finish(continuous)

    paused_executor = _executor()
    paused = _state(paused_executor)
    observations = []
    paused = paused_executor.advance(paused, observer=observations.append)
    checkpoint = paused.clone()
    restored = checkpoint.clone()
    restored = paused_executor.advance(restored, observer=observations.append)
    restored_output = paused_executor.finish(restored)

    assert torch.equal(restored_output[0], continuous_output[0])
    assert torch.equal(restored_output[1], continuous_output[1])
    assert torch.equal(restored_output[2], continuous_output[2])
    assert [item.step_id for item in observations] == [0, 1]
    assert all(item.execution_time_ns >= 0 for item in observations)


def test_profiler_keeps_step_state_change_fields():
    profiler = FlowStepProfiler()
    record = FlowStepObservation(
        step_id=0,
        execution_time_ns=10,
        memory_change_bytes=20,
        state_change_bytes=0,
        state_change={"next_step_after": 2},
    )
    profiler.observe(record)
    assert profiler.to_list() == [{
        "step_id": 0,
        "execution_time_ns": 10,
        "execution_time_ms": 0.00001,
        "memory_change_bytes": 20,
        "state_change_bytes": 0,
        "state_change": {"next_step_after": 2},
    }]
