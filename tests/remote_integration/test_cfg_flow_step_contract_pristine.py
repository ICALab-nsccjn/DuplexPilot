from __future__ import annotations

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "Step-Audio2"))

from cosyvoice2.flow.flow_matching import CausalCFMStepState, CausalConditionalCFM
from cosyvoice2.flow.decoder_dit import DiT


class TrackingEstimator:
    out_channels = 2

    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor | tuple[int, ...]]] = []

    def forward_chunk(self, *, x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
        self.calls.append(
            {
                "x": x.detach().clone(),
                "mu": mu.detach().clone(),
                "t": t.detach().clone(),
                "spks": spks.detach().clone(),
                "cond": cond.detach().clone(),
                "cnn_cache_shape": tuple(cnn_cache.shape) if cnn_cache is not None else (),
                "att_cache_shape": tuple(att_cache.shape) if att_cache is not None else (),
            }
        )
        batch = x.shape[0]
        cnn = torch.arange(batch, dtype=x.dtype).view(1, batch, 1, 1)
        att = torch.arange(batch, dtype=x.dtype).view(1, batch, 1, 1, 1)
        return torch.full_like(x, 0.25), cnn, att


class SmallLegacyEstimator(TrackingEstimator):
    def forward_chunk(self, *, x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
        dphi, _, _ = super().forward_chunk(
            x=x, mu=mu, t=t, spks=spks, cond=cond,
            cnn_cache=cnn_cache, att_cache=att_cache,
        )
        batch = x.shape[0]
        cnn = torch.arange(batch, dtype=x.dtype).view(1, batch, 1, 1).expand(16, -1, -1, -1)
        att = torch.arange(batch, dtype=x.dtype).view(1, batch, 1, 1, 1).expand(16, -1, -1, x.shape[2], -1)
        return dphi, cnn, att


def _state(request_id: str, *, x_value: float = 1.0) -> CausalCFMStepState:
    return CausalCFMStepState(
        x=torch.full((1, 2, 3), x_value),
        t=torch.tensor([0.0]),
        dt=torch.tensor(0.5),
        step_index=0,
        t_span=torch.tensor([0.0, 0.5, 1.0]),
        mu=torch.full((1, 2, 3), x_value),
        speaker=torch.full((1, 4), x_value),
        condition=torch.full((1, 2, 3), x_value),
        input_cnn_cache=None,
        input_att_cache=None,
        request_id=request_id,
        generation_id=3,
        sequence_no=7,
        version=11,
    )


def test_logical_batch_two_expands_every_cfg_input_to_four_rows():
    estimator = TrackingEstimator()
    cfm = CausalConditionalCFM(estimator, inference_cfg_rate=0.7)

    first, second = cfm.advance_chunk_step_batch((_state("a", x_value=1.0), _state("b", x_value=2.0)))
    call = estimator.calls[-1]

    assert call["x"].shape[0] == 4
    assert call["mu"].shape[0] == 4
    assert call["t"].shape == (4,)
    assert call["spks"].shape[0] == 4
    assert call["cond"].shape[0] == 4
    assert torch.equal(call["mu"][:2], torch.cat((first.mu, second.mu), dim=0))
    assert torch.count_nonzero(call["mu"][2:]) == 0
    assert first.request_id == "a" and second.request_id == "b"


def test_batched_cfg_cache_rows_are_split_back_by_request_identity():
    estimator = TrackingEstimator()
    cfm = CausalConditionalCFM(estimator, inference_cfg_rate=0.7)

    first, second = cfm.advance_chunk_step_batch((_state("a"), _state("b")))

    assert first.completed_cnn_cache is not None
    assert second.completed_cnn_cache is not None
    assert first.completed_att_cache is not None
    assert second.completed_att_cache is not None
    assert first.completed_cnn_cache[:, :, 0, 0].tolist() == [[0.0, 1.0]]
    assert second.completed_cnn_cache[:, :, 0, 0].tolist() == [[2.0, 3.0]]
    assert first.completed_att_cache[:, :, 0, 0, 0].tolist() == [[0.0, 1.0]]
    assert second.completed_att_cache[:, :, 0, 0, 0].tolist() == [[2.0, 3.0]]


def test_cfm_batch_two_uses_real_dit_batch_four_path():
    estimator = DiT(
        in_channels=7,
        out_channels=2,
        depth=1,
        num_heads=1,
        head_dim=2,
        hidden_size=4,
    )
    cfm = CausalConditionalCFM(estimator, inference_cfg_rate=0.7)
    states = (
        CausalCFMStepState(
            x=torch.ones(1, 2, 3), t=torch.tensor([0.0]), dt=torch.tensor(0.5),
            step_index=0, t_span=torch.tensor([0.0, 0.5]),
            mu=torch.ones(1, 2, 3), speaker=torch.ones(1, 1),
            condition=torch.ones(1, 2, 3), input_cnn_cache=None,
            input_att_cache=None, request_id="a",
        ),
        CausalCFMStepState(
            x=torch.ones(1, 2, 3), t=torch.tensor([0.0]), dt=torch.tensor(0.5),
            step_index=0, t_span=torch.tensor([0.0, 0.5]),
            mu=torch.ones(1, 2, 3), speaker=torch.ones(1, 1),
            condition=torch.ones(1, 2, 3), input_cnn_cache=None,
            input_att_cache=None, request_id="b",
        ),
    )

    first, second = cfm.advance_chunk_step_batch(states)

    assert first.x.shape == (1, 2, 3)
    assert second.x.shape == (1, 2, 3)
    assert first.completed_cnn_cache.shape[1] == 2
    assert second.completed_cnn_cache.shape[1] == 2


def test_b1_public_steps_match_legacy_euler_result_and_logical_size_excludes_scratch():
    estimator = SmallLegacyEstimator()
    cfm = CausalConditionalCFM(estimator, inference_cfg_rate=0.7)
    cfm.cnn_cache_buffer = torch.zeros(2, 16, 2, 1, 1)
    cfm.att_cache_buffer = torch.zeros(2, 16, 2, 1, 3, 1)
    x = torch.ones(1, 2, 3)
    t_span = torch.tensor([0.0, 0.5, 1.0])
    mu = torch.ones(1, 2, 3)
    speaker = torch.ones(1, 4)
    condition = torch.ones(1, 2, 3)

    legacy_x, legacy_cnn, legacy_att = cfm.solve_euler_chunk(
        x.clone(), t_span, mu, speaker, condition
    )
    state = cfm.begin_chunk_steps(
        x=x.clone(),
        t_span=t_span,
        mu=mu,
        spks=speaker,
        cond=condition,
        request_id="a",
        generation_id=3,
        sequence_no=7,
        version=11,
    )
    cfm.advance_chunk_step(state)
    cfm.advance_chunk_step(state)
    public_x, public_cnn, public_att = cfm.finish_chunk_steps(state)

    assert torch.equal(public_x, legacy_x)
    assert isinstance(public_cnn, tuple)
    assert isinstance(public_att, tuple)
    assert torch.equal(torch.stack(public_cnn), legacy_cnn[: len(public_cnn)])
    assert torch.equal(torch.stack(public_att), legacy_att[: len(public_att)])
    assert state.logical_state_size_bytes() < 10_000_000


def test_step_state_uses_one_request_owned_cache_slot_set():
    estimator = TrackingEstimator()
    cfm = CausalConditionalCFM(estimator, inference_cfg_rate=0.7)
    steps = 2
    cnn_cache = torch.arange(
        steps * 1 * 2 * 2 * 1, dtype=torch.float32
    ).reshape(steps, 1, 2, 2, 1)
    att_cache = torch.arange(
        steps * 1 * 2 * 1 * 3 * 2, dtype=torch.float32
    ).reshape(steps, 1, 2, 1, 3, 2)

    state = cfm.begin_chunk_steps(
        x=torch.ones(1, 2, 3),
        t_span=torch.tensor([0.0, 0.5, 1.0]),
        mu=torch.ones(1, 2, 3),
        spks=torch.ones(1, 4),
        cond=torch.ones(1, 2, 3),
        cnn_cache=cnn_cache,
        att_cache=att_cache,
    )

    assert isinstance(state.input_cnn_cache, tuple)
    assert isinstance(state.input_att_cache, tuple)
    assert state.input_cnn_cache is state._cnn_history
    assert state.input_att_cache is state._att_history
    assert len(state.input_cnn_cache) == steps
    assert len(state.input_att_cache) == steps

    cfm.advance_chunk_step(state)

    assert state.input_cnn_cache is state._cnn_history
    assert state.input_att_cache is state._att_history
    assert state.completed_cnn_cache is state._cnn_history[0]
    assert state.completed_att_cache is state._att_history[0]


def test_finish_cache_keeps_actual_length_without_stack_or_capacity_padding(
    monkeypatch,
):
    history = (
        torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        torch.arange(6, 12, dtype=torch.float32).reshape(1, 2, 3),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("finish must not materialize a padded cache")

    monkeypatch.setattr(torch, "stack", forbidden)
    monkeypatch.setattr(torch, "cat", forbidden)

    finished = CausalConditionalCFM._finish_cache(history, capacity=16)

    assert finished is history
    assert len(finished) == 2


def test_exact_b2_reorders_per_request_cfg_cache_rows_before_estimator():
    class CacheEchoEstimator:
        out_channels = 2

        def __init__(self):
            self.cnn_rows = None
            self.att_rows = None

        def forward_chunk(
            self, *, x, mu, t, spks, cond, cnn_cache=None, att_cache=None
        ):
            self.cnn_rows = cnn_cache[0, :, 0, 0].tolist()
            self.att_rows = att_cache[0, :, 0, 0, 0].tolist()
            return torch.zeros_like(x), cnn_cache.clone(), att_cache.clone()

    def cached_state(request_id: str, cond_value: float):
        state = _state(request_id, x_value=cond_value)
        cnn = torch.tensor(
            [[[[cond_value]]], [[[cond_value + 0.5]]]], dtype=torch.float32
        ).reshape(1, 2, 1, 1)
        att = torch.tensor(
            [[[[[cond_value]]]], [[[[cond_value + 0.5]]]]],
            dtype=torch.float32,
        ).reshape(1, 2, 1, 1, 1)
        state.input_cnn_cache = (cnn, None)
        state.input_att_cache = (att, None)
        state._cnn_history = state.input_cnn_cache
        state._att_history = state.input_att_cache
        return state

    estimator = CacheEchoEstimator()
    cfm = CausalConditionalCFM(estimator, inference_cfg_rate=0.7)
    cfm.advance_chunk_step_batch(
        (cached_state("a", 10.0), cached_state("b", 20.0))
    )
    # The estimator's CFG rows are conditional [a, b], then unconditional
    # [a, b].  Per-request caches must use the same physical ordering.
    assert estimator.cnn_rows == [10.0, 20.0, 10.5, 20.5]
    assert estimator.att_rows == [10.0, 20.0, 10.5, 20.5]
