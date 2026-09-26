"""Real A100 equivalence gate for the public CFG-aware Flow step contract.

The test is opt-in because loading the production Token2Wav checkpoint requires
an A100. It compares the frozen inference_chunk control path with the
public B=1 step path, then compares B=2/B=4 step execution with independent
single-session executions.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch


MODEL_PATH = Path(
    os.environ.get("LYCHEEFD_REAL_T2W_MODEL", "/mnt/DuplexPilot/data/models/token2wav")
)
PROMPT_WAV = Path(
    os.environ.get(
        "LYCHEEFD_REAL_T2W_PROMPT",
        "/mnt/DuplexPilot/data/DuplexPilot/lychee_dsv_closure_worktree/"
        "frontend/public/clone_24k_mono/default_male.wav",
    )
)
N_TIMESTEPS = 10
SEED_TOKENS = (1493, 4299, 4218, 2049, 528, 2752, 4850, 4569)


def _real_gate_enabled() -> bool:
    return os.environ.get("RUN_REAL_FLOW_BATCH_A100") == "1"


def _tokens(length: int = 16) -> list[int]:
    return list((SEED_TOKENS * ((length + len(SEED_TOKENS) - 1) // len(SEED_TOKENS)))[:length])


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_tree(item) for item in value)
    return value


def _assert_tree_close(left, right, *, rtol: float, atol: float, path: str = "root"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.shape == right.shape, path
        assert left.dtype == right.dtype, path
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=path)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict), path
        assert list(left) == list(right), path
        for key in left:
            _assert_tree_close(left[key], right[key], rtol=rtol, atol=atol, path=f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)):
        assert isinstance(right, (tuple, list)), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_tree_close(left_item, right_item, rtol=rtol, atol=atol, path=f"{path}[{index}]")
        return
    assert left == right, path

def _assert_public_cache_matches_frozen(public, frozen, *, rtol, atol):
    expected_keys = {
        "conformer_cnn_cache",
        "conformer_att_cache",
        "estimator_cnn_cache",
        "estimator_att_cache",
    }
    assert set(public) == expected_keys
    assert set(frozen) == expected_keys
    for key in ("conformer_cnn_cache", "conformer_att_cache"):
        _assert_tree_close(
            public[key], frozen[key], rtol=rtol, atol=atol, path=key
        )
    for key in ("estimator_cnn_cache", "estimator_att_cache"):
        logical = public[key]
        assert isinstance(logical, tuple)
        materialized = torch.stack(logical)
        control = frozen[key][: len(logical)]
        _assert_tree_close(
            materialized, control, rtol=rtol, atol=atol, path=key
        )



def _pcm_metrics(left: bytes, right: bytes) -> dict[str, float | int | bool]:
    left_i16 = np.frombuffer(left, dtype="<i2")
    right_i16 = np.frombuffer(right, dtype="<i2")
    if left_i16.shape != right_i16.shape:
        return {
            "same_length": False,
            "sample_count": int(left_i16.size),
            "other_sample_count": int(right_i16.size),
            "normalized_rmse": float("inf"),
            "correlation": -1.0,
            "snr_db": float("-inf"),
        }
    if left_i16.size == 0:
        return {
            "same_length": True,
            "sample_count": 0,
            "normalized_rmse": 0.0,
            "correlation": 1.0,
            "snr_db": float("inf"),
        }
    left_f = left_i16.astype(np.float64) / 32768.0
    right_f = right_i16.astype(np.float64) / 32768.0
    error = left_f - right_f
    rmse = float(np.sqrt(np.mean(error * error)))
    scale = max(float(np.max(np.abs(left_f))), float(np.max(np.abs(right_f))), 1e-12)
    if np.std(left_f) == 0.0 or np.std(right_f) == 0.0:
        correlation = 1.0 if np.array_equal(left_i16, right_i16) else 0.0
    else:
        correlation = float(np.corrcoef(left_f, right_f)[0, 1])
    signal_power = float(np.sum(left_f * left_f))
    error_power = float(np.sum(error * error))
    snr_db = float("inf") if error_power == 0.0 else 10.0 * math.log10(signal_power / error_power)
    return {
        "same_length": True,
        "sample_count": int(left_i16.size),
        "normalized_rmse": rmse / scale,
        "correlation": correlation,
        "snr_db": snr_db,
    }


def _pcm_from_mel(model, prompt_wav: str, mel: torch.Tensor, *, last_chunk: bool) -> bytes:
    from token2wav import fade_in_out

    stream_state = model.create_stream_state(prompt_wav)
    hift_cache = stream_state["hift_cache"]
    combined = torch.cat([hift_cache["mel"], mel], dim=2)
    speech, _ = model.hift(combined, hift_cache["source"])
    if hift_cache["speech"].shape[-1] > 0:
        speech = fade_in_out(speech, hift_cache["speech"], model.speech_window)
    if not last_chunk:
        speech = speech[:, :-model.source_cache_len]
    return (speech.squeeze(0).cpu().numpy().clip(-1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _finish_states(model, states):
    values = tuple(states)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(N_TIMESTEPS):
            if len(values) == 1:
                values = (model.advance_chunk_step(values[0]),)
            else:
                values = tuple(model.advance_chunk_step_batch(values))
    finished = tuple(model.finish_chunk_steps(state) for state in values)
    return values, finished


@pytest.fixture(scope="module")
def real_model():
    if not _real_gate_enabled():
        pytest.skip("set RUN_REAL_FLOW_BATCH_A100=1 to run the A100 gate")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not MODEL_PATH.is_dir() or not PROMPT_WAV.is_file():
        pytest.skip(f"real assets unavailable: model={MODEL_PATH}, prompt={PROMPT_WAV}")
    from token2wav import Token2wav

    torch.cuda.set_device(int(os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE", "1")))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = Token2wav(str(MODEL_PATH), float16=False)
    yield model
    del model
    torch.cuda.empty_cache()


def _begin(model, prompt_wav: str, request_id: str, length: int = 16):
    stream_state = model.create_stream_state(prompt_wav)
    return model.begin_chunk_steps(
        _tokens(length),
        prompt_wav,
        stream_state,
        last_chunk=False,
        n_timesteps=N_TIMESTEPS,
        request_id=request_id,
        generation_id=7,
        sequence_no=3,
        version=11,
    )


def _frozen_control(model, prompt_wav: str, length: int = 16):
    stream_state = model.create_stream_state(prompt_wav)
    speaker = model.cache[prompt_wav][2]
    token = torch.tensor([_tokens(length)], dtype=torch.int32, device="cuda")
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        mel, cache = model.flow.inference_chunk(
            token=token,
            spk=speaker,
            cache=_clone_tree(stream_state["flow_cache"]),
            last_chunk=False,
            n_timesteps=N_TIMESTEPS,
        )
    return mel.detach().clone(), _clone_tree(cache)


def _single_public(model, prompt_wav: str, count: int, length: int = 16):
    states = tuple(_begin(model, prompt_wav, f"single-{index}", length) for index in range(count))
    return _finish_states(model, states)


def _finish_variable_single(model, state):
    values = (state,)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(N_TIMESTEPS):
            values = tuple(model.advance_chunk_step_variable_batch(values))
    return values, tuple(model.finish_chunk_steps(value) for value in values)


def _clone_state(state):
    from dataclasses import replace

    values = {}
    for key, value in state.__dict__.items():
        if isinstance(value, torch.Tensor):
            values[key] = value.detach().clone()
        elif isinstance(value, tuple):
            values[key] = tuple(
                item.detach().clone() if isinstance(item, torch.Tensor) else item
                for item in value
            )
    return replace(
        state,
        **{key: value for key, value in values.items() if key in state.__dataclass_fields__},
    )


def _run_prior_chunk(model, prompt_wav: str, length: int, request_id: str):
    stream = model.create_stream_state(prompt_wav)
    state = model.begin_chunk_steps(
        _tokens(length),
        prompt_wav,
        stream,
        last_chunk=False,
        n_timesteps=N_TIMESTEPS,
        request_id=request_id,
        generation_id=3,
        sequence_no=0,
        version=0,
    )
    _, finished = _finish_states(model, (state,))
    _, cache = finished[0]
    return cache


def test_b1_public_step_matches_frozen_control_and_pcm_contract(real_model):
    prompt_wav = str(PROMPT_WAV)
    frozen_mel, frozen_cache = _frozen_control(real_model, prompt_wav)
    states, finished = _single_public(real_model, prompt_wav, 1)
    public_state = states[0]
    public_mel, public_cache = finished[0]

    assert (public_state.request_id, public_state.generation_id, public_state.sequence_no, public_state.version) == (
        "single-0", 7, 3, 11
    )
    assert public_state.step_index == N_TIMESTEPS
    assert torch.equal(public_mel, frozen_mel)
    _assert_public_cache_matches_frozen(
        public_cache, frozen_cache, rtol=0.0, atol=0.0
    )
    assert len(public_state.input_att_cache) == N_TIMESTEPS
    assert max(cache.shape[-2] for cache in public_state.input_att_cache) < 1000
    assert public_state.logical_state_size_bytes() < 2 * 1024 * 1024 * 1024

    baseline_stream = real_model.create_stream_state(prompt_wav)
    baseline_pcm = real_model.stream_with_state(
        _tokens(), prompt_wav, baseline_stream, last_chunk=False
    )
    public_pcm = _pcm_from_mel(real_model, prompt_wav, public_mel, last_chunk=False)
    metrics = _pcm_metrics(baseline_pcm, public_pcm)
    assert metrics["same_length"]
    assert metrics["normalized_rmse"] <= 0.02
    assert metrics["correlation"] >= 0.99
    assert metrics["snr_db"] >= 34.0


def test_b1_variable_public_step_matches_frozen_control_and_pcm_contract(real_model):
    prompt_wav = str(PROMPT_WAV)
    frozen_mel, frozen_cache = _frozen_control(real_model, prompt_wav)
    state = _begin(real_model, prompt_wav, "variable-single")
    states, finished = _finish_variable_single(real_model, state)
    public_state = states[0]
    public_mel, public_cache = finished[0]

    assert public_state.request_id == "variable-single"
    assert torch.equal(public_mel, frozen_mel)
    _assert_public_cache_matches_frozen(public_cache, frozen_cache, rtol=0.0, atol=0.0)
    baseline_stream = real_model.create_stream_state(prompt_wav)
    baseline_pcm = real_model.stream_with_state(
        _tokens(), prompt_wav, baseline_stream, last_chunk=False
    )
    public_pcm = _pcm_from_mel(real_model, prompt_wav, public_mel, last_chunk=False)
    metrics = _pcm_metrics(baseline_pcm, public_pcm)
    assert metrics["same_length"]
    assert metrics["normalized_rmse"] <= 0.02
    assert metrics["correlation"] >= 0.99
    assert metrics["snr_db"] >= 34.0


@pytest.mark.parametrize("logical_batch", [2, 4])
def test_batched_step_matches_independent_logical_rows(real_model, logical_batch):
    prompt_wav = str(PROMPT_WAV)
    _, single_finished = _single_public(real_model, prompt_wav, logical_batch)
    batch_states = tuple(_begin(real_model, prompt_wav, f"batch-{index}") for index in range(logical_batch))
    batch_states, batch_finished = _finish_states(real_model, batch_states)

    assert len(batch_finished) == logical_batch
    for index, ((single_mel, single_cache), (batch_mel, batch_cache), state) in enumerate(
        zip(single_finished, batch_finished, batch_states)
    ):
        assert state.request_id == f"batch-{index}"
        assert state.generation_id == 7
        assert state.sequence_no == 3
        assert state.version == 11
        assert torch.equal(batch_mel, single_mel)
        _assert_tree_close(batch_cache, single_cache, rtol=1e-4, atol=1e-5)
        single_pcm = _pcm_from_mel(real_model, prompt_wav, single_mel, last_chunk=False)
        batch_pcm = _pcm_from_mel(real_model, prompt_wav, batch_mel, last_chunk=False)
        metrics = _pcm_metrics(single_pcm, batch_pcm)
        assert metrics["same_length"]
        assert metrics["normalized_rmse"] <= 0.02
        assert metrics["correlation"] >= 0.99
        assert metrics["snr_db"] >= 34.0


def test_b1_public_steps_match_frozen_control_across_two_chunks(real_model):
    prompt_wav = str(PROMPT_WAV)
    control_stream = real_model.create_stream_state(prompt_wav)
    public_stream = real_model.create_stream_state(prompt_wav)
    speaker = real_model.cache[prompt_wav][2]
    control_stream["flow_cache"] = _clone_tree(control_stream["flow_cache"])
    public_stream["flow_cache"] = _clone_tree(public_stream["flow_cache"])

    for sequence_no, length in enumerate((16, 20)):
        token = torch.tensor(
            [_tokens(length)], dtype=torch.int32, device="cuda"
        )
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float32
        ):
            control_mel, control_cache = real_model.flow.inference_chunk(
                token=token,
                spk=speaker,
                cache=control_stream["flow_cache"],
                last_chunk=False,
                n_timesteps=N_TIMESTEPS,
            )
        control_stream["flow_cache"] = _clone_tree(control_cache)
        control_pcm = real_model.render_chunk_pcm(
            control_mel, control_stream, last_chunk=False
        )

        step_state = real_model.begin_chunk_steps(
            _tokens(length),
            prompt_wav,
            public_stream,
            last_chunk=False,
            n_timesteps=N_TIMESTEPS,
            request_id="multichunk",
            generation_id=9,
            sequence_no=sequence_no,
            version=sequence_no,
        )
        _, finished = _finish_states(real_model, (step_state,))
        public_mel, public_cache = finished[0]
        public_stream["flow_cache"].update(public_cache)
        public_pcm = real_model.render_chunk_pcm(
            public_mel, public_stream, last_chunk=False
        )

        torch.testing.assert_close(
            public_mel, control_mel, rtol=1e-4, atol=1e-5
        )
        _assert_public_cache_matches_frozen(
            public_cache,
            control_stream["flow_cache"],
            rtol=1e-4,
            atol=1e-5,
        )
        metrics = _pcm_metrics(control_pcm, public_pcm)
        assert metrics["same_length"]
        assert metrics["normalized_rmse"] <= 0.02
        assert metrics["correlation"] >= 0.99
        assert metrics["snr_db"] >= 34.0


def test_variable_b2_unequal_attention_cache_matches_independent_rows(real_model):
    prompt_wav = str(PROMPT_WAV)
    first_cache = _run_prior_chunk(real_model, prompt_wav, 12, "prior-a")
    second_cache = _run_prior_chunk(real_model, prompt_wav, 20, "prior-b")
    first_stream = real_model.create_stream_state(prompt_wav)
    second_stream = real_model.create_stream_state(prompt_wav)
    first_stream["flow_cache"].update(_clone_tree(first_cache))
    second_stream["flow_cache"].update(_clone_tree(second_cache))

    first = real_model.begin_chunk_steps(
        _tokens(16), prompt_wav, first_stream, request_id="var-a", generation_id=11
    )
    second = real_model.begin_chunk_steps(
        _tokens(16), prompt_wav, second_stream, request_id="var-b", generation_id=29
    )
    assert first.input_att_cache[0].shape[3] != second.input_att_cache[0].shape[3]

    independent = (_clone_state(first), _clone_state(second))
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        independent = tuple(real_model.advance_chunk_step(value) for value in independent)
        variable = tuple(real_model.advance_chunk_step_variable_batch((first, second)))
        for _ in range(N_TIMESTEPS - 1):
            independent = tuple(real_model.advance_chunk_step(value) for value in independent)
            variable = tuple(real_model.advance_chunk_step_variable_batch(variable))
    finished_independent = tuple(real_model.finish_chunk_steps(value) for value in independent)
    finished_variable = tuple(real_model.finish_chunk_steps(value) for value in variable)

    for expected, actual, expected_finished, actual_finished in zip(
        independent, variable, finished_independent, finished_variable
    ):
        torch.testing.assert_close(actual.x, expected.x, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            actual.completed_att_cache,
            expected.completed_att_cache,
            rtol=1e-4,
            atol=2e-5,
        )
        assert actual.completed_att_cache.shape[3] == expected.completed_att_cache.shape[3]
        metrics = _pcm_metrics(
            _pcm_from_mel(real_model, prompt_wav, expected_finished[0], last_chunk=False),
            _pcm_from_mel(real_model, prompt_wav, actual_finished[0], last_chunk=False),
        )
        assert metrics["same_length"]
        assert metrics["normalized_rmse"] <= 0.02
        assert metrics["correlation"] >= 0.99
        assert metrics["snr_db"] >= 34.0


def test_mixed_step_b2_different_euler_positions_match_independent_rows(real_model):
    """A mixed-step call must be equivalent to two independent B=1 calls."""
    prompt_wav = str(PROMPT_WAV)
    first_cache = _run_prior_chunk(real_model, prompt_wav, 12, "mixed-prior-a")
    second_cache = _run_prior_chunk(real_model, prompt_wav, 20, "mixed-prior-b")
    first_stream = real_model.create_stream_state(prompt_wav)
    second_stream = real_model.create_stream_state(prompt_wav)
    first_stream["flow_cache"].update(_clone_tree(first_cache))
    second_stream["flow_cache"].update(_clone_tree(second_cache))
    first = real_model.begin_chunk_steps(
        _tokens(16), prompt_wav, first_stream, request_id="mixed-a", generation_id=41
    )
    second = real_model.begin_chunk_steps(
        _tokens(16), prompt_wav, second_stream, request_id="mixed-b", generation_id=73
    )

    # Put the rows at different Euler positions without changing their
    # current chunk shape or t_span.  This is the only compatibility dimension
    # relaxed by the exploratory mixed-step API.
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(2):
            first = real_model.advance_chunk_step(first)
        for _ in range(4):
            second = real_model.advance_chunk_step(second)
    assert first.step_index == 2
    assert second.step_index == 4
    assert first.input_att_cache[first.step_index].shape[3] != second.input_att_cache[second.step_index].shape[3]

    independent = (_clone_state(first), _clone_state(second))
    mixed_inputs = (_clone_state(first), _clone_state(second))
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        independent = tuple(real_model.advance_chunk_step(value) for value in independent)
        mixed = tuple(
            real_model.advance_chunk_step_variable_mixed_batch(mixed_inputs)
        )

        # Finish both trajectories with the frozen B=1 method.  This isolates
        # the mixed transition while still checking final mel/PCM semantics.
        while any(value.step_index < N_TIMESTEPS for value in independent):
            independent = tuple(
                real_model.advance_chunk_step(value)
                if value.step_index < N_TIMESTEPS
                else value
                for value in independent
            )
        while any(value.step_index < N_TIMESTEPS for value in mixed):
            mixed = tuple(
                real_model.advance_chunk_step(value)
                if value.step_index < N_TIMESTEPS
                else value
                for value in mixed
            )
    for expected, actual in zip(independent, mixed):
        assert actual.step_index == expected.step_index == N_TIMESTEPS
        torch.testing.assert_close(actual.x, expected.x, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            actual.completed_att_cache,
            expected.completed_att_cache,
            rtol=1e-4,
            atol=2e-5,
        )
        expected_mel, _ = real_model.finish_chunk_steps(expected)
        actual_mel, _ = real_model.finish_chunk_steps(actual)
        torch.testing.assert_close(actual_mel, expected_mel, rtol=1e-4, atol=1e-5)
        expected_pcm = _pcm_from_mel(
            real_model, prompt_wav, expected_mel, last_chunk=False
        )
        actual_pcm = _pcm_from_mel(
            real_model, prompt_wav, actual_mel, last_chunk=False
        )
        metrics = _pcm_metrics(expected_pcm, actual_pcm)
        assert metrics["same_length"]
        assert metrics["normalized_rmse"] <= 0.02
        assert metrics["correlation"] >= 0.99
        assert metrics["snr_db"] >= 34.0
