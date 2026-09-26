"""Diagnostic-only study of RNG and CUDA nondeterminism in local Token2Wav."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from lychee_fd.runtime.acoustic_equivalence import compare_pcm_equivalence
from lychee_fd.runtime.token2wav_checkpoint import Token2WavCheckpoint
from token2wav import Token2wav


MODEL_PATH = os.environ.get("LYCHEEFD_REAL_T2W_MODEL", "/mnt/DuplexPilot/data/models/token2wav")
PROMPT_WAV = os.environ.get(
    "LYCHEEFD_REAL_T2W_PROMPT",
    "/mnt/DuplexPilot/data/DuplexPilot/lychee_dsv_closure_worktree/frontend/public/clone_24k_mono/default_male.wav",
)


def _capture_rng():
    return (
        torch.get_rng_state().clone(),
        tuple(state.clone() for state in torch.cuda.get_rng_state_all()),
    )


def _restore_rng(snapshot):
    cpu_state, cuda_states = snapshot
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(list(cuda_states))


def _metrics(reference, migrated):
    result = compare_pcm_equivalence(reference, migrated, 24000, 24000)
    left = np.frombuffer(reference, dtype="<i2").astype(np.int32)
    right = np.frombuffer(migrated, dtype="<i2").astype(np.int32)
    return {
        "sample_count": result.sample_count,
        "migrated_sample_count": result.migrated_sample_count,
        "finite": result.finite,
        "bitwise_equal": result.bitwise_equal,
        "normalized_rmse": result.normalized_rmse,
        "correlation": result.correlation,
        "snr_db": result.snr_db,
        "differing_samples": int(np.count_nonzero(left != right)),
        "max_abs_int16": int(np.max(np.abs(left - right))),
    }


def main():
    model = Token2wav(MODEL_PATH, float16=False)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        tokens = ((1493, 4299, 4218, 2049, 528, 2752, 4850, 4569) * 8)[:50]
        split = 25
        state = model.create_stream_state(PROMPT_WAV)
        model.stream_with_state(tokens[:split], PROMPT_WAV, state, last_chunk=False)
        checkpoint = Token2WavCheckpoint.capture(
            request_id="nondeterminism-study",
            stream_id="nondeterminism-stream",
            generation_id=1,
            prompt_wav=PROMPT_WAV,
            stream_state=state,
        )
        tail_rng = _capture_rng()

        _restore_rng(tail_rng)
        continuous_pcm = model.stream_with_state(
            tokens[split:], PROMPT_WAV, state, last_chunk=True
        )
        restored_state = checkpoint.restore(
            request_id="nondeterminism-study",
            stream_id="nondeterminism-stream",
        )
        _restore_rng(tail_rng)
        migrated_same_rng_pcm = model.stream_with_state(
            tokens[split:], PROMPT_WAV, restored_state, last_chunk=True
        )

        no_restore_state = checkpoint.restore(
            request_id="nondeterminism-study",
            stream_id="nondeterminism-stream",
        )
        uncontrolled_pcm = model.stream_with_state(
            tokens[split:], PROMPT_WAV, no_restore_state, last_chunk=True
        )
    finally:
        torch.use_deterministic_algorithms(False)

    print(json.dumps({
        "same_rng_continuation": _metrics(continuous_pcm, migrated_same_rng_pcm),
        "uncontrolled_rng_continuation": _metrics(continuous_pcm, uncontrolled_pcm),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
