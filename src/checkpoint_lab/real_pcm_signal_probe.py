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


def main():
    model = Token2wav(MODEL_PATH, float16=False)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        tokens = ((1493, 4299, 4218, 2049, 528, 2752, 4850, 4569) * 8)[:50]
        split = 25
        continuous_state = model.create_stream_state(PROMPT_WAV)
        migrated_state = model.create_stream_state(PROMPT_WAV)
        model.stream_with_state(tokens[:split], PROMPT_WAV, continuous_state, last_chunk=False)
        model.stream_with_state(tokens[:split], PROMPT_WAV, migrated_state, last_chunk=False)
        checkpoint = Token2WavCheckpoint.capture(
            request_id="signal-probe",
            stream_id="signal-stream",
            generation_id=1,
            prompt_wav=PROMPT_WAV,
            stream_state=migrated_state,
        )
        restored_state = checkpoint.restore(
            request_id="signal-probe", stream_id="signal-stream"
        )
        continuous_pcm = model.stream_with_state(
            tokens[split:], PROMPT_WAV, continuous_state, last_chunk=True
        )
        migrated_pcm = model.stream_with_state(
            tokens[split:], PROMPT_WAV, restored_state, last_chunk=True
        )
    finally:
        torch.use_deterministic_algorithms(False)

    result = compare_pcm_equivalence(continuous_pcm, migrated_pcm, 24000, 24000)
    reference = np.frombuffer(continuous_pcm, dtype="<i2").astype(np.int32)
    migrated = np.frombuffer(migrated_pcm, dtype="<i2").astype(np.int32)
    print(json.dumps({
        "sample_count": result.sample_count,
        "migrated_sample_count": result.migrated_sample_count,
        "same_sample_rate": result.same_sample_rate,
        "same_signal_length": result.same_signal_length,
        "finite": result.finite,
        "bitwise_equal": result.bitwise_equal,
        "normalized_rmse": result.normalized_rmse,
        "correlation": result.correlation,
        "snr_db": result.snr_db,
        "failure_reasons": result.failure_reasons,
        "passed": result.passed,
        "max_abs_int16": int(np.max(np.abs(reference - migrated))),
        "differing_samples": int(np.count_nonzero(reference != migrated)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
