from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

torch.cuda.set_device(int(os.environ.get("LYCHEEFD_TOKEN2WAV_DEVICE", "1")))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "third_party" / "Step-Audio2"))
from token2wav import Token2wav


def report(label: str):
    torch.cuda.synchronize()
    print(
        label,
        "allocated_mib=", round(torch.cuda.memory_allocated(1) / 2**20, 2),
        "reserved_mib=", round(torch.cuda.memory_reserved(1) / 2**20, 2),
        "peak_allocated_mib=", round(torch.cuda.max_memory_allocated(1) / 2**20, 2),
        "peak_reserved_mib=", round(torch.cuda.max_memory_reserved(1) / 2**20, 2),
        flush=True,
    )


model = Token2wav("/mnt/DuplexPilot/data/models/token2wav", float16=False)
report("after_model")
prompt = "/mnt/DuplexPilot/data/DuplexPilot/lychee_dsv_closure_worktree/frontend/public/clone_24k_mono/default_male.wav"
state = model.create_stream_state(prompt)
report("after_stream")
tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569] * 2
step = model.begin_chunk_steps(tokens, prompt, state, last_chunk=False, n_timesteps=10)
report("after_begin")
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
    for index in range(10):
        step = model.advance_chunk_step(step)
        report(f"after_step_{index}")
finished = model.finish_chunk_steps(step)
report("after_finish")
_ = model.render_chunk_pcm(
    finished[0], model.create_stream_state(prompt), last_chunk=False
)
report("after_render_chunk_pcm")

# Unequal-cache physical B=2 diagnostic.
def prior(length, request_id):
    local_state = model.create_stream_state(prompt)
    local_step = model.begin_chunk_steps(tokens[:length], prompt, local_state, last_chunk=False, n_timesteps=10, request_id=request_id)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        for _ in range(10):
            local_step = model.advance_chunk_step(local_step)
    return model.finish_chunk_steps(local_step)[1]

first_cache = prior(12, "prior-a")
report("after_prior_a")
second_cache = prior(20, "prior-b")
report("after_prior_b")
first_stream = model.create_stream_state(prompt)
second_stream = model.create_stream_state(prompt)
first_stream["flow_cache"].update(first_cache)
second_stream["flow_cache"].update(second_cache)
first = model.begin_chunk_steps(tokens[:16], prompt, first_stream, last_chunk=False, n_timesteps=10, request_id="var-a")
second = model.begin_chunk_steps(tokens[:16], prompt, second_stream, last_chunk=False, n_timesteps=10, request_id="var-b")
report("after_b2_begin")
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
    for index in range(10):
        first, second = model.advance_chunk_step_variable_batch((first, second))
        report(f"after_b2_step_{index}")
