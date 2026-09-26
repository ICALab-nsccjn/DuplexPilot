import torch

from token2wav import Token2wav


prompt = "/mnt/DuplexPilot/data/DuplexPilot/worktrees/graph-b2-causal-closure/frontend/public/clone_24k_mono/default_male.wav"
model = Token2wav("/mnt/DuplexPilot/data/models/token2wav", float16=False)
print(
    "attrs",
    getattr(model, "stream_lookahead_len", None),
    getattr(model.flow, "pre_lookahead_len", None),
    getattr(model.flow, "up_rate", None),
)
for length in (30, 48, 96):
    stream = model.create_stream_state(prompt)
    state = model.begin_chunk_steps(
        list(range(length)),
        prompt,
        stream,
        last_chunk=False,
        n_timesteps=10,
        request_id="inspect",
    )
    def shape(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
        return type(value).__name__
    print(
        "input_tokens",
        length,
        "x",
        shape(state.x),
        "mu",
        shape(state.mu),
        "condition",
        shape(state.condition),
        "cnn",
        shape(state.input_cnn_cache),
        "att",
        shape(state.input_att_cache),
    )
