"""Red tests for the eager state boundary around compiled stateless Flow work."""

import torch

from lychee_fd.runtime.apr.flow_inductor_runtime import (
    FlowInductorRuntime,
    InductorCompileSpec,
)


def test_compiled_callable_receives_only_tensor_arguments():
    seen = []

    def operator(x, t, mask):
        seen.append((x, t, mask))
        return x + t + mask.to(x.dtype)

    runtime = FlowInductorRuntime(
        operator,
        config=InductorCompileSpec(enabled=True, target="mlp_only"),
        compiler=lambda fn, **kwargs: fn,
    )
    runtime.register_shape((2, 8))
    result = runtime(
        torch.zeros(2, 8),
        torch.ones(2, 8),
        torch.ones(2, 8, dtype=torch.bool),
        shape_key=(2, 8),
    )
    assert len(seen) == 1
    assert result.shape == (2, 8)


def test_b1_and_b2_have_distinct_compile_keys():
    runtime = FlowInductorRuntime(
        lambda x: x + 1,
        config=InductorCompileSpec(enabled=True, target="mlp_only"),
        compiler=lambda fn, **kwargs: fn,
    )
    assert runtime.compile_key((1, 8), logical_batch_size=1) != runtime.compile_key(
        (1, 8), logical_batch_size=2
    )

