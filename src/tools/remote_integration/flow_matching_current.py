# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, replace
from typing import Any, List, Sequence
import onnxruntime
import torch
import torch.nn.functional as F

from cosyvoice2.flow.decoder_dit import DiT
from cosyvoice2.utils.mask import make_pad_mask


def _logical_tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        marker = id(value)
        if marker in seen:
            return 0
        seen.add(marker)
        return int(value.numel() * value.element_size())
    if isinstance(value, (tuple, list)):
        return sum(_logical_tensor_bytes(item, seen) for item in value)
    if isinstance(value, dict):
        return sum(_logical_tensor_bytes(item, seen) for item in value.values())
    return 0


@dataclass
class CausalCFMStepState:
    """Request-owned state for one resumable causal Flow chunk.

    The cache history contains only the per-Euler-step outputs needed by the
    next chunk.  Model weights, module scratch buffers, and temporary CFG
    tensors are deliberately not fields of this object.
    """

    x: torch.Tensor
    t: torch.Tensor
    dt: torch.Tensor
    step_index: int
    t_span: torch.Tensor
    mu: torch.Tensor
    speaker: torch.Tensor
    condition: torch.Tensor
    input_cnn_cache: Any
    input_att_cache: Any
    completed_cnn_cache: torch.Tensor | None = None
    completed_att_cache: torch.Tensor | None = None
    conformer_cnn_cache: torch.Tensor | None = None
    conformer_att_cache: torch.Tensor | None = None
    request_id: str = ""
    generation_id: int = 0
    sequence_no: int = 0
    version: int = 0
    _cnn_history: tuple[torch.Tensor | None, ...] = ()
    _att_history: tuple[torch.Tensor | None, ...] = ()
    _cache_capacity: int = 0

    def logical_state_size_bytes(self) -> int:
        """Return bytes represented by logical fields only."""
        return _logical_tensor_bytes(
            (
                self.x, self.t, self.dt, self.t_span,
                self.mu, self.speaker, self.condition,
                self.input_cnn_cache, self.input_att_cache,
                self.completed_cnn_cache, self.completed_att_cache,
                self.conformer_cnn_cache, self.conformer_att_cache,
                self._cnn_history, self._att_history,
            )
        )

    def logical_state_dict(self) -> dict[str, Any]:
        """Return an explicit checkpoint payload without worker-local data."""
        return {
            "x": self.x.detach().clone(),
            "t": self.t.detach().clone(),
            "dt": self.dt.detach().clone(),
            "step_index": int(self.step_index),
            "t_span": self.t_span.detach().clone(),
            "mu": self.mu.detach().clone(),
            "speaker": self.speaker.detach().clone(),
            "condition": self.condition.detach().clone(),
            "input_cnn_cache": self.input_cnn_cache,
            "input_att_cache": self.input_att_cache,
            "completed_cnn_cache": self.completed_cnn_cache,
            "completed_att_cache": self.completed_att_cache,
            "conformer_cnn_cache": self.conformer_cnn_cache,
            "conformer_att_cache": self.conformer_att_cache,
            "request_id": self.request_id,
            "generation_id": int(self.generation_id),
            "sequence_no": int(self.sequence_no),
            "version": int(self.version),
            "cnn_history": self._cnn_history,
            "att_history": self._att_history,
            "cache_capacity": int(self._cache_capacity),
        }


"""
Inference wrapper
"""
class CausalConditionalCFM(torch.nn.Module):
    def __init__(self, estimator: DiT, inference_cfg_rate:float=0.7):
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
         # a maximum of 600s
        self.register_buffer('rand_noise', torch.randn([1, self.out_channels, 50 * 600]), persistent=False)

        self.register_buffer('cnn_cache_buffer', torch.zeros(16, 16, 2, 1024, 2), persistent=False)
        self.register_buffer('att_cache_buffer', torch.zeros(16, 16, 2, 8, 1000, 128), persistent=False)

    @staticmethod
    def _cache_step(cache: Any, step_index: int) -> Any:
        if cache is None:
            return None
        if isinstance(cache, (tuple, list)):
            return cache[step_index]
        if isinstance(cache, torch.Tensor):
            if cache.shape[0] <= step_index:
                raise ValueError("Flow cache has fewer steps than t_span")
            return cache[step_index]
        raise TypeError("Flow cache must be a tensor, tuple, list, or None")

    @staticmethod
    def _cache_prefix(cache: Any, steps: int) -> Any:
        """Keep only the logical timestep prefix; capacity remains worker-local."""
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            if cache.shape[0] < steps:
                raise ValueError("Flow cache has fewer steps than t_span")
            return cache[:steps].detach().clone()
        if isinstance(cache, (tuple, list)):
            if len(cache) < steps:
                raise ValueError("Flow cache has fewer steps than t_span")
            return type(cache)(item.detach().clone() for item in cache[:steps])
        raise TypeError("Flow cache must be a tensor, tuple, list, or None")

    @staticmethod
    def _cache_history(cache: Any, steps: int) -> tuple[torch.Tensor | None, ...]:
        """Create independently releasable request-owned Euler cache slots."""
        if cache is None:
            return (None,) * steps
        if isinstance(cache, torch.Tensor):
            if cache.shape[0] < steps:
                raise ValueError("Flow cache has fewer steps than t_span")
            return tuple(cache[index].detach().clone() for index in range(steps))
        if isinstance(cache, (tuple, list)):
            if len(cache) < steps:
                raise ValueError("Flow cache has fewer steps than t_span")
            return tuple(
                item.detach() if isinstance(item, torch.Tensor) else item
                for item in cache[:steps]
            )
        raise TypeError("Flow cache must be a tensor, tuple, list, or None")

    @staticmethod
    def _cache_capacity(cache: Any, fallback: int) -> int:
        if isinstance(cache, torch.Tensor):
            return int(cache.shape[0])
        return len(cache) if isinstance(cache, (tuple, list)) else int(fallback)

    @staticmethod
    def _join_step_caches(caches: Sequence[Any], *, name: str) -> Any:
        if not caches or all(cache is None for cache in caches):
            return None
        if any(cache is None for cache in caches):
            raise ValueError(f"{name} cache presence differs across logical states")
        tensors = tuple(caches)
        if any(
            not isinstance(cache, torch.Tensor) or cache.ndim < 2
            for cache in tensors
        ):
            raise ValueError(f"{name} cache must be a tensor with a batch dimension")
        row_counts = tuple(int(cache.shape[1]) for cache in tensors)
        if any(rows <= 0 or rows % 2 for rows in row_counts):
            raise ValueError(
                f"{name} cache batch dimension must contain conditional/unconditional pairs"
            )
        logical_rows = tuple(rows // 2 for rows in row_counts)
        # Each request-owned cache is stored as [conditional, unconditional],
        # while the estimator consumes CFG rows as [all conditional, all
        # unconditional]. Reorder at the physical boundary so cache rows
        # follow the same contract as cfg_x/cfg_mu/cfg_t below.
        conditional = tuple(
            cache[:, :logical_size]
            for cache, logical_size in zip(tensors, logical_rows)
        )
        unconditional = tuple(
            cache[:, logical_size:]
            for cache, logical_size in zip(tensors, logical_rows)
        )
        return torch.cat((*conditional, *unconditional), dim=1)

    def begin_chunk_steps(
        self,
        *,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: Any = None,
        att_cache: Any = None,
        request_id: str = "",
        generation_id: int = 0,
        sequence_no: int = 0,
        version: int = 0,
    ) -> CausalCFMStepState:
        """Create explicit logical state at the first Euler step."""
        if t_span.ndim != 1 or t_span.numel() < 2:
            raise ValueError("t_span must contain at least two time points")
        steps = int(t_span.numel() - 1)
        capacity = self._cache_capacity(cnn_cache, self.cnn_cache_buffer.shape[0])
        logical_cnn_cache = self._cache_history(cnn_cache, steps)
        logical_att_cache = self._cache_history(att_cache, steps)
        return CausalCFMStepState(
            x=x.detach().clone(),
            t=t_span[0].expand(x.shape[0]).clone(),
            dt=(t_span[1] - t_span[0]).clone(),
            step_index=0,
            t_span=t_span.detach().clone(),
            mu=mu.detach().clone(),
            speaker=spks.detach().clone(),
            condition=cond.detach().clone(),
            input_cnn_cache=logical_cnn_cache,
            input_att_cache=logical_att_cache,
            request_id=str(request_id),
            generation_id=int(generation_id),
            sequence_no=int(sequence_no),
            version=int(version),
            _cnn_history=logical_cnn_cache,
            _att_history=logical_att_cache,
            _cache_capacity=capacity,
        )

    def begin_chunk_steps_from_conditions(
        self,
        *,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        cnn_cache: Any = None,
        att_cache: Any = None,
        request_id: str = "",
        generation_id: int = 0,
        sequence_no: int = 0,
        version: int = 0,
    ) -> CausalCFMStepState:
        """Prepare a state from Flow conditions using safe batch noise expansion."""
        if isinstance(n_timesteps, bool) or not isinstance(n_timesteps, int) or n_timesteps <= 0:
            raise ValueError("n_timesteps must be a positive integer")
        if isinstance(att_cache, torch.Tensor):
            offset = int(att_cache.shape[4])
        elif isinstance(att_cache, (tuple, list)):
            first_cache = next((item for item in att_cache if item is not None), None)
            offset = int(first_cache.shape[3]) if first_cache is not None else 0
        else:
            offset = 0
        x = self.rand_noise[:, :, offset:offset + mu.size(2)] * temperature
        if x.shape[0] == 1 and mu.shape[0] > 1:
            x = x.expand(mu.shape[0], -1, -1).contiguous()
        t_span = torch.linspace(
            0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype
        )
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.begin_chunk_steps(
            x=x,
            t_span=t_span,
            mu=mu,
            spks=spks,
            cond=cond,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
            request_id=request_id,
            generation_id=generation_id,
            sequence_no=sequence_no,
            version=version,
        )

    @staticmethod
    def _variable_current_cache(
        state: CausalCFMStepState, kind: str, step_index: int
    ) -> Any:
        """Return one request's cache slot without exposing it to callers."""
        history = getattr(state, f"_{kind}_history", ())
        if history:
            return CausalConditionalCFM._cache_step(history, step_index)
        return CausalConditionalCFM._cache_step(
            getattr(state, f"input_{kind}_cache", None), step_index
        )

    @staticmethod
    def _variable_history(
        state: CausalCFMStepState, kind: str, steps: int
    ) -> list[torch.Tensor | None]:
        history = getattr(state, f"_{kind}_history", ())
        if not history:
            history = getattr(state, f"input_{kind}_cache", None)
        if history is None:
            return [None] * steps
        if isinstance(history, torch.Tensor):
            if history.shape[0] < steps:
                raise ValueError(f"{kind} cache has fewer steps than t_span")
            return [history[index] for index in range(steps)]
        if isinstance(history, (tuple, list)):
            if len(history) < steps:
                raise ValueError(f"{kind} cache has fewer steps than t_span")
            return list(history[:steps])
        raise TypeError(f"{kind} cache must be a tensor, tuple, list, or None")

    @staticmethod
    def _variable_row_cache(
        cache: torch.Tensor, logical_row: int, logical_batch: int, end: int
    ) -> torch.Tensor:
        """Restore one logical row in conditional/unconditional order."""
        conditional = cache[:, logical_row:logical_row + 1, ..., :end, :]
        unconditional = cache[
            :, logical_batch + logical_row:logical_batch + logical_row + 1, ..., :end, :
        ]
        return torch.cat((conditional, unconditional), dim=1).detach().clone()

    def _validate_variable_states(
        self,
        states: tuple[CausalCFMStepState, ...],
        *,
        allow_mixed_step: bool = False,
        allowed_batch_sizes: tuple[int, ...] = (2,),
    ) -> tuple[
        tuple[int, ...],
        int,
        tuple[torch.Tensor | None, ...],
        tuple[torch.Tensor | None, ...],
        tuple[int, ...],
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Validate and pack metadata needed by the variable-length path.

        The default contract is deliberately limited to two logical singleton
        requests.  An explicitly named exploratory caller may pass a larger
        allow-list (currently three or four) without changing the default
        same-step or B=2 contracts.  All checks happen before the estimator is
        called.
        """
        if not allowed_batch_sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in allowed_batch_sizes
        ):
            raise ValueError("variable Flow allowed batch sizes are invalid")
        if len(states) not in allowed_batch_sizes:
            if len(allowed_batch_sizes) == 1:
                message = f"variable-length Flow V1 requires exactly {allowed_batch_sizes[0]} states"
            else:
                message = (
                    "variable-length Flow requires logical batch size in "
                    + ", ".join(str(size) for size in allowed_batch_sizes)
                )
            raise ValueError(message)
        if any(not isinstance(state, CausalCFMStepState) for state in states):
            raise TypeError("variable Flow batch contains a non-Flow state")

        request_ids = [str(state.request_id) for state in states]
        if any(not value for value in request_ids):
            raise ValueError("variable Flow request_id must be non-empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("variable Flow batch contains duplicate request_id")
        for state in states:
            for name in ("generation_id", "version", "sequence_no"):
                value = getattr(state, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid variable Flow {name}")

        step_indices = tuple(int(state.step_index) for state in states)
        if any(
            isinstance(state.step_index, bool)
            or not isinstance(state.step_index, int)
            or state.step_index < 0
            for state in states
        ):
            raise ValueError("invalid variable Flow step_index")
        if not allow_mixed_step and len(set(step_indices)) != 1:
            raise ValueError("variable Flow step_index mismatch")

        first_span = states[0].t_span
        if not isinstance(first_span, torch.Tensor) or first_span.ndim != 1:
            raise ValueError("variable Flow t_span must be one-dimensional")
        if first_span.numel() < 2:
            raise ValueError("variable Flow state has no remaining Euler step")
        for state, state_step in zip(states, step_indices):
            if state_step >= first_span.numel() - 1:
                raise ValueError("variable Flow state has no remaining Euler step")
        for state in states[1:]:
            if (
                not isinstance(state.t_span, torch.Tensor)
                or state.t_span.device != first_span.device
                or state.t_span.dtype != first_span.dtype
                or state.t_span.shape != first_span.shape
                or not torch.equal(state.t_span, first_span)
            ):
                raise ValueError("variable Flow states must share t_span")

        # last_chunk is chunk metadata in the frozen state contract.  If a
        # caller supplies it as an optional dynamic attribute, validate it;
        # absence on both states keeps the frozen dataclass unchanged.
        last_chunk_values = [getattr(state, "last_chunk", None) for state in states]
        if any(value is not None for value in last_chunk_values):
            if any(value is None for value in last_chunk_values):
                raise ValueError("variable Flow last_chunk metadata is incomplete")
            if len({bool(value) for value in last_chunk_values}) != 1:
                raise ValueError("variable Flow last_chunk mismatch")

        for name in ("x", "mu", "condition", "speaker"):
            reference = getattr(states[0], name)
            if not isinstance(reference, torch.Tensor) or reference.ndim < 2:
                raise ValueError(f"variable Flow {name} must be a tensor")
            if int(reference.shape[0]) != 1:
                raise ValueError("variable Flow states must have logical batch size one")
            for state in states[1:]:
                candidate = getattr(state, name)
                if (
                    name in ("x", "mu", "condition")
                    and isinstance(candidate, torch.Tensor)
                    and candidate.ndim == reference.ndim
                    and candidate.shape[:-1] == reference.shape[:-1]
                    and candidate.shape[-1] != reference.shape[-1]
                ):
                    raise ValueError("variable Flow current chunk length mismatch")
                if (
                    not isinstance(candidate, torch.Tensor)
                    or candidate.shape != reference.shape
                    or candidate.dtype != reference.dtype
                    or candidate.device != reference.device
                ):
                    raise ValueError(f"variable Flow {name} metadata mismatch")
        for name in ("t", "dt"):
            reference = getattr(states[0], name)
            if not isinstance(reference, torch.Tensor):
                raise ValueError(f"variable Flow {name} must be a tensor")
            for state in states[1:]:
                candidate = getattr(state, name)
                if (
                    not isinstance(candidate, torch.Tensor)
                    or candidate.shape != reference.shape
                    or candidate.dtype != reference.dtype
                    or candidate.device != reference.device
                ):
                    raise ValueError(f"variable Flow {name} metadata mismatch")

        # A mixed-step call is valid only when each state carries the time
        # point and interval belonging to its own Euler position.  This makes
        # removing the step-index equality check fail closed instead of
        # accidentally applying one row's dt to another row.
        for state, state_step in zip(states, step_indices):
            expected_t = first_span[state_step].reshape_as(state.t)
            expected_dt = (
                first_span[state_step + 1] - first_span[state_step]
            ).reshape_as(state.dt)
            if not torch.equal(state.t, expected_t) or not torch.equal(
                state.dt, expected_dt
            ):
                raise ValueError("variable Flow t/dt does not match step_index")

        current_length = int(states[0].x.shape[-1])
        if any(int(state.x.shape[-1]) != current_length for state in states[1:]):
            raise ValueError("variable Flow current chunk length mismatch")

        cnn_caches = tuple(
            self._variable_current_cache(state, "cnn", state_step)
            for state, state_step in zip(states, step_indices)
        )
        if any(cache is None for cache in cnn_caches) and any(
            cache is not None for cache in cnn_caches
        ):
            raise ValueError("variable Flow CNN cache presence mismatch")
        cnn_physical: torch.Tensor | None = None
        if all(cache is not None for cache in cnn_caches):
            reference = cnn_caches[0]
            assert reference is not None
            if reference.ndim != 4 or int(reference.shape[1]) != 2:
                raise ValueError("variable Flow CNN cache must have shape [depth, 2, channels, history]")
            for row, cache in enumerate(cnn_caches[1:], start=1):
                assert cache is not None
                if (
                    cache.shape != reference.shape
                    or cache.dtype != reference.dtype
                    or cache.device != reference.device
                ):
                    raise ValueError(f"variable Flow CNN cache shape mismatch at row {row}")
            conditional = torch.cat(
                [cache[:, 0:1] for cache in cnn_caches if cache is not None], dim=1
            )
            unconditional = torch.cat(
                [cache[:, 1:2] for cache in cnn_caches if cache is not None], dim=1
            )
            cnn_physical = torch.cat((conditional, unconditional), dim=1)

        att_caches = tuple(
            self._variable_current_cache(state, "att", state_step)
            for state, state_step in zip(states, step_indices)
        )
        nonempty_att = [cache for cache in att_caches if cache is not None]
        att_physical: torch.Tensor | None = None
        attention_lengths = (0,) * len(states)
        if nonempty_att:
            reference = nonempty_att[0]
            if reference.ndim != 5 or int(reference.shape[1]) != 2:
                raise ValueError(
                    "variable Flow attention cache must have shape [depth, 2, heads, time, kv]"
                )
            for cache in nonempty_att:
                if (
                    cache.ndim != 5
                    or int(cache.shape[1]) != 2
                    or int(cache.shape[0]) != int(reference.shape[0])
                    or int(cache.shape[2]) != int(reference.shape[2])
                    or int(cache.shape[4]) != int(reference.shape[4])
                    or cache.dtype != reference.dtype
                    or cache.device != reference.device
                ):
                    raise ValueError("variable Flow attention cache non-time shape mismatch")
            # The DiT block API receives one mask shared by all layers.  Real
            # causal Flow caches have one common time length per request; fail
            # closed rather than silently applying a layer-inaccurate mask.
            row_lengths: list[int] = []
            for cache in att_caches:
                if cache is None:
                    row_lengths.append(0)
                    continue
                layer_lengths = [int(cache.shape[3])]
                if len(set(layer_lengths)) != 1:
                    raise ValueError("variable Flow attention layer lengths mismatch")
                row_lengths.append(layer_lengths[0])
            attention_lengths = tuple(row_lengths)
            max_length = max(attention_lengths)
            padded_rows: list[torch.Tensor] = []
            for cache in att_caches:
                if cache is None:
                    shape = list(reference.shape)
                    shape[1] = 2
                    shape[3] = 0
                    cache = reference.new_zeros(shape)
                if int(cache.shape[3]) < max_length:
                    padding_shape = list(cache.shape)
                    padding_shape[3] = max_length - int(cache.shape[3])
                    cache = torch.cat((cache, cache.new_zeros(padding_shape)), dim=3)
                padded_rows.append(cache.detach().clone())
            conditional = torch.cat([cache[:, 0:1] for cache in padded_rows], dim=1)
            unconditional = torch.cat([cache[:, 1:2] for cache in padded_rows], dim=1)
            att_physical = torch.cat((conditional, unconditional), dim=1)

        physical_lengths = attention_lengths + attention_lengths
        max_attention_length = max(physical_lengths) if physical_lengths else 0
        mask = torch.zeros(
            (2 * len(states), current_length, current_length + max_attention_length),
            dtype=torch.bool,
            device=states[0].x.device,
        )
        mask[:, :, :current_length] = True
        if max_attention_length:
            positions = torch.arange(max_attention_length, device=mask.device)
            lengths = torch.as_tensor(physical_lengths, device=mask.device)
            valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
            mask[:, :, current_length:] = valid.unsqueeze(1)
        return (
            step_indices,
            current_length,
            cnn_caches,
            att_caches,
            physical_lengths,
            cnn_physical,
            att_physical,
            first_span,
            mask,
        )

    def _advance_states_variable(
        self,
        states: tuple[CausalCFMStepState, ...],
        *,
        allow_mixed_step: bool = False,
        allowed_batch_sizes: tuple[int, ...] = (2,),
    ) -> tuple[CausalCFMStepState, ...]:
        (
            step_indices,
            current_length,
            cnn_caches,
            att_caches,
            physical_lengths,
            cnn_cache,
            att_cache,
            t_span,
            attention_valid_mask,
        ) = self._validate_variable_states(
            states,
            allow_mixed_step=allow_mixed_step,
            allowed_batch_sizes=allowed_batch_sizes,
        )
        first = states[0]
        steps = int(t_span.numel() - 1)
        x = torch.cat(tuple(state.x for state in states), dim=0)
        mu = torch.cat(tuple(state.mu for state in states), dim=0)
        spks = torch.cat(tuple(state.speaker for state in states), dim=0)
        cond = torch.cat(tuple(state.condition for state in states), dim=0)
        t = torch.cat(tuple(state.t for state in states), dim=0)

        # The estimator's physical CFG order is conditional [0:B] followed by
        # unconditional [0:B].  This is intentionally different from simply
        # concatenating each request's two-row cache.
        cfg_x = torch.cat((x, x), dim=0)
        cfg_mu = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        cfg_t = torch.cat((t, t), dim=0)
        cfg_spks = torch.cat((spks, torch.zeros_like(spks)), dim=0)
        cfg_cond = torch.cat((cond, torch.zeros_like(cond)), dim=0)

        estimator_step = getattr(self.estimator, "forward_chunk_variable", None)
        if estimator_step is None:
            raise RuntimeError("estimator does not expose forward_chunk_variable")
        result = estimator_step(
            x=cfg_x,
            mu=cfg_mu,
            t=cfg_t,
            spks=cfg_spks,
            cond=cfg_cond,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
            attention_cache_lengths=physical_lengths,
            attention_valid_mask=attention_valid_mask,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("variable Flow estimator must return dphi and two caches")
        dphi, new_cnn_cache, new_att_cache = result
        if not isinstance(dphi, torch.Tensor) or tuple(dphi.shape) != tuple(cfg_x.shape):
            raise ValueError("variable Flow estimator output shape mismatch")
        physical_batch = 2 * len(states)
        if (
            not isinstance(new_cnn_cache, torch.Tensor)
            or int(new_cnn_cache.shape[1]) != physical_batch
        ):
            raise ValueError("variable Flow estimator CNN cache output mismatch")
        if (
            not isinstance(new_att_cache, torch.Tensor)
            or new_att_cache.ndim != 5
            or int(new_att_cache.shape[1]) != physical_batch
            or int(new_att_cache.shape[3]) < current_length + max(physical_lengths)
        ):
            raise ValueError("variable Flow estimator attention cache output mismatch")

        conditional, unconditional = dphi.chunk(2, dim=0)
        combined = (
            (1.0 + self.inference_cfg_rate) * conditional
            - self.inference_cfg_rate * unconditional
        )
        outputs: list[CausalCFMStepState] = []
        for logical_row, state in enumerate(states):
            old_attention_length = int(att_caches[logical_row].shape[3]) if att_caches[logical_row] is not None else 0
            end = current_length + old_attention_length
            att_slice = self._variable_row_cache(
                new_att_cache, logical_row, len(states), end
            )
            if cnn_cache is not None:
                cnn_conditional = new_cnn_cache[:, logical_row:logical_row + 1]
                cnn_unconditional = new_cnn_cache[
                    :, len(states) + logical_row:len(states) + logical_row + 1
                ]
                cnn_slice = torch.cat(
                    (cnn_conditional, cnn_unconditional), dim=1
                ).detach().clone()
            else:
                # A real estimator always returns a CNN cache for a chunk.  A
                # missing input cache must not cause a partial logical state.
                cnn_slice = new_cnn_cache[
                    :, logical_row:logical_row + 1
                ].detach().clone()
                cnn_slice = torch.cat(
                    (
                        cnn_slice,
                        new_cnn_cache[
                            :, len(states) + logical_row:len(states) + logical_row + 1
                        ].detach().clone(),
                    ),
                    dim=1,
                )
            cnn_history = self._variable_history(state, "cnn", steps)
            att_history = self._variable_history(state, "att", steps)
            state_step = step_indices[logical_row]
            next_step = state_step + 1
            next_dt = (
                t_span[next_step + 1] - t_span[next_step]
                if next_step < steps
                else torch.zeros_like(state.dt)
            )
            cnn_history[state_step] = cnn_slice
            att_history[state_step] = att_slice
            updated = replace(
                state,
                x=(state.x + state.dt * combined[logical_row:logical_row + 1]).detach().clone(),
                t=(state.t + state.dt).detach().clone(),
                dt=next_dt.detach().clone(),
                step_index=next_step,
                completed_cnn_cache=cnn_slice,
                completed_att_cache=att_slice,
                input_cnn_cache=tuple(cnn_history),
                input_att_cache=tuple(att_history),
                _cnn_history=tuple(cnn_history),
                _att_history=tuple(att_history),
            )
            # Preserve optional chunk metadata without adding it to the
            # checkpoint/dataclass contract.
            updated.__dict__.update(
                {
                    key: value
                    for key, value in state.__dict__.items()
                    if key not in updated.__dict__
                }
            )
            outputs.append(updated)
        return tuple(outputs)

    def _advance_states(self, states: tuple[CausalCFMStepState, ...]) -> tuple[CausalCFMStepState, ...]:
        if not states:
            raise ValueError("at least one Flow state is required")
        first = states[0]
        steps = int(first.t_span.numel() - 1)
        if any(state.step_index != first.step_index for state in states):
            raise ValueError("all Flow states in a batch must have the same step index")
        if first.step_index >= steps:
            raise ValueError("Flow state has no remaining Euler steps")
        if any(state.t_span.shape != first.t_span.shape or not torch.equal(state.t_span, first.t_span) for state in states):
            raise ValueError("all Flow states in a batch must share t_span")

        logical_batch_sizes = tuple(int(state.x.shape[0]) for state in states)
        x = torch.cat(tuple(state.x for state in states), dim=0)
        mu = torch.cat(tuple(state.mu for state in states), dim=0)
        spks = torch.cat(tuple(state.speaker for state in states), dim=0)
        cond = torch.cat(tuple(state.condition for state in states), dim=0)
        t = torch.cat(tuple(state.t for state in states), dim=0)
        cnn_cache = self._join_step_caches(
            tuple(self._cache_step(state._cnn_history or state.input_cnn_cache, state.step_index) for state in states),
            name="cnn",
        )
        att_cache = self._join_step_caches(
            tuple(self._cache_step(state._att_history or state.input_att_cache, state.step_index) for state in states),
            name="attention",
        )

        cfg_x = torch.cat((x, x), dim=0)
        cfg_mu = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        cfg_t = torch.cat((t, t), dim=0)
        cfg_spks = torch.cat((spks, torch.zeros_like(spks)), dim=0)
        cfg_cond = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        dphi, new_cnn_cache, new_att_cache = self.estimator.forward_chunk(
            x=cfg_x,
            mu=cfg_mu,
            t=cfg_t,
            spks=cfg_spks,
            cond=cfg_cond,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        total_batch = x.shape[0]
        if dphi.shape[0] != total_batch * 2:
            raise ValueError("estimator CFG output batch does not match logical batch")
        conditional, unconditional = dphi.chunk(2, dim=0)
        combined = ((1.0 + self.inference_cfg_rate) * conditional
                    - self.inference_cfg_rate * unconditional)
        next_step = first.step_index + 1
        next_dt = (
            first.t_span[next_step + 1] - first.t_span[next_step]
            if next_step < steps else torch.zeros_like(first.dt)
        )

        outputs: list[CausalCFMStepState] = []
        logical_start = 0
        total_logical = sum(logical_batch_sizes)
        for state, logical_size in zip(states, logical_batch_sizes):
            logical_end = logical_start + logical_size
            # Estimator output rows are globally CFG ordered:
            # [conditional rows for every request,
            #  unconditional rows for every request]. Restore each
            # request-owned cache to [conditional, unconditional].
            conditional_start = logical_start
            conditional_end = logical_end
            unconditional_start = total_logical + logical_start
            unconditional_end = total_logical + logical_end
            cnn_slice = torch.cat(
                (
                    new_cnn_cache[:, conditional_start:conditional_end],
                    new_cnn_cache[:, unconditional_start:unconditional_end],
                ),
                dim=1,
            ).detach().clone()
            att_slice = torch.cat(
                (
                    new_att_cache[:, conditional_start:conditional_end],
                    new_att_cache[:, unconditional_start:unconditional_end],
                ),
                dim=1,
            ).detach().clone()
            cnn_history = list(state._cnn_history or (None,) * steps)
            att_history = list(state._att_history or (None,) * steps)
            cnn_history[state.step_index] = cnn_slice
            att_history[state.step_index] = att_slice
            cnn_history = tuple(cnn_history)
            att_history = tuple(att_history)
            outputs.append(replace(
                state,
                x=(state.x + state.dt * combined[logical_start:logical_end]).detach().clone(),
                t=(state.t + state.dt).detach().clone(),
                dt=next_dt.detach().clone(),
                step_index=next_step,
                completed_cnn_cache=cnn_slice,
                completed_att_cache=att_slice,
                input_cnn_cache=cnn_history,
                input_att_cache=att_history,
                _cnn_history=cnn_history,
                _att_history=att_history,
            ))
            logical_start = logical_end
        return tuple(outputs)

    @staticmethod
    def schedule_terminal_step(state: CausalCFMStepState) -> CausalCFMStepState:
        # Compress unfinished logical Flow work to one terminal step.
        from lychee_fd.runtime.apr.adaptive_flow_policy import schedule_terminal_step

        return schedule_terminal_step(state)

    def advance_chunk_step(self, state: CausalCFMStepState) -> CausalCFMStepState:
        """Advance one logical state and update it in place for pause/resume."""
        updated = self._advance_states((state,))[0]
        extra = {key: value for key, value in state.__dict__.items() if key not in updated.__dict__}
        state.__dict__.update(updated.__dict__)
        state.__dict__.update(extra)
        return state

    def advance_chunk_step_batch(
        self, states: Sequence[CausalCFMStepState]
    ) -> tuple[CausalCFMStepState, ...]:
        """Advance compatible logical states through one shared CFG call."""
        values = tuple(states)
        if any(not isinstance(state, CausalCFMStepState) for state in values):
            raise TypeError("Flow batch contains a non-CausalCFMStepState value")
        return self._advance_states(values)

    def advance_chunk_step_variable_batch(
        self, states: Sequence[CausalCFMStepState]
    ) -> tuple[CausalCFMStepState, ...]:
        """Advance the opt-in unequal-attention-cache B=2 path.

        A singleton explicitly delegates to the frozen B=1 implementation so
        that enabling the new public API cannot perturb the baseline path.
        """
        values = tuple(states)
        if len(values) == 1:
            if not isinstance(values[0], CausalCFMStepState):
                raise TypeError("Flow batch contains a non-CausalCFMStepState value")
            return (self.advance_chunk_step(values[0]),)
        return self._advance_states_variable(values)

    def advance_chunk_step_variable_mixed_batch(
        self, states: Sequence[CausalCFMStepState]
    ) -> tuple[CausalCFMStepState, ...]:
        """Advance exactly two variable-cache states at different Euler steps.

        This is an explicitly exploratory API.  The legacy B=1 method and the
        same-step variable batch method retain their original contracts.  A
        mixed call keeps each row's own ``t``, ``dt`` and cache-history slot;
        only the physical attention-cache time axis is temporarily padded.
        """
        values = tuple(states)
        if len(values) == 1:
            if not isinstance(values[0], CausalCFMStepState):
                raise TypeError("Flow batch contains a non-CausalCFMStepState value")
            return (self.advance_chunk_step(values[0]),)
        return self._advance_states_variable(values, allow_mixed_step=True)

    def advance_chunk_step_variable_mixed_batch_b4(
        self, states: Sequence[CausalCFMStepState]
    ) -> tuple[CausalCFMStepState, ...]:
        """Advance an explicit exploratory mixed-step batch with cap four.

        This API is intentionally separate from the validated B=2 method.  It
        accepts a physical logical batch of two, three, or four states so a
        runtime configured with a cap of four can safely fall back to a
        smaller naturally compatible group.  A two-row call delegates to the
        existing B=2 implementation; only three- and four-row calls exercise
        the generalized path.  No caller may use this to mix timestep
        schedules or alter the logical checkpoint contract.
        """
        values = tuple(states)
        if len(values) == 2:
            return self.advance_chunk_step_variable_mixed_batch(values)
        return self._advance_states_variable(
            values,
            allow_mixed_step=True,
            allowed_batch_sizes=(3, 4),
        )

    @staticmethod
    def _finish_cache(
        history: tuple[torch.Tensor | None, ...], capacity: int
    ) -> tuple[torch.Tensor, ...]:
        if not history or any(item is None for item in history):
            raise ValueError("Flow state cannot finish before every step has completed")
        return history

    def finish_chunk_steps(
        self, state: CausalCFMStepState
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return final Flow output and step-indexed caches."""
        if state.step_index != state.t_span.numel() - 1:
            raise ValueError("Flow state still has unfinished Euler steps")
        return (
            state.x,
            self._finish_cache(state._cnn_history, state._cache_capacity),
            self._finish_cache(state._att_history, state._cache_capacity),
        )

    def scatter_cuda_graph(
        self,
        enable_cuda_graph: bool,
        *,
        logical_batch_sizes: tuple[int, ...] = (1,),
    ):
        if enable_cuda_graph:
            self.estimator.cuda_graph_logical_batch_sizes = tuple(
                int(value) for value in logical_batch_sizes
            )
            self.estimator._init_cuda_graph_all(
                logical_batch_sizes=tuple(logical_batch_sizes)
            )

    def cuda_graph_stats(self) -> dict:
        """Return fixed-shape graph dispatch counters from the estimator."""
        getter = getattr(self.estimator, "cuda_graph_stats", None)
        if not callable(getter):
            return {
                "forward_chunk_calls": 0,
                "graph_replay_calls": 0,
                "fallback_calls": 0,
                "fallback_reasons": {},
            }
        return dict(getter())

    def reset_cuda_graph_stats(self) -> None:
        reset = getattr(self.estimator, "reset_cuda_graph_stats", None)
        if callable(reset):
            reset()

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        assert self.inference_cfg_rate > 0, 'inference_cfg_rate better > 0'

        # constant during denoising
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        
        for step in range(1, len(t_span)):

            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)

            dphi_dt = self.estimator.forward(
                x_in,
                mask_in,
                mu_in,
                t_in,
                spks_in,
                cond_in,
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x

    @torch.inference_mode()
    def forward(self, mu, mask, spks, cond, n_timesteps=10, temperature=1.0):
        z = self.rand_noise[:, :, :mu.size(2)] * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # cosine scheduling
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond)

    def solve_euler_chunk(self, 
                          x:torch.Tensor, 
                          t_span:torch.Tensor, 
                          mu:torch.Tensor, 
                          spks:torch.Tensor, 
                          cond:torch.Tensor, 
                          cnn_cache:torch.Tensor=None,
                          att_cache:torch.Tensor=None,
                          ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        assert self.inference_cfg_rate > 0, 'cfg rate should be > 0'
        
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)  # (b,)

        # setup initial cache
        if cnn_cache is None:
            cnn_cache = [None for _ in range(len(t_span)-1)]
        if att_cache is None:
            att_cache = [None for _ in range(len(t_span)-1)]
        # next chunk's cache at each timestep

        if att_cache[0] is not None:
            last_att_len = att_cache.shape[4]
        else:
            last_att_len = 0

        # constant during denoising
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        for step in range(1, len(t_span)):
            # torch.cuda.memory._record_memory_history(max_entries=100000)
            # torch.cuda.memory._record_memory_history(max_entries=100000)
            this_att_cache = att_cache[step-1]
            this_cnn_cache = cnn_cache[step-1]

            dphi_dt, this_new_cnn_cache, this_new_att_cache = self.estimator.forward_chunk(
                x = x.repeat(2, 1, 1),
                mu = mu_in,
                t = t.repeat(2),
                spks = spks_in,
                cond = cond_in,
                cnn_cache = this_cnn_cache,
                att_cache = this_att_cache,
            )
            dphi_dt, cfg_dphi_dt = dphi_dt.chunk(2, dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

            self.cnn_cache_buffer[step-1] = this_new_cnn_cache
            self.att_cache_buffer[step-1][:, :, :, :x.shape[2]+last_att_len, :] = this_new_att_cache
        
        cnn_cache = self.cnn_cache_buffer
        att_cache = self.att_cache_buffer[:, :, :, :, :x.shape[2]+last_att_len, :]
        return x, cnn_cache, att_cache
    
    @torch.inference_mode()
    def forward_chunk(self, 
                      mu:torch.Tensor, 
                      spks:torch.Tensor, 
                      cond:torch.Tensor,
                      n_timesteps:int=10, 
                      temperature:float=1.0, 
                      cnn_cache:torch.Tensor=None,
                      att_cache:torch.Tensor=None,
                      ):
        """
        Args:
            mu(torch.Tensor): shape (b, c, t)
            spks(torch.Tensor): shape (b, 192)
            cond(torch.Tensor): shape (b, c, t)
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        # get offset from att_cache
        offset = att_cache.shape[4] if att_cache is not None else 0
        z = self.rand_noise[:, :, offset:offset+mu.size(2)] * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # cosine scheduling
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        x, new_cnn_cache, new_att_cache = self.solve_euler_chunk(
            x=z,
            t_span=t_span,
            mu=mu,
            spks=spks,
            cond=cond,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
        )
        return x, new_cnn_cache, new_att_cache
