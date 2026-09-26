import math
import os
import torch
import numpy as np
from typing import Optional
from einops import pack, rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_CUDA_GRAPH_CHUNK_SIZES = (30, 48, 96)
_SUPPORTED_CUDA_GRAPH_LOGICAL_BATCHES = (1, 2)


def _new_cuda_graph_stats() -> dict:
    return {
        "forward_chunk_calls": 0,
        "graph_replay_calls": 0,
        "fallback_calls": 0,
        "fallback_reasons": {},
        "forward_chunk_calls_by_logical_batch": {"1": 0, "2": 0},
        "graph_replay_calls_by_logical_batch": {"1": 0, "2": 0},
        "fallback_calls_by_logical_batch": {"1": 0, "2": 0},
        "fallback_reasons_by_logical_batch": {"1": {}, "2": {}},
        "eligible_calls": 0,
    }


def _resolve_cuda_graph_logical_batches() -> tuple[int, ...]:
    """Resolve the bounded logical batches captured by the opt-in graph path.

    The default remains the historical B=1 graph.  B=2 is deliberately an
    explicit opt-in because it allocates a separate physical CFG=4 graph and
    its static buffers; unsupported sizes fail closed instead of silently
    changing the execution contract.
    """
    raw = os.environ.get("LYCHEEFD_FLOW_CUDA_GRAPH_LOGICAL_BATCHES", "").strip()
    if not raw:
        return (1,)
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError(
                "LYCHEEFD_FLOW_CUDA_GRAPH_LOGICAL_BATCHES must contain logical batch sizes"
            )
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                "LYCHEEFD_FLOW_CUDA_GRAPH_LOGICAL_BATCHES must contain logical batch sizes"
            ) from exc
        if value not in _SUPPORTED_CUDA_GRAPH_LOGICAL_BATCHES:
            raise ValueError(
                "unsupported logical batch size; CUDA Graph supports logical batch 1 or 2"
            )
        values.append(value)
    return tuple(sorted(set(values)))


def _cfg_row_order(logical_batch: int) -> tuple[tuple[str, int], ...]:
    """Return the public conditional/unconditional physical row ordering."""
    logical_batch = int(logical_batch)
    if logical_batch not in _SUPPORTED_CUDA_GRAPH_LOGICAL_BATCHES:
        raise ValueError("logical batch must be 1 or 2")
    return tuple(
        [("conditional", row) for row in range(logical_batch)]
        + [("unconditional", row) for row in range(logical_batch)]
    )


def _cuda_graph_static_shapes(
    logical_batch: int, chunk_size: int, max_cache: int
) -> dict[str, tuple[int, ...] | int]:
    """Describe graph-local physical buffers without exposing logical state."""
    logical_batch = int(logical_batch)
    chunk_size = int(chunk_size)
    max_cache = int(max_cache)
    if logical_batch not in _SUPPORTED_CUDA_GRAPH_LOGICAL_BATCHES:
        raise ValueError("logical batch must be 1 or 2")
    if chunk_size <= 0 or max_cache <= 0:
        raise ValueError("chunk and cache sizes must be positive")
    physical = 2 * logical_batch
    return {
        "logical_batch": logical_batch,
        "physical_cfg_batch": physical,
        "x": (physical, 320, chunk_size),
        "t": (physical, 1, 512),
        "mask": (physical, chunk_size, max_cache + chunk_size),
        "cnn_cache": (16, physical, 1024, 2),
        "att_cache": (16, physical, 8, max_cache, 128),
    }


def _graph_fallback_reason(
    *,
    logical_batch_size: int,
    captured_logical_batch_sizes: tuple[int, ...],
    chunk_sizes: tuple[int, ...],
    current_chunk_lengths: tuple[int, ...],
    cnn_cache_shapes: tuple[tuple[int, ...], ...],
    last_chunks: tuple[bool, ...],
    n_timesteps: tuple[int, ...],
    generation_valid: bool,
    attention_cache_lengths: tuple[int, ...],
    max_graph_cache: int,
    step_indices: tuple[int, ...],
) -> str | None:
    """Classify a graph miss using only bounded public metadata.

    ``step_indices`` are intentionally not compared: the B=2 graph accepts
    per-row Euler positions, provided each row supplies its own t/dt values.
    """
    if not generation_valid:
        return "invalid_generation"
    if logical_batch_size not in tuple(captured_logical_batch_sizes):
        return "logical_batch_not_captured"
    if len(current_chunk_lengths) != logical_batch_size:
        return "logical_batch_shape"
    if len(set(int(value) for value in current_chunk_lengths)) != 1:
        return "current_chunk_shape"
    # ``chunk_sizes`` is normally the captured shape set (e.g. (30, 48, 96));
    # callers may also provide one requested shape per row.  Treat both forms
    # without confusing the captured set with a row-mismatch signal.
    requested_chunk = int(current_chunk_lengths[0])
    if requested_chunk not in {int(value) for value in chunk_sizes}:
        return "chunk_shape"
    if len(chunk_sizes) == logical_batch_size and len(set(int(value) for value in chunk_sizes)) != 1:
        return "chunk_shape"
    if len(cnn_cache_shapes) != logical_batch_size or len(set(cnn_cache_shapes)) != 1:
        return "cnn_cache_shape"
    if len(last_chunks) != logical_batch_size or len(set(bool(value) for value in last_chunks)) != 1:
        return "last_chunk"
    if len(n_timesteps) != logical_batch_size or len(set(int(value) for value in n_timesteps)) != 1:
        return "n_timesteps"
    if len(attention_cache_lengths) != logical_batch_size:
        return "attention_cache_shape"
    if any(int(value) < 0 or int(value) > int(max_graph_cache) for value in attention_cache_lengths):
        return "graph_cache_limit"
    return None


def _resolve_cuda_graph_chunk_sizes() -> tuple[int, ...]:
    """Resolve the bounded fixed-shape CUDA Graph capture set.

    The graph path is opt-in and must never silently capture an unbounded set
    of shapes: every key owns static buffers and a CUDA graph.  Keeping the
    default set unchanged preserves the upstream behavior; deployments that
    have measured additional shapes can explicitly add them through the
    benchmark-only environment variable.
    """
    raw = os.environ.get("LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS", "").strip()
    if not raw:
        return _DEFAULT_CUDA_GRAPH_CHUNK_SIZES
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError(
                "LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS must contain positive integers"
            )
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                "LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS must contain positive integers"
            ) from exc
        if value <= 0 or value > 256:
            raise ValueError(
                "LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS values must be in [1, 256]"
            )
        values.append(value)
    resolved = tuple(sorted(set(values)))
    if not resolved:
        raise ValueError(
            "LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS must contain at least one shape"
        )
    if len(resolved) > 8:
        raise ValueError(
            "LYCHEEFD_FLOW_CUDA_GRAPH_CHUNKS supports at most 8 shapes"
        )
    return resolved


def _resolve_cuda_graph_max_cache(chunk_size: int, attention_capacity: int) -> int:
    """Return the bounded cache length used by a fixed-shape graph.

    The legacy defaults are intentionally preserved.  A measured deployment
    may opt in to a larger bound, but it can never exceed the logical Flow
    cache capacity allocated for the process; this prevents graph capture from
    silently creating a cache that the runtime cannot commit.
    """
    try:
        chunk_size = int(chunk_size)
        attention_capacity = int(attention_capacity)
    except (TypeError, ValueError) as exc:
        raise ValueError("CUDA Graph cache dimensions must be integers") from exc
    if chunk_size <= 0 or attention_capacity <= 0:
        raise ValueError("CUDA Graph cache dimensions must be positive")
    default = 500 if chunk_size in (30, 48) else 1000
    raw = os.environ.get("LYCHEEFD_FLOW_CUDA_GRAPH_MAX_CACHE", "").strip()
    if not raw:
        return min(default, attention_capacity)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "LYCHEEFD_FLOW_CUDA_GRAPH_MAX_CACHE must be a positive integer"
        ) from exc
    if value <= 0 or value > attention_capacity:
        raise ValueError(
            "LYCHEEFD_FLOW_CUDA_GRAPH_MAX_CACHE must be in "
            f"[1, {attention_capacity}]"
        )
    return value



"""
DiT-v5
- Add convolution in DiTBlock to increase high-freq component
"""


class MLP(torch.nn.Module):
    def __init__(
            self,
            in_features:int,
            hidden_features:Optional[int]=None,
            out_features:Optional[int]=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            head_dim: int = 64,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj = nn.Linear(self.inner_dim, dim)

    def to_heads(self, ts:torch.Tensor):
        b, t, c = ts.shape
        # (b, t, nh, c)
        ts = ts.reshape(b, t, self.num_heads, c // self.num_heads)
        ts = ts.transpose(1, 2)
        return ts
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Args:
            x(torch.Tensor): shape (b, t, c)
            attn_mask(torch.Tensor): shape (b, t, t)
        """
        b, t, c = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.to_heads(q)    # (b, nh, t, c)
        k = self.to_heads(k)
        v = self.to_heads(v)
    
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )   # (b, nh, t, c)
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def forward_chunk(self, x: torch.Tensor, att_cache: torch.Tensor=None, attn_mask: torch.Tensor=None):
        """
        Args:
            x: shape (b, dt, c)
            att_cache: shape (b, nh, t, c*2)
        """
        b, t, c = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.to_heads(q)    # (b, nh, t, c)
        k = self.to_heads(k)
        v = self.to_heads(v)
    
        q = self.q_norm(q)
        k = self.k_norm(k)

        # unpack {k,v}_cache
        if att_cache is not None:
            if attn_mask is not None:
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                k = torch.cat([k, k_cache], dim=2)
                v = torch.cat([v, v_cache], dim=2)    

            else:    
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                k = torch.cat([k, k_cache], dim=2)
                v = torch.cat([v, v_cache], dim=2)      
        
        # new {k,v}_cache
        new_att_cache = torch.cat([k, v], dim=3)
        # attn_mask = torch.ones((b, 1, t, t1), dtype=torch.bool, device=x.device)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)   # (b, nh, t, c)
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, new_att_cache


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # from SinusoidalPosEmb
        self.scale = 1000

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.scale, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# Convolution related
class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super(CausalConv1d, self).forward(x)
        return x
    
    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor=None):
        if cnn_cache is None:
            cnn_cache = x.new_zeros((x.shape[0], self.in_channels, self.causal_padding[0]))
        x = torch.cat([cnn_cache, x], dim=2)
        new_cnn_cache = x[..., -self.causal_padding[0]:]
        x = super(CausalConv1d, self).forward(x)
        return x, new_cnn_cache


class CausalConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.block = torch.nn.Sequential(
            # norm
            # conv1
            Transpose(1, 2),
            CausalConv1d(in_channels, out_channels, kernel_size),
            Transpose(1, 2),
            # norm & act
            nn.LayerNorm(out_channels),
            nn.Mish(),
            # conv2
            Transpose(1, 2),
            CausalConv1d(out_channels, out_channels, kernel_size),
            Transpose(1, 2),
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: shape (b, t, c)
            mask: shape (b, t, 1)
        """
        if mask is not None: x = x * mask
        x = self.block(x)
        if mask is not None: x = x * mask
        return x
    
    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor=None):
        """
        Args:
            x: shape (b, dt, c)
            cnn_cache: shape (b, c1+c2, 2)
        """
        if cnn_cache is not None:
            cnn_cache1, cnn_cache2 = cnn_cache.split((self.in_channels, self.out_channels), dim=1)
        else:
            cnn_cache1, cnn_cache2 = None, None
        x = self.block[0](x)
        x, new_cnn_cache1 = self.block[1].forward_chunk(x, cnn_cache1)
        x = self.block[2:6](x)
        x, new_cnn_cache2 = self.block[6].forward_chunk(x, cnn_cache2)
        x = self.block[7](x)
        new_cnn_cache = torch.cat((new_cnn_cache1, new_cnn_cache2), dim=1)
        return x, new_cnn_cache


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, head_dim, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, qk_norm=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.conv = CausalConvBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x:torch.Tensor, c:torch.Tensor, attn_mask:torch.Tensor):
        """Args
            x: shape (b, t, c)
            c: shape (b, 1, c)
            attn_mask: shape (b, t, t), bool type attention mask
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv \
              = self.adaLN_modulation(c).chunk(9, dim=-1)
        # attention
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        # conv
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv))
        # mlp
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    def forward_chunk(self, x: torch.Tensor, c: torch.Tensor, cnn_cache: torch.Tensor=None, att_cache: torch.Tensor=None, mask: torch.Tensor=None):
        """
        Args:
            x: shape (b, dt, c)
            c: shape (b, 1, c)
            cnn_cache: shape (b, c1+c2, 2)
            att_cache: shape (b, nh, t, c * 2)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv \
              = self.adaLN_modulation(c).chunk(9, dim=-1)
        # attention
        x_att, new_att_cache = self.attn.forward_chunk(modulate(self.norm1(x), shift_msa, scale_msa), att_cache, mask)
        x = x + gate_msa * x_att
        # conv
        x_conv, new_cnn_cache = self.conv.forward_chunk(modulate(self.norm3(x), shift_conv, scale_conv), cnn_cache)
        x = x + gate_conv * x_conv
        # mlp
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, new_cnn_cache, new_att_cache


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        head_dim: int = 64,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.in_proj = nn.Linear(in_channels, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, head_dim, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        self.initialize_weights()

        self.enable_cuda_graph = False
        self.use_cuda_graph = False

        self.graph_chunk = {}
        self.inference_buffers_chunk = {}
        self.max_size_chunk = {}
        # New graph records are keyed by (logical_batch, chunk_size).  The
        # legacy B=1 dictionaries above remain populated as aliases so older
        # diagnostic callers see the same shape and lookup behavior.
        self.graph_chunk_by_logical_batch = {}
        self.inference_buffers_chunk_by_logical_batch = {}
        self.max_size_chunk_by_logical_batch = {}
        self.cuda_graph_logical_batch_sizes = (1,)
        self._cuda_graph_stats = _new_cuda_graph_stats()

        from lychee_fd.runtime.apr.flow_cache_capacity import (
            resolve_flow_attention_cache_capacity,
        )
        attention_cache_capacity = resolve_flow_attention_cache_capacity()
        self.register_buffer(
            'att_cache_buffer',
            torch.zeros((16, 2, 8, attention_cache_capacity, 128)),
            persistent=False,
        )
        self.register_buffer('cnn_cache_buffer', torch.zeros((16, 2, 1024, 2)), persistent=False)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def reset_cuda_graph_stats(self):
        """Reset optional dispatch counters without touching model state."""
        old = getattr(self, "_cuda_graph_stats", None)
        # A few downstream compatibility tests construct a DiT instance with
        # ``__new__`` and install the original four-field dictionary.  Keep
        # that old snapshot contract for such callers while real instances
        # use the extended per-logical-batch counters.
        legacy = isinstance(old, dict) and "eligible_calls" not in old
        if legacy:
            self._cuda_graph_stats = {
                "forward_chunk_calls": 0,
                "graph_replay_calls": 0,
                "fallback_calls": 0,
                "fallback_reasons": {},
            }
        else:
            self._cuda_graph_stats = _new_cuda_graph_stats()

    def cuda_graph_stats(self) -> dict:
        """Return a detached, JSON-serializable graph dispatch snapshot."""
        stats = getattr(self, "_cuda_graph_stats", None)
        if not isinstance(stats, dict):
            stats = _new_cuda_graph_stats()
            self._cuda_graph_stats = stats
        snapshot = {
            "forward_chunk_calls": int(stats.get("forward_chunk_calls", 0)),
            "graph_replay_calls": int(stats.get("graph_replay_calls", 0)),
            "fallback_calls": int(stats.get("fallback_calls", 0)),
            "fallback_reasons": dict(stats.get("fallback_reasons", {})),
        }
        if "eligible_calls" in stats:
            snapshot.update({
                "forward_chunk_calls_by_logical_batch": dict(
                    stats.get(
                        "forward_chunk_calls_by_logical_batch",
                        {"1": 0, "2": 0},
                    )
                ),
                "graph_replay_calls_by_logical_batch": dict(
                    stats.get(
                        "graph_replay_calls_by_logical_batch",
                        {"1": 0, "2": 0},
                    )
                ),
                "fallback_calls_by_logical_batch": dict(
                    stats.get(
                        "fallback_calls_by_logical_batch",
                        {"1": 0, "2": 0},
                    )
                ),
                "fallback_reasons_by_logical_batch": {
                    str(key): dict(value)
                    for key, value in stats.get(
                        "fallback_reasons_by_logical_batch",
                        {"1": {}, "2": {}},
                    ).items()
                },
                "eligible_calls": int(stats.get("eligible_calls", 0)),
            })
        return snapshot

    def _record_cuda_graph_call(self, logical_batch_size: int) -> None:
        stats = self._cuda_graph_stats
        key = str(int(logical_batch_size))
        stats["forward_chunk_calls"] = int(stats.get("forward_chunk_calls", 0)) + 1
        if "forward_chunk_calls_by_logical_batch" in stats:
            by_batch = stats["forward_chunk_calls_by_logical_batch"]
            by_batch[key] = int(by_batch.get(key, 0)) + 1

    def _record_cuda_graph_fallback(
        self, reason: str, logical_batch_size: int = 1
    ) -> None:
        stats = self._cuda_graph_stats
        stats["fallback_calls"] = int(stats.get("fallback_calls", 0)) + 1
        reasons = stats.setdefault("fallback_reasons", {})
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        key = str(int(logical_batch_size))
        if "fallback_calls_by_logical_batch" in stats:
            by_batch = stats["fallback_calls_by_logical_batch"]
            by_batch[key] = int(by_batch.get(key, 0)) + 1
            reason_by_batch = stats["fallback_reasons_by_logical_batch"]
            bucket = reason_by_batch.setdefault(key, {})
            bucket[reason] = int(bucket.get(reason, 0)) + 1

    @staticmethod
    def _graph_key(logical_batch_size: int, chunk_size: int) -> tuple[int, int]:
        return int(logical_batch_size), int(chunk_size)

    def _capture_cuda_graph_shape(
        self, logical_batch_size: int, chunk_size: int, max_size: int
    ) -> tuple[torch.cuda.CUDAGraph, dict[str, list[torch.Tensor]]]:
        """Capture one graph using only graph-local physical buffers."""
        shapes = _cuda_graph_static_shapes(logical_batch_size, chunk_size, max_size)
        physical_batch = int(shapes["physical_cfg_batch"])
        dtype = self.cnn_cache_buffer.dtype
        device = self.cnn_cache_buffer.device
        static_x = torch.zeros(shapes["x"], dtype=dtype, device=device)
        static_t = torch.zeros(shapes["t"], dtype=dtype, device=device)
        static_mask = torch.ones(
            shapes["mask"], dtype=torch.bool, device=device
        )
        static_att_cache = torch.zeros(
            shapes["att_cache"], dtype=dtype, device=device
        )
        static_cnn_cache = torch.zeros(
            shapes["cnn_cache"], dtype=dtype, device=device
        )
        static_inputs = [
            static_x,
            static_t,
            static_mask,
            static_cnn_cache,
            static_att_cache,
        ]
        static_new_cnn_cache = torch.zeros(
            (16, physical_batch, 1024, 2), dtype=dtype, device=device
        )
        static_new_att_cache = torch.zeros(
            (16, physical_batch, 8, max_size + chunk_size, 128),
            dtype=dtype,
            device=device,
        )
        # A warmup establishes all allocations before capture.  The returned
        # tensors are deliberately retained only in this graph-local record.
        self.blocks_forward_chunk(
            static_x,
            static_t,
            static_mask,
            static_cnn_cache,
            static_att_cache,
            static_new_cnn_cache,
            static_new_att_cache,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.blocks_forward_chunk(
                static_x,
                static_t,
                static_mask,
                static_cnn_cache,
                static_att_cache,
                static_new_cnn_cache,
                static_new_att_cache,
            )
        return graph, {
            "static_inputs": static_inputs,
            "static_outputs": [
                static_out,
                static_new_cnn_cache,
                static_new_att_cache,
            ],
        }

    def _init_cuda_graph_chunk(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph requires a CUDA device")
        self.graph_chunk.clear()
        self.inference_buffers_chunk.clear()
        self.max_size_chunk.clear()
        self.graph_chunk_by_logical_batch.clear()
        self.inference_buffers_chunk_by_logical_batch.clear()
        self.max_size_chunk_by_logical_batch.clear()
        # Capture a separately-owned physical graph for every explicitly
        # enabled logical batch.  B=1 remains the default.
        with torch.no_grad():
            for logical_batch_size in self.cuda_graph_logical_batch_sizes:
                for chunk_size in _resolve_cuda_graph_chunk_sizes():
                    max_size = _resolve_cuda_graph_max_cache(
                        chunk_size,
                        int(self.att_cache_buffer.shape[3]),
                    )
                    key = self._graph_key(logical_batch_size, chunk_size)
                    graph_chunk, buffers = self._capture_cuda_graph_shape(
                        logical_batch_size, chunk_size, max_size
                    )
                    self.max_size_chunk_by_logical_batch[key] = max_size
                    self.graph_chunk_by_logical_batch[key] = graph_chunk
                    self.inference_buffers_chunk_by_logical_batch[key] = buffers
                    if logical_batch_size == 1:
                        # Preserve the historical B=1 lookup tables.
                        self.max_size_chunk[chunk_size] = max_size
                        self.graph_chunk[chunk_size] = graph_chunk
                        self.inference_buffers_chunk[chunk_size] = buffers

    def _init_cuda_graph_all(
        self, logical_batch_sizes: tuple[int, ...] | None = None
    ):
        self.reset_cuda_graph_stats()
        if logical_batch_sizes is None:
            logical_batch_sizes = _resolve_cuda_graph_logical_batches()
        else:
            values = tuple(int(value) for value in logical_batch_sizes)
            if not values or any(
                value not in _SUPPORTED_CUDA_GRAPH_LOGICAL_BATCHES
                for value in values
            ):
                raise ValueError("logical batch sizes must contain only 1 or 2")
            logical_batch_sizes = tuple(sorted(set(values)))
        self.cuda_graph_logical_batch_sizes = logical_batch_sizes
        self._init_cuda_graph_chunk()
        self.use_cuda_graph = True
        print(f"CUDA Graph initialized successfully for chunk decoder")

    def _replay_cuda_graph(
        self,
        *,
        logical_batch_size: int,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        attention_cache_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay a captured graph using graph-local padded physical buffers.

        Padding exists only in the execution buffers.  The caller crops the
        returned attention cache to each request's real length before it is
        placed back into logical Flow state.
        """
        logical_batch_size = int(logical_batch_size)
        chunk_size = int(x.shape[2])
        key = self._graph_key(logical_batch_size, chunk_size)
        try:
            graph = self.graph_chunk_by_logical_batch[key]
            buffers = self.inference_buffers_chunk_by_logical_batch[key]
            max_cache = int(self.max_size_chunk_by_logical_batch[key])
        except (AttributeError, KeyError) as exc:
            raise KeyError("CUDA Graph shape is not captured") from exc
        physical_batch = 2 * logical_batch_size
        expected_x = (physical_batch, 320, chunk_size)
        expected_t = (physical_batch, 1, 512)
        if tuple(x.shape) != expected_x or tuple(t.shape) != expected_t:
            raise ValueError("CUDA Graph input shape does not match captured shape")
        if tuple(mask.shape[:2]) != (physical_batch, chunk_size):
            raise ValueError("CUDA Graph mask batch/chunk shape mismatch")
        if mask.ndim != 3 or int(mask.shape[-1]) > max_cache + chunk_size:
            raise ValueError("CUDA Graph mask width exceeds captured shape")
        if int(attention_cache_length) < 0 or int(attention_cache_length) > max_cache:
            raise ValueError("CUDA Graph attention cache length exceeds capture")
        if cnn_cache is not None and tuple(cnn_cache.shape) != (16, physical_batch, 1024, 2):
            raise ValueError("CUDA Graph CNN cache shape mismatch")
        if att_cache is not None:
            if tuple(att_cache.shape[:3]) != (16, physical_batch, 8):
                raise ValueError("CUDA Graph attention cache shape mismatch")
            if int(att_cache.shape[3]) > max_cache or int(att_cache.shape[4]) != 128:
                raise ValueError("CUDA Graph attention cache shape mismatch")

        static_inputs = buffers["static_inputs"]
        static_inputs[0].copy_(x)
        static_inputs[1].copy_(t)
        static_inputs[2].zero_()
        static_inputs[2][..., : mask.shape[-1]].copy_(mask)
        if cnn_cache is None:
            static_inputs[3].zero_()
        else:
            static_inputs[3].copy_(cnn_cache)
        static_inputs[4].zero_()
        if att_cache is not None and att_cache.shape[3] > 0:
            static_inputs[4][..., : att_cache.shape[3], :].copy_(att_cache)

        graph.replay()
        static_outputs = buffers["static_outputs"]
        output = static_outputs[0][:, :, :chunk_size]
        new_cnn_cache = static_outputs[1]
        # The physical cache has current chunk first, followed by the padded
        # historical cache.  Crop only the common maximum input length; the
        # Flow state splitter applies each row's exact length afterwards.
        new_att_cache = static_outputs[2][
            :, :, :, : chunk_size + int(attention_cache_length), :
        ]
        return output, new_cnn_cache, new_att_cache

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Args:
            x: shape (b, c, t)
            mask: shape (b, 1, t)
            t: shape (b,)
            spks: shape (b, c)
            cond: shape (b, c, t)
        """
        # (sfy) chunk training strategy should not be open-sourced

        # time
        t = self.t_embedder(t).unsqueeze(1)  # (b, 1, c)
        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        return self.blocks_forward(x, t, mask)

    def blocks_forward(self, x, t, mask):
        x = x.transpose(1, 2)
        attn_mask = mask.bool()
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, t, attn_mask)
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        return x

    def forward_chunk(self, 
                      x: torch.Tensor, 
                      mu: torch.Tensor, 
                      t: torch.Tensor, 
                      spks: torch.Tensor, 
                      cond: torch.Tensor, 
                      cnn_cache: torch.Tensor = None,
                      att_cache: torch.Tensor = None,
                      ):
        """
        Args:
            x: shape (b, dt, c)
            mu: shape (b, dt, c)
            t: shape (b,)
            spks: shape (b, c)
            cond: shape (b, dt, c)
            cnn_cache: shape (depth, b, c1+c2, 2)
            att_cache: shape (depth, b, nh, t, c * 2)
        """

        # time
        t = self.t_embedder(t).unsqueeze(1)  # (b, 1, c)
        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        # Keep the tensor references for the graph path before normalizing
        # missing caches to the legacy per-layer list representation.
        input_cnn_cache = cnn_cache if isinstance(cnn_cache, torch.Tensor) else None
        input_att_cache = att_cache if isinstance(att_cache, torch.Tensor) else None
        if cnn_cache is None:
            cnn_cache = [None] * len(self.blocks)
        if att_cache is None:
            att_cache = [None] * len(self.blocks)
        if input_att_cache is not None:
            last_att_len = int(input_att_cache.shape[3])
        else:
            last_att_len = 0
        chunk_size = x.shape[2]
        mask = torch.ones(
            x.shape[0], chunk_size, last_att_len + chunk_size,
            dtype=torch.bool, device=x.device
        )
        logical_batch_size = x.shape[0] // 2 if x.shape[0] in (2, 4) else 0
        use_fixed_worker_buffers = logical_batch_size == 1
        graph_tracking = bool(self.use_cuda_graph)
        if graph_tracking and logical_batch_size:
            self._record_cuda_graph_call(logical_batch_size)
        required_att_capacity = int(last_att_len + chunk_size)
        available_att_capacity = int(self.att_cache_buffer.shape[3])
        if logical_batch_size and required_att_capacity > available_att_capacity:
            if graph_tracking:
                self._record_cuda_graph_fallback(
                    "runtime_cache_capacity", logical_batch_size
                )
            raise RuntimeError(
                "Flow attention cache capacity exceeded: "
                f"required={required_att_capacity}, available={available_att_capacity}; "
                "set LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY before loading Token2Wav"
            )


        graph_key = (
            self._graph_key(logical_batch_size, chunk_size)
            if logical_batch_size else None
        )
        graph_hit = (
            self.use_cuda_graph
            and logical_batch_size in getattr(
                self, "cuda_graph_logical_batch_sizes", ()
            )
            and graph_key in self.graph_chunk_by_logical_batch
            and last_att_len <= self.max_size_chunk_by_logical_batch[graph_key]
            and (
                input_cnn_cache is None
                or tuple(input_cnn_cache.shape)
                == (16, 2 * logical_batch_size, 1024, 2)
            )
        )
        if graph_hit:
            self._cuda_graph_stats["graph_replay_calls"] += 1
            by_batch = self._cuda_graph_stats.get(
                "graph_replay_calls_by_logical_batch"
            )
            if by_batch is not None:
                key = str(logical_batch_size)
                by_batch[key] = int(by_batch.get(key, 0)) + 1
            self._cuda_graph_stats["eligible_calls"] = int(
                self._cuda_graph_stats.get("eligible_calls", 0)
            ) + 1
            try:
                x, new_cnn_cache, new_att_cache = self._replay_cuda_graph(
                    logical_batch_size=logical_batch_size,
                    x=x,
                    t=t,
                    mask=mask,
                    cnn_cache=input_cnn_cache,
                    att_cache=input_att_cache,
                    attention_cache_length=last_att_len,
                )
            except (KeyError, ValueError):
                # An unexpected shape mismatch must fail closed to the
                # existing dynamic path and be visible in telemetry.
                self._record_cuda_graph_fallback(
                    "graph_input_shape", logical_batch_size
                )
                mask = None
                if use_fixed_worker_buffers:
                    self.blocks_forward_chunk(
                        x, t, mask, cnn_cache, att_cache,
                        self.cnn_cache_buffer, self.att_cache_buffer
                    )
                    new_cnn_cache = self.cnn_cache_buffer
                    new_att_cache = self.att_cache_buffer[
                        :, :, :, : last_att_len + chunk_size, :
                    ]
                else:
                    x, new_cnn_cache, new_att_cache = self.blocks_forward_chunk(
                        x, t, mask, cnn_cache, att_cache, None, None
                    )
        elif use_fixed_worker_buffers:
            if graph_tracking:
                if input_att_cache is None:
                    self._record_cuda_graph_fallback("missing_attention_cache", 1)
                elif graph_key not in self.graph_chunk_by_logical_batch:
                    self._record_cuda_graph_fallback("chunk_not_captured", 1)
                elif last_att_len > self.max_size_chunk_by_logical_batch[graph_key]:
                    self._record_cuda_graph_fallback("graph_cache_limit", 1)
                else:
                    self._record_cuda_graph_fallback("predicate_unknown", 1)
            mask = None
            x = self.blocks_forward_chunk(x, t, mask, cnn_cache, att_cache, self.cnn_cache_buffer, self.att_cache_buffer)
            new_cnn_cache = self.cnn_cache_buffer
            new_att_cache = self.att_cache_buffer[:, :, :, :last_att_len+chunk_size, :]
        else:
            if graph_tracking:
                self._record_cuda_graph_fallback(
                    "logical_batch_shape", logical_batch_size or 1
                )
            # Dynamic CFG batches cannot use the legacy batch-2 worker buffers.
            # Collect per-layer caches from this call instead; the caller owns
            # the returned tensors and can split them by logical request.
            mask = None
            x, new_cnn_cache, new_att_cache = self.blocks_forward_chunk(
                x, t, mask, cnn_cache, att_cache, None, None
            )

        return x, new_cnn_cache, new_att_cache

    def forward_chunk_variable(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
        attention_cache_lengths=None,
        attention_valid_mask: torch.Tensor = None,
    ):
        """Run an opt-in physical batch with per-row attention cache lengths.

        The legacy ``forward_chunk`` path deliberately remains untouched.  A
        variable batch is executed through the dynamic cache path only: cache
        time positions are padded temporarily, while ``attention_valid_mask``
        prevents a shorter row from attending to that padding.  The caller
        owns cropping the returned cache back to each logical row's length.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("variable Flow x must have shape [batch, channels, time]")
        if not isinstance(mu, torch.Tensor) or tuple(mu.shape) != tuple(x.shape):
            raise ValueError("variable Flow mu shape must match x")
        if not isinstance(cond, torch.Tensor) or tuple(cond.shape) != tuple(x.shape):
            raise ValueError("variable Flow condition shape must match x")
        if not isinstance(t, torch.Tensor) or t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("variable Flow t must have one value per physical row")
        if not isinstance(spks, torch.Tensor) or spks.ndim != 2 or spks.shape[0] != x.shape[0]:
            raise ValueError("variable Flow speaker tensor must have one row per input")

        batch = int(x.shape[0])
        if attention_cache_lengths is None:
            raise ValueError("attention_cache_lengths is required for variable Flow")
        lengths = tuple(int(value) for value in attention_cache_lengths)
        if len(lengths) != batch or any(value < 0 for value in lengths):
            raise ValueError("attention cache lengths must match the physical batch")

        input_cnn_cache = cnn_cache if isinstance(cnn_cache, torch.Tensor) else None
        input_att_cache = att_cache if isinstance(att_cache, torch.Tensor) else None
        if att_cache is None:
            if any(lengths):
                raise ValueError("non-zero attention cache length requires a cache tensor")
            max_att_len = 0
        else:
            if not isinstance(att_cache, torch.Tensor) or att_cache.ndim != 5:
                raise ValueError("variable attention cache must have rank 5")
            if int(att_cache.shape[1]) != batch:
                raise ValueError("attention cache physical batch mismatch")
            max_att_len = int(att_cache.shape[3])
            if any(value > max_att_len for value in lengths):
                raise ValueError("attention cache length exceeds padded cache")

        chunk_size = int(x.shape[2])
        expected_mask_shape = (batch, chunk_size, max_att_len + chunk_size)
        if attention_valid_mask is None:
            attention_valid_mask = torch.zeros(
                expected_mask_shape,
                dtype=torch.bool,
                device=x.device,
            )
            attention_valid_mask[:, :, :chunk_size] = True
            if max_att_len:
                positions = torch.arange(max_att_len, device=x.device)
                row_lengths = torch.as_tensor(lengths, device=x.device)
                cache_valid = positions.unsqueeze(0) < row_lengths.unsqueeze(1)
                attention_valid_mask[:, :, chunk_size:] = cache_valid.unsqueeze(1)
        else:
            if tuple(attention_valid_mask.shape) != expected_mask_shape:
                raise ValueError("attention valid mask shape mismatch")
            if attention_valid_mask.dtype != torch.bool:
                raise ValueError("attention valid mask must be boolean")
            if attention_valid_mask.device != x.device:
                raise ValueError("attention valid mask device mismatch")

        # Keep the embedding and dynamic block path separate from the legacy
        # implementation.  In particular, never use the fixed worker buffers
        # or clear the row-specific mask here.
        t_emb = self.t_embedder(t).unsqueeze(1)
        x_emb = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks_emb = repeat(spks, "b c -> b c t", t=x_emb.shape[-1])
            x_emb = pack([x_emb, spks_emb], "b * t")[0]
        if cond is not None:
            x_emb = pack([x_emb, cond], "b * t")[0]

        if cnn_cache is not None:
            if not isinstance(cnn_cache, torch.Tensor) or cnn_cache.ndim != 4:
                raise ValueError("variable CNN cache must have rank 4")
            if int(cnn_cache.shape[1]) != batch:
                raise ValueError("CNN cache physical batch mismatch")

        if cnn_cache is None:
            cnn_cache = [None] * len(self.blocks)
        if att_cache is None:
            att_cache = [None] * len(self.blocks)

        logical_batch_size = batch // 2 if batch in (2, 4) else 0
        graph_tracking = bool(self.use_cuda_graph)
        if graph_tracking and logical_batch_size:
            self._record_cuda_graph_call(logical_batch_size)

        graph_key = (
            self._graph_key(logical_batch_size, chunk_size)
            if logical_batch_size
            else None
        )
        graph_max_cache = (
            self.max_size_chunk_by_logical_batch.get(graph_key, -1)
            if graph_key is not None
            else -1
        )
        graph_hit = bool(
            graph_tracking
            and logical_batch_size in getattr(self, "cuda_graph_logical_batch_sizes", ())
            and graph_key in getattr(self, "graph_chunk_by_logical_batch", {})
            and max_att_len <= graph_max_cache
            and (
                input_cnn_cache is None
                or tuple(input_cnn_cache.shape) == (16, batch, 1024, 2)
            )
            and (input_att_cache is None or tuple(input_att_cache.shape[:3]) == (16, batch, 8))
        )
        if graph_hit:
            self._cuda_graph_stats["graph_replay_calls"] += 1
            by_batch = self._cuda_graph_stats.get(
                "graph_replay_calls_by_logical_batch"
            )
            if by_batch is not None:
                key = str(logical_batch_size)
                by_batch[key] = int(by_batch.get(key, 0)) + 1
            self._cuda_graph_stats["eligible_calls"] = int(
                self._cuda_graph_stats.get("eligible_calls", 0)
            ) + 1
            try:
                return self._replay_cuda_graph(
                    logical_batch_size=logical_batch_size,
                    x=x_emb,
                    t=t_emb,
                    mask=attention_valid_mask,
                    cnn_cache=input_cnn_cache,
                    att_cache=input_att_cache,
                    attention_cache_length=max_att_len,
                )
            except (KeyError, ValueError):
                self._record_cuda_graph_fallback(
                    "graph_input_shape", logical_batch_size
                )
        elif graph_tracking and logical_batch_size:
            if graph_key not in getattr(self, "graph_chunk_by_logical_batch", {}):
                reason = "logical_batch_not_captured" if logical_batch_size not in getattr(
                    self, "cuda_graph_logical_batch_sizes", ()
                ) else "chunk_not_captured"
            elif max_att_len > graph_max_cache:
                reason = "graph_cache_limit"
            elif input_cnn_cache is None or tuple(input_cnn_cache.shape) != (16, batch, 1024, 2):
                reason = "cnn_cache_shape"
            else:
                reason = "predicate_unknown"
            self._record_cuda_graph_fallback(reason, logical_batch_size)

        return self.blocks_forward_chunk(
            x_emb,
            t_emb,
            attention_valid_mask,
            cnn_cache,
            att_cache,
            None,
            None,
        )

    def blocks_forward_chunk(self, x, t, mask, cnn_cache=None, att_cache=None, cnn_cache_buffer=None, att_cache_buffer=None):
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        new_cnn_caches = []
        new_att_caches = []
        for b_idx, block in enumerate(self.blocks):
            x, this_new_cnn_cache, this_new_att_cache \
                = block.forward_chunk(x, t, cnn_cache[b_idx], att_cache[b_idx], mask)
            if cnn_cache_buffer is None or att_cache_buffer is None:
                new_cnn_caches.append(this_new_cnn_cache)
                new_att_caches.append(this_new_att_cache)
            else:
                cnn_cache_buffer[b_idx] = this_new_cnn_cache
                att_cache_buffer[b_idx][:, :, :this_new_att_cache.shape[2], :] = this_new_att_cache
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        if cnn_cache_buffer is None or att_cache_buffer is None:
            return x, torch.stack(new_cnn_caches, dim=0), torch.stack(new_att_caches, dim=0)
        return x
