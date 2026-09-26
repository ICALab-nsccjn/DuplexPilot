"""Optional, failure-safe NVTX markers for APR/acoustic diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


_KNOWN_RANGES = frozenset(
    {
        "APR_CHECKPOINT",
        "APR_RESTORE",
        "STREAMING_DECODER",
        "TOKEN2WAV_FLOW",
        "TOKEN2WAV_HIFT",
        "VOCODER",
    }
)


def _load_provider() -> Any | None:
    try:
        import torch

        provider = getattr(getattr(torch, "cuda", None), "nvtx", None)
        if provider is not None and hasattr(provider, "range_push") and hasattr(provider, "range_pop"):
            return provider
    except Exception:
        pass
    try:
        import nvtx as provider

        if hasattr(provider, "range_push") and hasattr(provider, "range_pop"):
            return provider
    except Exception:
        return None
    return None


def _valid_name(name: str) -> str:
    value = str(name)
    if value not in _KNOWN_RANGES:
        raise ValueError(f"unknown NVTX range: {value}")
    return value


@contextmanager
def range(name: str) -> Iterator[None]:
    """Push/pop a known range, or no-op when NVTX is unavailable."""
    value = _valid_name(name)
    provider = _load_provider()
    if provider is None:
        yield
        return
    try:
        provider.range_push(value)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            provider.range_pop()
        except Exception:
            pass


def mark(name: str) -> None:
    """Emit a zero-duration marker using the same stable range name."""
    with range(name):
        pass
