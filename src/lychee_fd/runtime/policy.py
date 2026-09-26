"""Fail-closed selector for frozen Lychee execution policies.

The selector is deliberately independent from scheduling, packing, session
state, and Token2Wav.  It only describes which already-frozen execution path
the official online runtime should invoke.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RuntimePolicyError(ValueError):
    """Raised when an online execution policy is not explicitly supported."""


class RuntimeMode(str, Enum):
    NATIVE = "native"
    RSV_B1 = "rsv_b1"
    DYNAMIC_EXACT = "dynamic_exact"
    DYNAMIC_VIRTUALIZED = "dynamic_virtualized"


@dataclass(frozen=True)
class RuntimePolicy:
    mode: RuntimeMode
    row_aware: bool
    uses_shared_execution: bool


def resolve_runtime_policy(runtime_mode: Optional[str]) -> RuntimePolicy:
    """Resolve a request's mode without silent fallback.

    An omitted mode is the sole compatibility default and selects the current
    native path.  Any supplied but unknown value is rejected before session
    construction.
    """
    if runtime_mode is None or str(runtime_mode).strip() == "":
        mode = RuntimeMode.NATIVE
    else:
        raw = str(runtime_mode).strip().lower()
        try:
            mode = RuntimeMode(raw)
        except ValueError as exc:
            raise RuntimePolicyError(
                f"unknown runtime_mode: {runtime_mode!r}"
            ) from exc

    return RuntimePolicy(
        mode=mode,
        row_aware=mode is not RuntimeMode.NATIVE,
        uses_shared_execution=mode in {
            RuntimeMode.DYNAMIC_EXACT,
            RuntimeMode.DYNAMIC_VIRTUALIZED,
        },
    )
