"""Benchmark-only contracts for the independent model/acoustic batch axes.

The production serving paths deliberately do not infer a joint configuration
from a few loosely related flags.  This module gives the experiment harness a
small, immutable and fail-closed description of the four configurations used
by the joint ``(B_model, B_acoustic)`` study.

The contract is metadata only: it does not own a scheduler, tensors, model
state, or acoustic state.  Keeping it separate makes it possible to audit the
two execution planes without accidentally claiming that their local speedups
multiply.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


_MODEL_MODES = frozenset({"legacy_serialized", "row_aware_cap1", "row_aware_cap2"})
_ACOUSTIC_MODES = frozenset({"apr_step_b1", "mixed_chunk_padding_b2"})


def _checked_batch_size(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be 1 or 2")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be 1 or 2") from exc
    if result not in (1, 2):
        raise ValueError(f"{field} must be 1 or 2")
    return result


def _normal_joint_id(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw.startswith("J") and len(raw) == 3 and raw[1:] in {"11", "21", "12", "22"}:
        return raw
    raise ValueError(f"unsupported joint_id: {value!r}")


@dataclass(frozen=True)
class JointExecutionSpec:
    """Immutable description of one independent two-plane configuration.

    ``max_model_batch_size`` is the logical model-driver cap.  It is not a
    promise that every live step will contain that many rows.  Likewise,
    ``max_acoustic_batch_size`` describes the selected acoustic API, while
    actual formation is measured separately by the benchmark telemetry.
    """

    joint_id: str
    model_execution_mode: str
    acoustic_execution_mode: str
    max_model_batch_size: int
    max_acoustic_batch_size: int

    def __post_init__(self) -> None:
        normalized = _normal_joint_id(self.joint_id)
        object.__setattr__(self, "joint_id", normalized)
        if self.model_execution_mode not in _MODEL_MODES:
            raise ValueError(
                f"unsupported model_execution_mode: {self.model_execution_mode!r}"
            )
        if self.acoustic_execution_mode not in _ACOUSTIC_MODES:
            raise ValueError(
                f"unsupported acoustic_execution_mode: {self.acoustic_execution_mode!r}"
            )
        model_batch = _checked_batch_size(self.max_model_batch_size, "max_model_batch_size")
        acoustic_batch = _checked_batch_size(
            self.max_acoustic_batch_size, "max_acoustic_batch_size"
        )
        object.__setattr__(self, "max_model_batch_size", model_batch)
        object.__setattr__(self, "max_acoustic_batch_size", acoustic_batch)

        expected_model = 2 if self.model_execution_mode == "row_aware_cap2" else 1
        expected_acoustic = 2 if self.acoustic_execution_mode == "mixed_chunk_padding_b2" else 1
        if model_batch != expected_model:
            raise ValueError(
                f"{self.model_execution_mode} requires model batch {expected_model}"
            )
        if acoustic_batch != expected_acoustic:
            raise ValueError(
                f"{self.acoustic_execution_mode} requires acoustic batch {expected_acoustic}"
            )

        expected_joint = {
            "J11": ("legacy_serialized", "apr_step_b1"),
            "J21": ("row_aware_cap2", "apr_step_b1"),
            "J12": ("legacy_serialized", "mixed_chunk_padding_b2"),
            "J22": ("row_aware_cap2", "mixed_chunk_padding_b2"),
        }[normalized]
        if (self.model_execution_mode, self.acoustic_execution_mode) != expected_joint:
            raise ValueError(
                f"joint {normalized} has inconsistent execution modes: "
                f"{self.model_execution_mode!r}, {self.acoustic_execution_mode!r}"
            )

    @property
    def paper_system_id(self) -> str:
        return f"rsv_dsv_apr_joint_{self.joint_id.lower()}"

    @property
    def model_batch_size(self) -> int:
        return self.max_model_batch_size

    @property
    def acoustic_batch_size(self) -> int:
        return self.max_acoustic_batch_size

    @property
    def vllm_max_num_seqs(self) -> int:
        """Physical vLLM row cap used by the benchmark service.

        The logical concurrency ``N`` is deliberately not used here.  Doing
        so would let the legacy controls batch prefill rows underneath a
        ``B_model=1`` label and would make the four-grid comparison invalid.
        Queued sessions remain concurrent; only one or two rows may execute
        in a model engine step, according to the registered joint cell.
        """
        return self.max_model_batch_size

    def payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload suitable for a request manifest."""
        return {
            "joint_id": self.joint_id,
            "paper_system_id": self.paper_system_id,
            "model_execution_mode": self.model_execution_mode,
            "acoustic_execution_mode": self.acoustic_execution_mode,
            "max_model_batch_size": self.max_model_batch_size,
            "max_acoustic_batch_size": self.max_acoustic_batch_size,
        }

    def validate_payload(self, payload: Mapping[str, Any]) -> None:
        """Fail closed if a serialized contract was modified in transit."""
        if not isinstance(payload, Mapping):
            raise ValueError("joint payload must be a mapping")
        expected = self.payload()
        for key, value in expected.items():
            if key not in payload:
                raise ValueError(f"joint payload is missing {key!r}")
            if payload[key] != value:
                raise ValueError(
                    f"joint payload field {key!r} diverged: "
                    f"expected={value!r} got={payload[key]!r}"
                )


_JOINT_SPECS = tuple(
    JointExecutionSpec(
        "J11", "legacy_serialized", "apr_step_b1", 1, 1
    )
    for _ in (0,)
) + (
    JointExecutionSpec("J21", "row_aware_cap2", "apr_step_b1", 2, 1),
    JointExecutionSpec("J12", "legacy_serialized", "mixed_chunk_padding_b2", 1, 2),
    JointExecutionSpec("J22", "row_aware_cap2", "mixed_chunk_padding_b2", 2, 2),
)

# The public mapping intentionally exposes only the canonical short IDs.  A
# mapping proxy prevents benchmark code from silently changing a registered
# configuration after a run has started.
JOINT_EXECUTION_SPECS: Mapping[str, JointExecutionSpec] = MappingProxyType(
    {spec.joint_id: spec for spec in _JOINT_SPECS}
)

_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        spec.paper_system_id: spec.joint_id
        for spec in _JOINT_SPECS
    }
    | {
        f"apr_joint_{spec.joint_id.lower()}": spec.joint_id
        for spec in _JOINT_SPECS
    }
)


def get_joint_execution_spec(value: str | JointExecutionSpec) -> JointExecutionSpec:
    """Resolve a canonical ID or an explicit benchmark system alias."""
    if isinstance(value, JointExecutionSpec):
        return value
    raw = str(value or "").strip()
    canonical = raw.upper()
    if canonical in JOINT_EXECUTION_SPECS:
        return JOINT_EXECUTION_SPECS[canonical]
    alias = _ALIASES.get(raw.lower())
    if alias is None:
        raise ValueError(f"unknown joint execution spec: {value!r}")
    return JOINT_EXECUTION_SPECS[alias]


__all__ = [
    "JointExecutionSpec",
    "JOINT_EXECUTION_SPECS",
    "get_joint_execution_spec",
]
