"""Paper-facing system specifications for the public baseline evaluation.

The registry is deliberately independent from the historical benchmark labels.
It prevents ``original_affinity`` (which historically used RSV/DSV) from being
silently presented as the native Lychee control in paper artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping


_ALLOWED_MODEL_MODES = frozenset({"native", "dynamic_virtualized"})
_ALLOWED_ACOUSTIC_MODES = frozenset({"fixed_affinity", "apr_step"})
_ALLOWED_FLOW_EXECUTION_MODES = frozenset({"sync", "async"})
_ALLOWED_FLOW_SCHEDULE_POLICIES = frozenset({"fifo", "deadline_aging"})
_ALLOWED_FLOW_STEP_POLICIES = frozenset({"fixed10", "residual_adaptive"})
_ALLOWED_FLOW_BATCH_VARIANTS = frozenset(
    {
        "exact",
        "variable_length",
        "chunk_padding",
        "mixed_step",
        "mixed_step_b4",
        "mixed_step_chunk_padding",
    }
)


@dataclass(frozen=True)
class PaperSystemSpec:
    """One immutable system configuration used by the common benchmark harness."""

    system_id: str
    model_runtime_mode: str
    acoustic_mode: str
    max_flow_batch_size: int
    flow_execution_mode: str = "sync"
    max_inflight_flow_steps: int = 1
    flow_schedule_policy: str = "fifo"
    flow_step_policy: str = "fixed10"
    adaptive_policy_id: str | None = None
    flow_batch_variant: str = "exact"

    def __post_init__(self) -> None:
        if not self.system_id or self.system_id.strip() != self.system_id:
            raise ValueError("system_id must be a non-empty normalized string")
        if self.model_runtime_mode not in _ALLOWED_MODEL_MODES:
            raise ValueError(
                f"unsupported model_runtime_mode: {self.model_runtime_mode!r}"
            )
        if self.acoustic_mode not in _ALLOWED_ACOUSTIC_MODES:
            raise ValueError(f"unsupported acoustic_mode: {self.acoustic_mode!r}")
        if isinstance(self.max_flow_batch_size, bool) or self.max_flow_batch_size not in (1, 2, 4):
            raise ValueError("max_flow_batch_size must be one of 1, 2, or 4")
        if self.acoustic_mode == "fixed_affinity" and self.max_flow_batch_size != 1:
            raise ValueError("fixed affinity cannot use Flow batch size > 1")
        if self.flow_execution_mode not in _ALLOWED_FLOW_EXECUTION_MODES:
            raise ValueError(
                f"unsupported flow_execution_mode: {self.flow_execution_mode!r}"
            )
        if (
            isinstance(self.max_inflight_flow_steps, bool)
            or self.max_inflight_flow_steps not in (1, 2, 4)
        ):
            raise ValueError("max_inflight_flow_steps must be one of 1, 2, or 4")
        if self.flow_schedule_policy not in _ALLOWED_FLOW_SCHEDULE_POLICIES:
            raise ValueError(
                f"unsupported flow_schedule_policy: {self.flow_schedule_policy!r}"
            )
        if self.flow_step_policy not in _ALLOWED_FLOW_STEP_POLICIES:
            raise ValueError(
                f"unsupported flow_step_policy: {self.flow_step_policy!r}"
            )
        if self.flow_batch_variant not in _ALLOWED_FLOW_BATCH_VARIANTS:
            raise ValueError(
                f"unsupported flow_batch_variant: {self.flow_batch_variant!r}"
            )
        if self.flow_execution_mode == "async" and self.max_flow_batch_size != 1:
            raise ValueError("async execution requires Flow batch size one")
        if (
            self.flow_step_policy == "residual_adaptive"
            and not self.adaptive_policy_id
        ):
            raise ValueError("adaptive_policy_id is required for residual_adaptive")
        if self.flow_step_policy == "fixed10" and self.adaptive_policy_id is not None:
            raise ValueError("adaptive_policy_id is only valid for residual_adaptive")
        if self.flow_batch_variant == "variable_length":
            if self.acoustic_mode != "apr_step":
                raise ValueError("variable-length batching requires apr_step acoustic mode")
            if self.max_flow_batch_size != 2:
                raise ValueError("variable-length V1 supports only Flow batch size two")
            if self.flow_execution_mode != "sync":
                raise ValueError("variable-length V1 requires synchronous execution")
        if self.flow_batch_variant == "chunk_padding":
            if self.acoustic_mode != "apr_step":
                raise ValueError("chunk-padding batching requires apr_step acoustic mode")
            if self.max_flow_batch_size != 2:
                raise ValueError("chunk-padding V1 supports only Flow batch size two")
            if self.flow_execution_mode != "sync":
                raise ValueError("chunk-padding batching requires synchronous execution")
        if self.flow_batch_variant == "mixed_step":
            if self.acoustic_mode != "apr_step":
                raise ValueError("mixed-step batching requires apr_step acoustic mode")
            if self.max_flow_batch_size != 2:
                raise ValueError("mixed-step V1 supports only Flow batch size two")
            if self.flow_execution_mode != "sync":
                raise ValueError("mixed-step V1 requires synchronous execution")
        if self.flow_batch_variant == "mixed_step_b4":
            if self.acoustic_mode != "apr_step":
                raise ValueError("mixed-step B4 requires apr_step acoustic mode")
            if self.max_flow_batch_size != 4:
                raise ValueError("mixed-step B4 requires Flow batch size four")
            if self.flow_execution_mode != "sync":
                raise ValueError("mixed-step B4 requires synchronous execution")
        if self.flow_batch_variant == "mixed_step_chunk_padding":
            if self.acoustic_mode != "apr_step":
                raise ValueError(
                    "mixed-step chunk-padding requires apr_step acoustic mode"
                )
            if self.max_flow_batch_size != 2:
                raise ValueError(
                    "mixed-step chunk-padding V1 supports only Flow batch size two"
                )
            if self.flow_execution_mode != "sync":
                raise ValueError(
                    "mixed-step chunk-padding requires synchronous execution"
                )
        if self.system_id == "lychee_native_affinity" and self.model_runtime_mode != "native":
            raise ValueError("native control must use native model runtime")


_PAPER_SYSTEMS = (
    PaperSystemSpec(
        "lychee_native_affinity", "native", "fixed_affinity", 1
    ),
    PaperSystemSpec(
        "rsv_dsv_affinity", "dynamic_virtualized", "fixed_affinity", 1
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_b1", "dynamic_virtualized", "apr_step", 1
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_batch_b2", "dynamic_virtualized", "apr_step", 2
    ),
)
_ASYNC_PAPER_SYSTEMS = (
    PaperSystemSpec(
        "rsv_dsv_apr_step_async_k2", "dynamic_virtualized", "apr_step", 1,
        "async", 2, "fifo", "fixed10", None,
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_deadline_k2", "dynamic_virtualized", "apr_step", 1,
        "async", 2, "deadline_aging", "fixed10", None,
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_async_adaptive", "dynamic_virtualized", "apr_step", 1,
        "async", 2, "deadline_aging", "residual_adaptive", "calibration-v1",
    ),
)
_EXPLORATORY_PAPER_SYSTEMS = (
    # B=4 is intentionally resolvable for a bounded feasibility experiment,
    # but is excluded from the primary iterator and paper waterfall.
    PaperSystemSpec(
        "rsv_dsv_apr_step_batch_b4_exploratory",
        "dynamic_virtualized",
        "apr_step",
        4,
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_variable_b2",
        "dynamic_virtualized",
        "apr_step",
        2,
        flow_batch_variant="variable_length",
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_chunk_padding_b2",
        "dynamic_virtualized",
        "apr_step",
        2,
        flow_batch_variant="chunk_padding",
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_mixed_b2",
        "dynamic_virtualized",
        "apr_step",
        2,
        flow_batch_variant="mixed_step",
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_mixed_b4_exploratory",
        "dynamic_virtualized",
        "apr_step",
        4,
        flow_batch_variant="mixed_step_b4",
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_step_mixed_chunk_padding_b2",
        "dynamic_virtualized",
        "apr_step",
        2,
        flow_batch_variant="mixed_step_chunk_padding",
    ),
)
# The joint (B_model, B_acoustic) matrix is benchmark-only.  Keeping these
# entries out of ``iter_paper_systems`` prevents them from silently changing
# the historical paper baseline iterator while still giving the harness a
# concrete, fail-closed system spec for each cell.
_JOINT_PAPER_SYSTEMS = (
    PaperSystemSpec(
        "rsv_dsv_apr_joint_j11",
        "dynamic_virtualized",
        "apr_step",
        1,
        flow_batch_variant="exact",
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_joint_j21",
        "dynamic_virtualized",
        "apr_step",
        1,
        flow_batch_variant="exact",
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_joint_j12",
        "dynamic_virtualized",
        "apr_step",
        2,
        flow_batch_variant="mixed_step_chunk_padding",
    ),
    PaperSystemSpec(
        "rsv_dsv_apr_joint_j22",
        "dynamic_virtualized",
        "apr_step",
        2,
        flow_batch_variant="mixed_step_chunk_padding",
    ),
)
PAPER_SYSTEMS: Mapping[str, PaperSystemSpec] = MappingProxyType(
    {
        spec.system_id: spec
        for spec in (
            _PAPER_SYSTEMS
            + _ASYNC_PAPER_SYSTEMS
            + _EXPLORATORY_PAPER_SYSTEMS
            + _JOINT_PAPER_SYSTEMS
        )
    }
)
LEGACY_LABELS: Mapping[str, str] = MappingProxyType(
    {"original_affinity": "rsv_dsv_affinity", "apr": "rsv_dsv_apr_step_b1"}
)


def iter_paper_systems() -> Iterator[PaperSystemSpec]:
    return iter(_PAPER_SYSTEMS)


def get_system_spec(system_id: str, *, allow_legacy: bool = False) -> PaperSystemSpec:
    """Resolve a paper system, failing closed for ambiguous historical labels."""
    if system_id in LEGACY_LABELS:
        if not allow_legacy:
            raise ValueError(
                f"legacy system label {system_id!r} is not a paper label; "
                "select an explicit B0-B3 system"
            )
        system_id = LEGACY_LABELS[system_id]
    try:
        return PAPER_SYSTEMS[system_id]
    except KeyError as exc:
        raise ValueError(f"unknown paper system: {system_id!r}") from exc
