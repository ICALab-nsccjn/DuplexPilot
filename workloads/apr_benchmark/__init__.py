"""Deterministic workload suite for APR advantage characterization."""

from .generator import (
    BASELINES,
    EVENT_TYPES,
    SUPPORTED_WORKLOADS,
    WorkloadEvent,
    WorkloadTrace,
    build_experiment_matrix,
    generate_workload,
    progression_rounds,
    summarize_trace,
    trace_hash,
    write_trace_jsonl,
)

__all__ = [
    "BASELINES",
    "EVENT_TYPES",
    "SUPPORTED_WORKLOADS",
    "WorkloadEvent",
    "WorkloadTrace",
    "build_experiment_matrix",
    "generate_workload",
    "progression_rounds",
    "summarize_trace",
    "trace_hash",
    "write_trace_jsonl",
]
