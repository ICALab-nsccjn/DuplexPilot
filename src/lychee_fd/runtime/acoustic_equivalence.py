"""Pure CPU metrics for layered PCM equivalence checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PcmEquivalenceThresholds:
    normalized_rmse_max: float = 0.02
    correlation_min: float = 0.99
    snr_db_min: float = 34.0

    def __post_init__(self) -> None:
        if self.normalized_rmse_max < 0:
            raise ValueError("normalized_rmse_max must be non-negative")
        if not -1.0 <= self.correlation_min <= 1.0:
            raise ValueError("correlation_min must be in [-1, 1]")


@dataclass(frozen=True)
class PcmEquivalenceMetrics:
    sample_rate: int
    migrated_sample_rate: int
    sample_count: int
    migrated_sample_count: int
    same_sample_rate: bool
    same_signal_length: bool
    finite: bool
    bitwise_equal: bool
    normalized_rmse: float
    correlation: float
    snr_db: float
    failure_reasons: tuple[str, ...]
    thresholds: PcmEquivalenceThresholds
    passed: bool


def _as_pcm(payload: bytes | bytearray | memoryview | Sequence[int]) -> np.ndarray:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        if len(raw) % 2:
            raise ValueError("PCM16 payload must contain an even number of bytes")
        return np.frombuffer(raw, dtype="<i2").astype(np.float64)
    array = np.asarray(payload, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("PCM must be one-dimensional")
    return array


def _correlation(reference: np.ndarray, migrated: np.ndarray) -> float:
    if np.array_equal(reference, migrated):
        return 1.0
    if reference.size == 0:
        return float("nan")
    reference_centered = reference - reference.mean()
    migrated_centered = migrated - migrated.mean()
    denominator = np.linalg.norm(reference_centered) * np.linalg.norm(migrated_centered)
    if denominator == 0:
        return 0.0
    return float(np.dot(reference_centered, migrated_centered) / denominator)


def compare_pcm_equivalence(
    reference_pcm,
    migrated_pcm,
    sample_rate: int,
    migrated_sample_rate: int,
    thresholds: PcmEquivalenceThresholds | None = None,
) -> PcmEquivalenceMetrics:
    """Compare PCM without synchronizing or touching a GPU execution path."""
    thresholds = thresholds or PcmEquivalenceThresholds()
    reference = _as_pcm(reference_pcm)
    migrated = _as_pcm(migrated_pcm)
    same_sample_rate = int(sample_rate) == int(migrated_sample_rate)
    same_signal_length = reference.size == migrated.size and reference.size > 0
    finite = bool(np.isfinite(reference).all() and np.isfinite(migrated).all())
    bitwise_equal = bool(
        same_signal_length and np.array_equal(reference, migrated)
    )

    if same_signal_length:
        error = migrated - reference
        normalized_rmse = float(np.sqrt(np.mean(np.square(error))) / 32768.0)
        correlation = _correlation(reference, migrated)
        error_power = float(np.mean(np.square(error)))
        signal_power = float(np.mean(np.square(reference)))
        if error_power == 0:
            snr_db = float("inf")
        elif signal_power == 0:
            snr_db = float("-inf")
        else:
            snr_db = float(10.0 * math.log10(signal_power / error_power))
    else:
        normalized_rmse = float("inf")
        correlation = float("nan")
        snr_db = float("-inf")

    failure_reasons: list[str] = []
    if not same_sample_rate:
        failure_reasons.append("sample_rate")
    if not same_signal_length:
        failure_reasons.append("signal_length")
    if not finite:
        failure_reasons.append("non_finite")
    if normalized_rmse > thresholds.normalized_rmse_max:
        failure_reasons.append("normalized_rmse")
    if not math.isfinite(correlation) or correlation < thresholds.correlation_min:
        failure_reasons.append("correlation")
    if snr_db < thresholds.snr_db_min:
        failure_reasons.append("snr_db")

    return PcmEquivalenceMetrics(
        sample_rate=int(sample_rate),
        migrated_sample_rate=int(migrated_sample_rate),
        sample_count=int(reference.size),
        migrated_sample_count=int(migrated.size),
        same_sample_rate=same_sample_rate,
        same_signal_length=same_signal_length,
        finite=finite,
        bitwise_equal=bitwise_equal,
        normalized_rmse=normalized_rmse,
        correlation=correlation,
        snr_db=snr_db,
        failure_reasons=tuple(failure_reasons),
        thresholds=thresholds,
        passed=not failure_reasons,
    )
