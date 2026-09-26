"""Metadata-only model work fingerprint contract tests."""

from __future__ import annotations

import hashlib

import pytest

from lychee_fd.runtime.model_execution_trace import (
    ModelWorkFingerprint,
)


def test_fingerprint_is_deterministic_and_order_sensitive():
    first = ModelWorkFingerprint()
    second = ModelWorkFingerprint()
    for acc in (first, second):
        acc.update(
            sequence_no=0,
            text_token=11,
            stoken_token=21,
            control_token=31,
        )
        acc.update(
            sequence_no=1,
            text_token=12,
            stoken_token=22,
            control_token=32,
        )

    assert first.generated_token_count == 2
    assert first.digest() == second.digest()
    assert len(first.digest()) == hashlib.sha256().digest_size * 2

    reordered = ModelWorkFingerprint()
    reordered.update(sequence_no=0, text_token=12, stoken_token=22, control_token=32)
    reordered.update(sequence_no=1, text_token=11, stoken_token=21, control_token=31)
    assert reordered.digest() != first.digest()


def test_fingerprint_rejects_invalid_metadata():
    acc = ModelWorkFingerprint()
    with pytest.raises(ValueError):
        acc.update(sequence_no=-1, text_token=1, stoken_token=2, control_token=3)
    with pytest.raises(ValueError):
        acc.update(sequence_no=1, text_token=None, stoken_token=2, control_token=3)


def test_fingerprint_summary_contains_only_bounded_metadata():
    acc = ModelWorkFingerprint()
    acc.update(sequence_no=0, text_token=1, stoken_token=2, control_token=3)
    summary = acc.summary(
        request_id="req-a",
        generation_id=4,
        termination_reason="eos",
        request_finished_reason=None,
    )
    assert summary == {
        "request_id": "req-a",
        "generation_id": 4,
        "generated_token_count": 1,
        "token_sequence_sha256": acc.digest(),
        "termination_reason": "eos",
        "request_finished_reason": None,
    }
    assert "tokens" not in summary
