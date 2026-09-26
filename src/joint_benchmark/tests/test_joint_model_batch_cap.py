"""The joint harness must configure a real model-driver batch cap."""

from __future__ import annotations

import pytest

from lychee_fd.runtime.model_batch_config import resolve_row_model_batch_size


def test_disabled_model_plane_is_a_real_single_row_cap(monkeypatch):
    monkeypatch.delenv("LYCHEEFD_ROW_AWARE_MODEL_MAX_BATCH_SIZE", raising=False)
    assert resolve_row_model_batch_size(False) == 1


def test_enabled_model_plane_defaults_to_two_and_accepts_explicit_one(monkeypatch):
    monkeypatch.delenv("LYCHEEFD_ROW_AWARE_MODEL_MAX_BATCH_SIZE", raising=False)
    assert resolve_row_model_batch_size(True) == 2
    monkeypatch.setenv("LYCHEEFD_ROW_AWARE_MODEL_MAX_BATCH_SIZE", "1")
    assert resolve_row_model_batch_size(True) == 1


def test_model_batch_cap_fails_closed_for_invalid_values(monkeypatch):
    monkeypatch.setenv("LYCHEEFD_ROW_AWARE_MODEL_MAX_BATCH_SIZE", "4")
    with pytest.raises(ValueError):
        resolve_row_model_batch_size(True)

