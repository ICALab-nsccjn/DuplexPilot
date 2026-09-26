"""Fail-closed configuration for the optional row-aware model driver.

The online benchmark has two deliberately different controls: disabling the
row-aware driver means a real singleton model step, while enabling it permits
an explicit logical cap of one or two rows.  Keeping parsing here avoids a
``B_model`` label that does not actually constrain the engine.
"""

from __future__ import annotations

import os
from typing import Any


ROW_MODEL_BATCH_ENV = "LYCHEEFD_ROW_AWARE_MODEL_MAX_BATCH_SIZE"


def resolve_row_model_batch_size(
    enabled: bool,
    value: Any | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> int:
    """Resolve the physical logical-row cap for the model execution plane.

    ``enabled=False`` is intentionally always a singleton, even if a stale
    environment variable requests two rows.  When enabled, the default is
    two, and only one or two are accepted.  Invalid explicit configuration is
    rejected before a model engine is constructed.
    """
    if isinstance(enabled, bool) is False:
        raise TypeError("enabled must be a bool")
    if not enabled:
        return 1
    if value is None:
        source = os.environ if environ is None else environ
        value = source.get(ROW_MODEL_BATCH_ENV, "2")
    if isinstance(value, bool):
        raise ValueError(f"{ROW_MODEL_BATCH_ENV} must be 1 or 2")
    try:
        cap = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ROW_MODEL_BATCH_ENV} must be 1 or 2") from exc
    if cap not in (1, 2):
        raise ValueError(f"{ROW_MODEL_BATCH_ENV} must be 1 or 2")
    return cap


__all__ = ["ROW_MODEL_BATCH_ENV", "resolve_row_model_batch_size"]
