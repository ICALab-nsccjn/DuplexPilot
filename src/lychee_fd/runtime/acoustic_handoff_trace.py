"""Observation-only probes for the acoustic-token handoff boundary.

The writer is deliberately fail-open: diagnostics must never change serving
semantics or turn a measurement failure into an inference failure.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


_TRACE_SCHEMA = "lychee-acoustic-handoff-v1"
_WRITE_LOCK = threading.Lock()


def _small_values(value: Any, limit: int = 32):
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) > limit:
            return None
        out = []
        for item in values:
            if isinstance(item, (list, tuple)):
                nested = _small_values(item, limit=limit)
                if nested is None:
                    return None
                out.append(nested)
            else:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    out.append(str(item))
        return out
    return None


def describe_value(value: Any, *, limit: int = 32) -> Dict[str, Any]:
    """Return JSON-safe value/shape/type metadata for a handoff probe."""
    result: Dict[str, Any] = {
        "python_type": type(value).__name__,
        "is_none": value is None,
    }
    if value is None:
        result["container"] = "none"
        return result

    shape = getattr(value, "shape", None)
    if shape is not None and hasattr(value, "dtype"):
        result.update(
            {
                "container": "tensor",
                "shape": [int(x) for x in shape],
                "ndim": int(getattr(value, "ndim", len(shape))),
                "dtype": str(getattr(value, "dtype", "")),
                "device": str(getattr(value, "device", "")),
                "numel": int(value.numel()) if hasattr(value, "numel") else None,
            }
        )
        try:
            numel = int(value.numel())
            if numel <= limit:
                result["values"] = value.detach().reshape(-1).tolist()
        except Exception:
            pass
        return result

    if isinstance(value, (list, tuple)):
        result["container"] = type(value).__name__
        result["length"] = len(value)
        small = _small_values(value, limit=limit)
        if small is not None:
            result["values"] = small
        return result

    if isinstance(value, dict):
        result["container"] = "dict"
        result["length"] = len(value)
        result["keys"] = [str(k) for k in list(value)[:limit]]
        return result

    result["container"] = "scalar"
    try:
        result["value"] = int(value)
    except (TypeError, ValueError):
        result["value"] = str(value)
    return result


def write_event(payload: Dict[str, Any], *, path: Optional[str] = None) -> bool:
    """Append one structured probe record; return False on any I/O failure."""
    target = str(path or os.getenv("LYCHEEFD_ACOUSTIC_HANDOFF_TRACE_PATH", "")).strip()
    if not target:
        return False
    record: Dict[str, Any] = {
        "schema": _TRACE_SCHEMA,
        "timestamp_monotonic_ns": int(time.monotonic_ns()),
    }
    record.update(dict(payload or {}))
    try:
        encoded = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":"))
        with _WRITE_LOCK:
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
        return True
    except Exception:
        return False
