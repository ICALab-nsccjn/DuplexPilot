"""Fail-open, observation-only tracing for N>Bmax progression diagnostics."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional


SCHEMA = "lychee-n4-progress-v1"
TRACE_ENV = "LYCHEEFD_N4_PROGRESS_TRACE_PATH"
_WRITE_LOCK = threading.Lock()


def _get(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _length(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(len(value))
    except (TypeError, ValueError):
        pass
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def state_signature(state: Any) -> dict[str, Any]:
    """Describe only fields used by the frozen compatibility contracts."""
    if state is None:
        return {
            "observable": False,
            "request_id": None,
            "lengths": {"text": None, "stoken": None, "control": None},
            "phase": None,
            "interrupt_active": None,
            "finished": None,
            "row_aware_enabled": None,
            "runtime_mode": None,
        }
    request_id = _get(state, "session_id") or _get(state, "request_id")
    lengths = {
        "text": _length(_get(state, "text_input_ids")),
        "stoken": _length(_get(state, "stoken_input_ids")),
        "control": _length(_get(state, "control_input_ids")),
    }
    return {
        "observable": True,
        "request_id": str(request_id) if request_id not in (None, "") else None,
        "lengths": lengths,
        "phase": str(_get(state, "phase", "speaking")),
        "interrupt_active": bool(_get(state, "interrupt_active", False)),
        "finished": bool(_get(state, "finished", False)),
        "row_aware_enabled": bool(_get(state, "row_aware_enabled", False)),
        "runtime_mode": _get(state, "runtime_mode"),
    }


def classify_pair(left: Any, right: Any) -> dict[str, Any]:
    """Evaluate live state descriptors without influencing runtime selection."""
    left_sig = state_signature(left)
    right_sig = state_signature(right)
    observable = bool(left_sig["observable"] and right_sig["observable"])
    complete_histories = observable and all(
        value is not None
        for signature in (left_sig, right_sig)
        for value in signature["lengths"].values()
    )
    distinct_ids = bool(
        left_sig["request_id"]
        and right_sig["request_id"]
        and left_sig["request_id"] != right_sig["request_id"]
    )
    unfinished = observable and not left_sig["finished"] and not right_sig["finished"]
    if complete_histories:
        exact: Optional[bool] = bool(
            left_sig["lengths"] == right_sig["lengths"]
            and left_sig["phase"] == right_sig["phase"]
            and left_sig["interrupt_active"] == right_sig["interrupt_active"]
            and unfinished
        )
    else:
        exact = None
    virtualizable = bool(
        complete_histories
        and distinct_ids
        and unfinished
        and left_sig["row_aware_enabled"]
        and right_sig["row_aware_enabled"]
    )
    restored = bool(exact is False and virtualizable)
    return {
        "left_state": left_sig,
        "right_state": right_sig,
        "exact_compatible": exact,
        "virtualizable_compatible": virtualizable,
        "restored_opportunity": restored,
        "policy_can_pack_result": "NOT_OBSERVABLE",
    }


def _group_request_id(group: Any) -> str:
    request_id = _get(group, "request_id")
    return str(request_id) if request_id not in (None, "") else ""


def _group_state(group: Any) -> Any:
    return _get(group, "multihead_request_state")


def _selected_group(item: Any) -> Any:
    return _get(item, "seq_group", item)


def build_admission_event(
    *,
    running: Any,
    waiting: Any,
    swapped: Any,
    selected: Any,
    opportunity_id: str,
) -> dict[str, Any]:
    """Build a decision record from existing queues without mutating them."""
    active_groups = list(running)
    pending_groups = list(waiting)
    swapped_groups = list(swapped)
    candidate_groups = active_groups + swapped_groups + pending_groups
    selected_groups = [_selected_group(item) for item in list(selected)]
    candidate_ids = [_group_request_id(group) for group in candidate_groups]
    selected_ids = [_group_request_id(group) for group in selected_groups]
    eligible_pair = bool(
        len(candidate_groups) >= 2
        and candidate_ids[0]
        and candidate_ids[1]
        and candidate_ids[0] != candidate_ids[1]
        and _group_state(candidate_groups[0]) is not None
        and _group_state(candidate_groups[1]) is not None
    )
    if eligible_pair:
        pair = classify_pair(
            _group_state(candidate_groups[0]),
            _group_state(candidate_groups[1]),
        )
        exact = pair["exact_compatible"]
        virtualizable = pair["virtualizable_compatible"]
        restored = pair["restored_opportunity"]
        pair_states = [pair["left_state"], pair["right_state"]]
    else:
        exact = virtualizable = restored = None
        pair_states = [
            state_signature(_group_state(group))
            for group in candidate_groups[:2]
        ]
    selected_set = set(selected_ids)
    return {
        "opportunity_id": str(opportunity_id),
        "active_request_ids": [_group_request_id(group) for group in active_groups],
        "pending_request_ids": [_group_request_id(group) for group in pending_groups],
        "swapped_request_ids": [_group_request_id(group) for group in swapped_groups],
        "candidate_request_ids": candidate_ids,
        "selected_request_ids": selected_ids,
        "physical_batch_size": len(selected_ids),
        "eligible_pair_opportunity": eligible_pair,
        "candidate_pair_states": pair_states,
        "exact_compatible": exact,
        "virtualizable_compatible": virtualizable,
        "restored_opportunity": restored,
        "policy_can_pack_result": "NOT_OBSERVABLE",
        "reason_selected": {
            request_id: "SELECTED_BY_EXISTING_SCHEDULER"
            for request_id in selected_ids
        },
        "reason_not_selected": {
            request_id: "NOT_SELECTED_BY_EXISTING_SCHEDULER"
            for request_id in candidate_ids
            if request_id not in selected_set
        },
    }


def build_physical_batch_event(rows: Any) -> dict[str, Any]:
    """Describe the row list already constructed by the model runner."""
    copied_rows = list(rows or [])
    signatures = [state_signature(row) for row in copied_rows]
    if len(copied_rows) >= 2:
        pair = classify_pair(copied_rows[0], copied_rows[1])
        exact = pair["exact_compatible"]
        virtualizable = pair["virtualizable_compatible"]
        restored = pair["restored_opportunity"]
    else:
        exact = virtualizable = restored = None
    return {
        "physical_batch_size": len(copied_rows),
        "physical_rows": {
            str(index): str(signature.get("request_id") or "")
            for index, signature in enumerate(signatures)
        },
        "row_states": signatures,
        "exact_compatible": exact,
        "virtualizable_compatible": virtualizable,
        "restored_opportunity": restored,
        "policy_can_pack_result": "NOT_OBSERVABLE",
    }


def write_event(
    event_type: str,
    *,
    path: Optional[str] = None,
    source_file: Optional[str] = None,
    source_function: Optional[str] = None,
    **fields: Any,
) -> bool:
    """Append one JSONL event and never turn trace failure into serving failure."""
    target = str(path or os.getenv(TRACE_ENV, "")).strip()
    if not target:
        return False
    record = {
        "schema": SCHEMA,
        "event_type": str(event_type),
        "timestamp_monotonic_ns": int(time.monotonic_ns()),
        "source_file": source_file,
        "source_function": source_function,
    }
    record.update(dict(fields))
    try:
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        target_path = Path(target)
        with _WRITE_LOCK:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
        return True
    except Exception:
        return False
