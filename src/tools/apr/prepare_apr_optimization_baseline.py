"""Build a non-claim optimization baseline from frozen APR evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


BASELINE_SOURCE = "reports/apr_workload_suite/APR_WORKLOAD_ADVANTAGE_REPORT.md"
BASELINE_CONCURRENCY = [4, 8, 16]
REQUIRED_IDENTITY = (
    "source_commit",
    "model_checkpoint",
    "token2wav_checkpoint",
    "gpu_mapping",
    "worker_count",
    "measurement_schema_version",
)


class BaselineContractError(ValueError):
    """Raised when frozen baseline metadata is incomplete or inconsistent."""


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        environment = {}
    assets = manifest.get("input_assets")
    if not isinstance(assets, Mapping):
        assets = {}
    runtime = manifest.get("runtime_config")
    if not isinstance(runtime, Mapping):
        runtime = {}
    values = {
        "source_commit": _first(
            manifest,
            "source_commit",
            "git_commit",
        )
        or _first(environment, "git_commit"),
        "model_checkpoint": _first(manifest, "model_checkpoint")
        or _first(assets, "model_checkpoint")
        or _first(runtime, "model_path"),
        "token2wav_checkpoint": _first(manifest, "token2wav_checkpoint")
        or _first(assets, "token2wav_checkpoint")
        or _first(runtime, "token2wav_path"),
        "gpu_mapping": _first(manifest, "gpu_mapping"),
        "worker_count": _first(manifest, "worker_count")
        or _first(runtime, "worker_count"),
        "measurement_schema_version": _first(
            manifest,
            "measurement_schema_version",
        ),
    }
    missing = [key for key, value in values.items() if value in (None, "")]
    if missing:
        raise BaselineContractError(
            f"baseline manifest is missing identity fields: {', '.join(missing)}"
        )
    return values


def _parse_count(report_text: str, label: str, default: int | None = None) -> int:
    pattern = rf"{re.escape(label)}\s*:\s*`?(\d+)"
    match = re.search(pattern, report_text)
    if match is None:
        if default is not None:
            return default
        raise BaselineContractError(f"baseline report is missing {label}")
    return int(match.group(1))


def build_baseline_summary(
    report_path: str | Path,
    manifest_paths: Iterable[str | Path],
) -> dict[str, Any]:
    report_path = Path(report_path)
    report_text = report_path.read_text(encoding="utf-8")
    manifests = []
    for path in manifest_paths:
        manifest_path = Path(path)
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    if not manifests:
        raise BaselineContractError("at least one baseline manifest is required")

    identities = [_identity(manifest) for manifest in manifests]
    first_identity = identities[0]
    for identity in identities[1:]:
        if identity != first_identity:
            raise BaselineContractError("baseline manifest identities disagree")

    systems = sorted(
        {
            str(manifest.get("system") or manifest.get("baseline"))
            for manifest in manifests
            if manifest.get("system") or manifest.get("baseline")
        }
    )
    if not systems:
        raise BaselineContractError("baseline manifests have no system identity")

    rows = []
    for manifest, identity in zip(manifests, identities):
        metrics = manifest.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = {}
        rows.append(
            {
                "system": str(manifest.get("system") or manifest.get("baseline")),
                "concurrency": int(manifest.get("concurrency", 0) or 0),
                "throughput_sps": metrics.get("throughput_sps", "NOT_MEASURED"),
                "checkpoint_overhead_s": metrics.get(
                    "checkpoint_overhead_s", "NOT_MEASURED"
                ),
                **identity,
            }
        )

    invalid_count = _parse_count(report_text, "Invalid counted runs", default=0)
    valid_match = re.search(
        r"Valid counted runs\s*:\s*`?(\d+)\s*/\s*(\d+)", report_text
    )
    if valid_match is None:
        raise BaselineContractError("baseline report is missing Valid counted runs")
    return {
        "systems": systems,
        "concurrency": list(BASELINE_CONCURRENCY),
        "source_report": BASELINE_SOURCE,
        "valid_counted_runs": int(valid_match.group(1)),
        "expected_counted_runs": int(valid_match.group(2)),
        "invalid_counted_runs": invalid_count,
        "rows": rows,
        "identity": first_identity,
    }


def render_baseline_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# APR Optimization Baseline",
        "",
        "This document freezes the pre-optimization evidence. It is a diagnostic reference and does not establish an APR speedup claim.",
        "",
        "## Scope",
        "",
        f"- Source report: `{summary['source_report']}`",
        f"- Systems: `{', '.join(summary['systems'])}`",
        f"- Concurrency: `{', '.join(str(value) for value in summary['concurrency'])}`",
        f"- Valid counted runs: `{summary['valid_counted_runs']} / {summary['expected_counted_runs']}`",
        f"- Invalid counted runs: `{summary['invalid_counted_runs']}`",
        "- Existing performance advantage status: `NOT_ESTABLISHED`.",
        "",
        "## Frozen environment",
        "",
    ]
    for key, value in summary["identity"].items():
        lines.append(f"- {key}: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Baseline handling",
            "",
            "The historical workload-suite rows remain separate from all future profiling and optimization rows. Missing measurements remain `NOT_MEASURED`; they are never converted to zero.",
            "",
            "No code or serving semantics are changed by this baseline artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def write_baseline(
    report_path: str | Path,
    manifest_paths: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    summary = build_baseline_summary(report_path, manifest_paths)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_baseline_markdown(summary), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_baseline(args.report, args.manifest, args.output), indent=2))
