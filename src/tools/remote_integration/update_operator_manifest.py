from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


WT = Path("/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-aware-flow-operator-e2e")
RESULTS = Path("/mnt/DuplexPilot/data/DuplexPilot/results/apr_aware_flow_operator_e2e_20260831")


def digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            h.update(block)
    return h.hexdigest(), size


def main() -> None:
    manifest_path = WT / "APR_OPERATOR_SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tracked = {item["path"]: item for item in manifest.get("source_files", [])}
    paths = [
        "tools/apr/run_token2wav_flow_profile.py",
        "profiling/apr_operator_profile/__init__.py",
        "profiling/apr_operator_profile/analysis.py",
        "tools/apr/analyze_apr_operator_critical_path.py",
        "tests/test_apr_operator_critical_path_profile.py",
        "docs/superpowers/plans/2026-08-31-apr-aware-flow-operator-e2e.md",
    ]
    changed = []
    for relative in paths:
        path = WT / relative
        sha, size = digest(path)
        if relative in tracked:
            tracked[relative].update({"sha256": sha, "bytes": size})
        changed.append({"path": relative, "sha256": sha, "bytes": size})
    manifest["source_files"] = list(tracked.values())
    manifest["new_or_modified_files"] = changed
    manifest["worktree_state"] = {
        "head": "fef6f2be025b0a76338255d0f009a7fe8f978d03",
        "uncommitted_files_at_manifest_update": [item["path"] for item in changed],
        "generated_reports_are_evidence_artifacts": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(manifest_path, RESULTS / manifest_path.name)


if __name__ == "__main__":
    main()
