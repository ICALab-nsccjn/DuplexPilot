"""Fail-closed matrix launcher for the APR workload advantage suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from workloads.apr_benchmark import BASELINES, SUPPORTED_WORKLOADS


NOT_AVAILABLE_SYSTEM = "no_rsv_dsv_apr"
RUNNABLE_SYSTEMS = tuple(system for system in BASELINES if system != NOT_AVAILABLE_SYSTEM)
SUPPORTED_CONCURRENCIES = (4, 8, 16)
DEFAULT_REPEATS = 5
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_PINNED_PYTHON = (
    "/mnt/DuplexPilot/data/DuplexPilot/runtime/"
    "lychee_fd_pinned_conda_20260818/envs/fdmodel-demo/bin/python"
)


@dataclass(frozen=True)
class MatrixSpec:
    workload: str
    concurrency: int
    system: str
    repeat: int
    warmup: bool
    seed: int
    status: str = "RUN"


def build_matrix_specs(
    *,
    workloads: Sequence[str] = SUPPORTED_WORKLOADS,
    concurrencies: Sequence[int] = SUPPORTED_CONCURRENCIES,
    systems: Sequence[str] = BASELINES,
    repeats: int = DEFAULT_REPEATS,
) -> tuple[MatrixSpec, ...]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rows: list[MatrixSpec] = []
    for workload in workloads:
        if workload not in SUPPORTED_WORKLOADS:
            raise ValueError(f"unsupported workload: {workload}")
        for concurrency in concurrencies:
            if concurrency not in SUPPORTED_CONCURRENCIES:
                raise ValueError(f"unsupported concurrency: {concurrency}")
            for system in systems:
                if system not in BASELINES:
                    raise ValueError(f"unsupported system: {system}")
                if system == NOT_AVAILABLE_SYSTEM:
                    rows.append(
                        MatrixSpec(
                            workload=workload,
                            concurrency=int(concurrency),
                            system=system,
                            repeat=0,
                            warmup=False,
                            seed=0,
                            status="NOT_AVAILABLE",
                        )
                    )
                    continue
                rows.append(
                    MatrixSpec(
                        workload=workload,
                        concurrency=int(concurrency),
                        system=system,
                        repeat=1,
                        warmup=True,
                        seed=1001,
                    )
                )
                for repeat in range(1, repeats + 1):
                    rows.append(
                        MatrixSpec(
                            workload=workload,
                            concurrency=int(concurrency),
                            system=system,
                            repeat=repeat,
                            warmup=False,
                            seed=1000 + repeat,
                        )
                    )
    return tuple(rows)


def build_runner_command(
    *,
    python_executable: str,
    runner_path: str,
    out_root: str | Path,
    spec: MatrixSpec,
    arrival_process: str,
) -> tuple[str, ...]:
    if spec.status != "RUN":
        raise ValueError(f"cannot build a command for status={spec.status}")
    command = [
        python_executable,
        runner_path,
        "--system",
        spec.system,
        "--concurrency",
        str(spec.concurrency),
        "--repeat",
        str(spec.repeat),
        "--workload",
        spec.workload,
        "--seed",
        str(spec.seed),
        "--arrival-process",
        arrival_process,
        "--performance",
        "--out-root",
        str(out_root),
    ]
    if spec.warmup:
        command.append("--warmup")
    return tuple(command)


def _read_csv_run_ids(root: Path, filename: str) -> set[str]:
    import csv

    run_ids: set[str] = set()
    for path in sorted(root.rglob(filename)):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("run_id"):
                    run_ids.add(str(row["run_id"]))
    return run_ids


def _next_attempt_root(cell_root: Path) -> Path:
    indices = []
    for path in cell_root.glob("attempt-*"):
        try:
            indices.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return cell_root / f"attempt-{max(indices, default=0) + 1:02d}"


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _run_one(
    *,
    repo_root: Path,
    raw_root: Path,
    spec: MatrixSpec,
    python_executable: str,
    arrival_process: str,
    timeout_s: int,
) -> dict[str, object]:
    cell_root = raw_root / "runs" / spec.workload / f"N{spec.concurrency}" / spec.system
    cell_root.mkdir(parents=True, exist_ok=True)
    attempt_root = _next_attempt_root(cell_root)
    command = build_runner_command(
        python_executable=python_executable,
        runner_path=str(repo_root / "tools" / "apr" / "run_real_apr_e2e.py"),
        out_root=attempt_root,
        spec=spec,
        arrival_process=arrival_process,
    )
    env = os.environ.copy()
    current_max = int(env.get("APR_E2E_MAX_NUM_SEQS", "0"))
    env["APR_E2E_MAX_NUM_SEQS"] = str(max(current_max, spec.concurrency))
    env.setdefault("APR_E2E_WORKERS", "2")
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = str(exc.stdout or "")
        stderr = f"TIMEOUT after {timeout_s}s\n{exc.stderr or ''}"
    attempt_root.mkdir(parents=True, exist_ok=True)
    (attempt_root / "matrix_launcher_stdout.txt").write_text(stdout, encoding="utf-8")
    (attempt_root / "matrix_launcher_stderr.txt").write_text(stderr, encoding="utf-8")
    record = {
        "workload": spec.workload,
        "concurrency": spec.concurrency,
        "system": spec.system,
        "repeat": spec.repeat,
        "warmup": spec.warmup,
        "seed": spec.seed,
        "attempt_root": str(attempt_root),
        "command": list(command),
        "return_code": return_code,
        "elapsed_launcher_s": time.time() - started,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    _append_jsonl(raw_root / "matrix_attempts.jsonl", record)
    return record


def _write_suite_manifest(
    raw_root: Path,
    *,
    repo_root: Path,
    workloads: Sequence[str],
    concurrencies: Sequence[int],
    systems: Sequence[str],
    repeats: int,
    arrival_process: str,
) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    path = raw_root / "suite_manifest.json"
    manifest = {
        "suite": "APR_PUBLIC_WORKLOAD_SUITE_V1",
        "repo_root": str(repo_root),
        "workloads": list(workloads),
        "concurrencies": [int(value) for value in concurrencies],
        "systems": list(systems),
        "repeats": int(repeats),
        "warmup_per_cell": True,
        "arrival_process": arrival_process,
        "raw_artifacts_are_append_only": True,
        "no_rsv_dsv_apr_status": "NOT_AVAILABLE",
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"existing suite manifest differs: {path}")
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_matrix(
    *,
    repo_root: Path,
    raw_root: Path,
    workloads: Sequence[str] = SUPPORTED_WORKLOADS,
    concurrencies: Sequence[int] = SUPPORTED_CONCURRENCIES,
    systems: Sequence[str] = BASELINES,
    repeats: int = DEFAULT_REPEATS,
    arrival_process: str = "poisson",
    python_executable: str = DEFAULT_PINNED_PYTHON,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout_s: int = 1800,
) -> dict[str, object]:
    if arrival_process not in {"poisson", "burst"}:
        raise ValueError("arrival_process must be 'poisson' or 'burst'")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    _write_suite_manifest(
        raw_root,
        repo_root=repo_root,
        workloads=workloads,
        concurrencies=concurrencies,
        systems=systems,
        repeats=repeats,
        arrival_process=arrival_process,
    )
    specs = build_matrix_specs(
        workloads=workloads,
        concurrencies=concurrencies,
        systems=systems,
        repeats=repeats,
    )
    unavailable = 0
    completed = 0
    failed = 0
    for spec in specs:
        cell_root = raw_root / "runs" / spec.workload / f"N{spec.concurrency}" / spec.system
        if spec.status == "NOT_AVAILABLE":
            unavailable += 1
            _append_jsonl(
                raw_root / "unavailable_cells.jsonl",
                {**asdict(spec), "reason": "no real no-RSV/DSV model runner is available"},
            )
            print(
                json.dumps(
                    {
                        "event": "cell_unavailable",
                        "workload": spec.workload,
                        "concurrency": spec.concurrency,
                        "system": spec.system,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        run_id = f"apr-real-{spec.system}-n{spec.concurrency}-r{spec.repeat}-{'warmup' if spec.warmup else 'counted'}"
        filename = "warmup_attempts.csv" if spec.warmup else "formal_primary_metrics.csv"
        if run_id in _read_csv_run_ids(cell_root, filename):
            completed += 1
            continue
        launched = False
        success = False
        for _ in range(max_attempts):
            launched = True
            record = _run_one(
                repo_root=repo_root,
                raw_root=raw_root,
                spec=spec,
                python_executable=python_executable,
                arrival_process=arrival_process,
                timeout_s=timeout_s,
            )
            if int(record["return_code"]) == 0 and run_id in _read_csv_run_ids(cell_root, filename):
                success = True
                break
        if success:
            completed += 1
        elif launched:
            failed += 1
        print(
            json.dumps(
                {
                    "event": "matrix_spec_complete",
                    "workload": spec.workload,
                    "concurrency": spec.concurrency,
                    "system": spec.system,
                    "repeat": spec.repeat,
                    "warmup": spec.warmup,
                    "success": success,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "planned_specs": len(specs),
        "completed_specs": completed,
        "failed_specs": failed,
        "unavailable_cells": unavailable,
        "raw_root": str(raw_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-root", type=Path, default=Path("reports/apr_workload_suite/formal_matrix"))
    parser.add_argument("--python-executable", default=os.environ.get("APR_PINNED_PYTHON", DEFAULT_PINNED_PYTHON))
    parser.add_argument("--workloads", default=",".join(SUPPORTED_WORKLOADS))
    parser.add_argument("--concurrencies", default="4,8,16")
    parser.add_argument("--systems", default=",".join(BASELINES))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--arrival-process", choices=("poisson", "burst"), default="poisson")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args(argv)
    result = run_matrix(
        repo_root=args.repo_root.resolve(),
        raw_root=args.out_root.resolve(),
        workloads=tuple(value for value in args.workloads.split(",") if value),
        concurrencies=tuple(int(value) for value in args.concurrencies.split(",") if value),
        systems=tuple(value for value in args.systems.split(",") if value),
        repeats=args.repeats,
        arrival_process=args.arrival_process,
        python_executable=args.python_executable,
        max_attempts=args.max_attempts,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failed_specs"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
