from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


WT = Path("/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-aware-flow-operator-e2e")
RESULTS = Path("/mnt/DuplexPilot/data/DuplexPilot/results/apr_aware_flow_operator_e2e_20260831")
PROFILE = RESULTS / "phase1_operator_profile_gpu1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def copy_if_absent(source: Path, target: Path) -> None:
    if not target.exists():
        shutil.copy2(source, target)


def metrics() -> dict[str, float]:
    rows = list(csv.DictReader((PROFILE / "APR_OPERATOR_CRITICAL_PATH_METRICS.csv").open()))
    by_name = {row["category"]: row for row in rows}
    return {
        "all_flow_bound": 1.122249,
        "all_flow_share": 0.108932,
        "attention_flow_cuda_share": float(by_name["single_dit_family"]["flow_wall_share"]),
        "attention_bound": float(by_name["single_dit_family"]["ideal_bound"]),
        "attention_critical_share": float(by_name["single_dit_family"]["critical_path_share"]),
        "unattributed_bound": float(by_name["unattributed"]["ideal_bound"]),
        "online_flow_share": 0.214459,
        "critical_fraction": 0.507940,
        "historical_b2_max": 1.0427101281100264,
        "historical_b2_median": 1.0243674621593823,
    }


def main() -> None:
    m = metrics()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # Promote the authoritative GPU1 analysis artifacts to the run root, without
    # overwriting any pre-existing user artifact.
    for name in (
        "APR_OPERATOR_CRITICAL_PATH_PROFILE.md",
        "APR_OPERATOR_CRITICAL_PATH_METRICS.csv",
        "APR_OPERATOR_WATERFALL.csv",
        "APR_OPERATOR_AMDAHL_ANALYSIS.md",
        "APR_OPERATOR_AMDAHL_BLOCKED_REPORT.md",
    ):
        copy_if_absent(PROFILE / name, RESULTS / name)
        copy_if_absent(PROFILE / name, WT / name)

    placement = f"""# APR Operator Profile Placement Correction

## Status

The first P10/P50/P90 diagnostic attempts were launched before the diagnostic
runner explicitly selected a CUDA device. Their profiles report `cuda:0` and
are retained as historical diagnostic evidence only. They are **superseded**
and are not used in the operator decision.

The authoritative profiles were rerun inside `duplexpilot` with
`torch.cuda.set_device(1)` before Token2Wav construction and with
`--device-index 1`:

| envelope | profile directory | device evidence | PCM status |
| --- | --- | --- | --- |
| P10 | `{RESULTS}/profile_p10_gpu1` | `cuda:1` | non-empty |
| P50 | `{RESULTS}/profile_p50_gpu1_confirmation` | `cuda:1` | non-empty |
| P90 | `{RESULTS}/profile_p90_gpu1` | `cuda:1` | non-empty |

The P50 confirmation observed a GPU1 peak of approximately 6147 MiB
(about 6.003 GiB) and GPU0 remained at approximately 17 MiB during the
sampled Flow run. The profiles use float32 and ten Flow steps.

The corrected profiles are the only profiles passed to
`APR_OPERATOR_CRITICAL_PATH_PROFILE.md` and the Amdahl analysis. No model,
Flow numerical path, or online scheduler was changed to correct this
measurement issue.
"""
    write(RESULTS / "APR_OPERATOR_PROFILE_PLACEMENT_CORRECTION.md", placement)
    write(WT / "APR_OPERATOR_PROFILE_PLACEMENT_CORRECTION.md", placement)

    ledger = """# Flow Runtime No-Repeat Ledger

| Evidence | Status | Action |
|---|---|---|
| Phase 1-4 N=1/2/4/8/16 correctness | REUSED_VALIDATED_RESULT | Do not rerun full matrix |
| B=2/B=4 mixed-step variable-length chunk-padding | REUSED_VALIDATED_RESULT | Freeze at base tag |
| CUDA Graph fixed-work and online | REUSED_VALIDATED_RESULT | Do not combine in this branch |
| Inductor strict consistency | BLOCKED | Do not tune further |
| QKV projection operator | BLOCKED_WEAK | Do not repeat; exclude as target |
| Public datasets and external baseline audit | REUSED_VALIDATED_RESULT | Reuse fixed traces and roles |
| New profile and one operator candidate | COMPLETED_BLOCKED | Read-only profile/Amdahl analysis; no production operator implemented |
| New online E2E pilot | NOT_RUN_PREREQUISITE_BLOCKED | Amdahl gate stopped operator implementation; vLLM `_C` ABI also unavailable |
"""
    write(RESULTS / "APR_OPERATOR_NO_REPEAT_LEDGER.md", ledger)
    write(WT / "APR_OPERATOR_NO_REPEAT_LEDGER.md", ledger)

    fixed_blocked = f"""# APR Operator Fixed-Work Blocked Report

## Decision

`APR_OPERATOR_AMDAHL_BLOCKED`: the fixed-work implementation gate was not
entered because no pre-registered operator candidate had sufficient measured
end-to-end critical-path headroom.

The corrected GPU1 profile and preserved online traces give an optimistic
upper bound of approximately **{m['all_flow_bound']:.3f}x** even if all
currently observed Flow critical contribution were removed at zero cost.
This is below the plan's 1.15x threshold. The identifiable attention-family
candidate has an ideal bound of only **{m['attention_bound']:.3f}x** and does
not meet its 20% Flow-CUDA / 10% critical-path eligibility requirements.

The generic remainder is intentionally not promoted to a candidate: the
available profile cannot prove that its `addmm`, `cat`, `copy`, or clone calls
belong to one safe APR-aware boundary. The prior QKV fixed-work experiment is
reused as a negative result (about 1.026x median with strict logical-cache
equivalence failure), not repeated.

Therefore this branch did not implement an operator and did not run a new
operator fixed-work speedup experiment. This avoids reporting a local
microbenchmark as if it were an E2E result.

## Preserved evidence

- Frozen B>1 correctness and mixed-step/variable-length/chunk-padding
  mechanism evidence.
- Generic CUDA Graph evidence and its limited/interaction-blocked result.
- Inductor state/cache consistency block.
- QKV fusion negative result.
- Corrected P10/P50/P90 GPU1 profiles and their raw traces.

This is a bounded prerequisite failure, not a claim that every conceivable
future kernel implementation is impossible.
"""
    write(RESULTS / "APR_OPERATOR_FIXED_WORK_BLOCKED_REPORT.md", fixed_blocked)
    write(WT / "APR_OPERATOR_FIXED_WORK_BLOCKED_REPORT.md", fixed_blocked)

    online_blocked = f"""# APR Operator Online Pilot Blocked Report

## Decision

The planned online operator pilot is `NOT_RUN_PREREQUISITE_BLOCKED`.
The Amdahl gate stopped production operator implementation before any Flow
source or B>1 state semantics were changed. Consequently, this branch has no
new operator online throughput, latency, or SLO result to report.

The observed bound from real preserved online timing is approximately
**{m['all_flow_bound']:.3f}x** under the optimistic all-Flow-elimination
counterfactual. The prior B>1 critical-path evidence has a maximum idealized
bound of approximately **{m['historical_b2_max']:.3f}x** (median
approximately **{m['historical_b2_median']:.3f}x**), so controlled B>1 local
speedups cannot be substituted for online E2E evidence.

There is also an independent environment limitation: the approved container
reports PyTorch `2.9.1+cu128` / CUDA `12.8`, but importing the worktree vLLM
path fails with `ModuleNotFoundError: No module named 'vllm._C'`. No
incompatible Python environment was used to manufacture an online result.

## Interpretation

`NOT_RUN` is not `performance=0` and is not a failed correctness result. It
means the registered candidate was rejected by a stronger prerequisite before
online implementation. Any future online result must first close the vLLM
ABI/environment gate and use the same public harness and frozen B>1 control.
"""
    write(RESULTS / "APR_OPERATOR_ONLINE_PILOT_BLOCKED_REPORT.md", online_blocked)
    write(WT / "APR_OPERATOR_ONLINE_PILOT_BLOCKED_REPORT.md", online_blocked)

    verdict = f"""# APR Operator Final Verdict

## Classification

**APR_OPERATOR_BLOCKED** (bounded Amdahl prerequisite failure)

## Evidence summary

| item | result |
| --- | --- |
| frozen B>1 mechanism | retained; no semantic changes in this branch |
| corrected GPU1 P10/P50/P90 profile | valid; all report `cuda:1` and non-empty PCM |
| observed median online Flow/E2E share | approximately {m['online_flow_share']:.4f} |
| observed median Flow critical fraction | approximately {m['critical_fraction']:.4f} |
| optimistic all-Flow ideal E2E bound | approximately {m['all_flow_bound']:.3f}x |
| registered attention-family ideal bound | approximately {m['attention_bound']:.3f}x |
| state/cache and dispatch candidate attribution | no explicit auditable boundary observed |
| new production operator implementation | not performed |
| new operator online pilot | not run; prerequisite blocked |
| focused regression | 56 passed |

The optimistic upper bound is already below the 1.15x paper gate, and the
only identifiable registered family is much smaller. Implementing an operator
after this gate would not be an evidence-based route to the required online
claim. The branch therefore stops before modifying Flow execution.

## What remains publishable evidence

APR still has a frozen, auditable B>1 execution mechanism with state/identity,
mixed-step and cancellation/flush evidence. Those results support a
mechanism/capability statement and a workload-dependent limitation. They do
not support a claim of universal online throughput gain. Generic Graph,
Inductor, and QKV results remain separately attributed and are not combined
with the B>1 claim.

## Required boundary for any future work

Any new performance direction must be separately approved and separately
versioned. It must first identify enough measured E2E critical-path headroom,
then pass numerical/state equivalence, memory, and the same online public
harness. This branch does not silently reopen B, N, padding, wait windows,
Graph, Inductor, or scheduler variants.
"""
    write(RESULTS / "APR_OPERATOR_FINAL_VERDICT.md", verdict)
    write(WT / "APR_OPERATOR_FINAL_VERDICT.md", verdict)

    boundary = """# APR Operator Claim Boundary

## Claims supported by this execution

- APR exposes an explicit acoustic state that supports the frozen B>1
  mechanism under the audited compatibility and identity contracts.
- The preserved measurements show that B>1 execution opportunity and its
  local speedup are workload-dependent and do not automatically become an
  end-to-end speedup.
- Under the registered candidates and observed traces, no auditable operator
  target passed the Amdahl gate; stopping before implementation is the
  reproducible result of this bounded study.

## Claims not supported

- Do not claim a new APR-specific operator speedup.
- Do not claim that eliminating the profiled Flow region produces a measured
  1.122x online speedup; 1.122x is an optimistic upper bound, not an
  experiment.
- Do not claim that controlled B=2/B=4 acceleration equals online E2E
  acceleration.
- Do not combine Graph, QKV, or generic runtime gains with B>1 as if they were
  one causal intervention.
- Do not present the uncorrected GPU0 diagnostic profiles as GPU1 evidence.
- Do not report the unrun online operator pilot as a zero-performance result.
"""
    write(RESULTS / "APR_OPERATOR_CLAIM_BOUNDARY.md", boundary)
    write(WT / "APR_OPERATOR_CLAIM_BOUNDARY.md", boundary)

    exit_decision = """# APR Flow Runtime Exit Decision

## Current decision

`STOP_REGISTERED_OPERATOR_ITERATION`

The APR-aware operator branch is closed by its pre-registered Amdahl gate.
No operator was implemented because the corrected real-trace analysis could
not identify a candidate with at least 1.15x ideal E2E headroom. This prevents
an ungrounded sequence of increasingly specialized kernels.

## Frozen assets

- `safe-variable-length-flow-b2-mixed-step-v1` remains the B>1 mechanism
  reference.
- Generic Graph, Inductor, QKV, capacity/SLO, preemption, and staging results
  remain immutable historical evidence with their original claim boundaries.
- The corrected GPU1 profiles and all analysis reports in this run are the
  authoritative artifacts for this branch.

## Not authorized by this result

This decision does not authorize more B/N/padding/wait-window variants,
operator stacking, Graph/Inductor composition, model approximation, or a
silent switch of baseline/environment. A new direction requires a new plan,
an independent branch, and a fresh E2E headroom argument.
"""
    write(RESULTS / "APR_FLOW_RUNTIME_EXIT_DECISION.md", exit_decision)
    write(WT / "APR_FLOW_RUNTIME_EXIT_DECISION.md", exit_decision)

    # Update generated manifests with the evidence status, retaining all
    # original source hashes and fields.
    source_manifest_path = WT / "APR_OPERATOR_SOURCE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.update(
        {
            "updated_utc": now,
            "worktree_commit": "fef6f2be025b0a76338255d0f009a7fe8f978d03",
            "phase1_decision": "APR_OPERATOR_AMDAHL_BLOCKED",
            "fixed_work_status": "NOT_RUN_PREREQUISITE_BLOCKED",
            "online_pilot_status": "NOT_RUN_PREREQUISITE_BLOCKED",
            "new_production_operator_implemented": False,
            "profile_placement_correction": {
                "invalid_profiles": [
                    "profile_p10_baseline",
                    "profile_p50_baseline",
                    "profile_p90_baseline",
                ],
                "invalid_reason": "diagnostic runner did not set current CUDA device; profiles attributed to cuda:0",
                "authoritative_profiles": [
                    str(RESULTS / "profile_p10_gpu1"),
                    str(RESULTS / "profile_p50_gpu1_confirmation"),
                    str(RESULTS / "profile_p90_gpu1"),
                ],
            },
            "vllm_environment_audit": {
                "status": "BLOCKED",
                "error": "ModuleNotFoundError: No module named 'vllm._C'",
                "torch": "2.9.1+cu128",
                "cuda": "12.8",
                "policy": "do not substitute an incompatible environment",
            },
            "analysis_artifacts": {
                "directory": str(PROFILE),
                "sha256": {
                    name: sha256(PROFILE / name)
                    for name in (
                        "APR_OPERATOR_CRITICAL_PATH_PROFILE.md",
                        "APR_OPERATOR_CRITICAL_PATH_METRICS.csv",
                        "APR_OPERATOR_WATERFALL.csv",
                        "APR_OPERATOR_AMDAHL_ANALYSIS.md",
                        "APR_OPERATOR_AMDAHL_BLOCKED_REPORT.md",
                    )
                },
            },
        }
    )
    source_manifest["commands"] = [
        "container focused pytest (56 passed)",
        "container GPU1 P10/P50/P90 diagnostic Flow profiles",
        "read-only critical-path/Amdahl analysis",
        "operator fixed-work and online pilot not run after prerequisite block",
    ]
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(source_manifest_path, RESULTS / source_manifest_path.name)

    env_path = WT / "APR_OPERATOR_ENVIRONMENT_MANIFEST.json"
    env_manifest = json.loads(env_path.read_text(encoding="utf-8"))
    env_manifest.update(
        {
            "updated_utc": now,
            "docker_digest_status": "not exposed by local image inspect",
            "vllm_import_status": "BLOCKED: missing vllm._C for torch 2.9.1+cu128",
            "profile_device_correction": "authoritative profiles use cuda device index 1",
            "online_status": "ONLINE_ENVIRONMENT_BLOCKED; no operator E2E claim",
        }
    )
    env_manifest["commands"] = [
        "container df/nvidia-smi/Python/PyTorch/CUDA audit",
        "container focused pytest: 56 passed",
        "container GPU1 Flow profile with --device-index 1",
        "online public operator pilot not run: Amdahl prerequisite and vLLM ABI blocked",
    ]
    env_path.write_text(json.dumps(env_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(env_path, RESULTS / env_path.name)

    # Keep the authoritative reports easily discoverable from both locations.
    summary = {
        "schema_version": 1,
        "generated_utc": now,
        "decision": "APR_OPERATOR_AMDAHL_BLOCKED",
        "operator_implementation": "NOT_RUN_PREREQUISITE_BLOCKED",
        "online_pilot": "NOT_RUN_PREREQUISITE_BLOCKED",
        "optimistic_all_flow_ideal_bound": m["all_flow_bound"],
        "registered_attention_ideal_bound": m["attention_bound"],
        "historical_b2_critical_path_ideal_bound_max": m["historical_b2_max"],
        "focused_tests_passed": 56,
    }
    write(RESULTS / "APR_OPERATOR_EXECUTION_SUMMARY.json", json.dumps(summary, indent=2, ensure_ascii=False))
    write(WT / "APR_OPERATOR_EXECUTION_SUMMARY.json", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
