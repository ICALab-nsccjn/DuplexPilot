#!/usr/bin/env bash
# Run the registered joint pilot sequentially on one dual-GPU service.
# This script never overwrites a case directory and keeps failed attempts.
set -u

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 SUITE [RESULT_ROOT]" >&2
  echo "SUITE: hd | fd | fdb" >&2
  exit 2
fi

SUITE="$1"
RESULT_ROOT="${2:-/mnt/DuplexPilot/data/DuplexPilot/results/apr_joint_b22_multidataset_e2e_20260831}"
WORKTREE="/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-joint-b22-multidataset-e2e"
RUNNER="$WORKTREE/tools/benchmarks/run_joint_b22_case.sh"
mkdir -p "$RESULT_ROOT/launcher_logs"

case "$SUITE" in
  hd)
    N_VALUES=(8 16)
    WORKLOADS=(HD-Balanced HD-LongTail HD-Burst)
    trace_for() {
      printf '/mnt/DuplexPilot/data/DuplexPilot/results/online_n_batch_formation_closure/traces_v2/%s/N%s/window-0.jsonl' "$1" "$2"
    }
    ;;
  fd)
    N_VALUES=(4 8)
    WORKLOADS=(FD-Balanced FD-Burst)
    trace_for() {
      local n="$2" workload="$1"
      # The canonical public filename retains the FD- workload prefix.
      # Keep the workload label identical to the matrix label so the trace
      # lookup cannot silently skip the entire FD-Bench suite.
      printf '/mnt/DuplexPilot/data/DuplexPilot/results/published_baseline_multidataset_20260829/traces_v4/FD-Bench-Audio-Input/FD-Bench-Audio-Input-%s-discovery-N%s-w0.public.jsonl' "$workload" "$n"
    }
    ;;
  fdb)
    N_VALUES=(4 8)
    WORKLOADS=(FDB-v1.5-Examples)
    trace_for() {
      local n="$2"
      printf '/mnt/DuplexPilot/data/DuplexPilot/results/published_baseline_multidataset_20260829/traces_v4/Full-Duplex-Bench-v1.5/Full-Duplex-Bench-v1.5-FDB-v1.5-Examples-limited_scope_examples-N%s-w0.public.jsonl' "$n"
    }
    ;;
  *)
    echo "unsupported suite: $SUITE" >&2
    exit 2
    ;;
esac

MODES=(J11 J21 J12 J22)
SYSTEMS=(
  rsv_dsv_apr_joint_j11
  rsv_dsv_apr_joint_j21
  rsv_dsv_apr_joint_j12
  rsv_dsv_apr_joint_j22
)

# Full-Duplex-Bench v1.5 has a smaller registered example set.  Keep its
# prescribed 1-warmup + 2-counted protocol while retaining 3 counted runs
# for the HD and FD smoke suites.
COUNTED_REPEATS=3
if [[ "$SUITE" == "fdb" ]]; then
  COUNTED_REPEATS=2
fi

run_one() {
  local n="$1" workload="$2" mode="$3" system="$4" repeat="$5" trace="$6" label="$7"
  local out="$RESULT_ROOT/${label}_${workload}_N${n}_${mode}_r${repeat}"
  local log="$RESULT_ROOT/launcher_logs/${label}_${workload}_N${n}_${mode}_r${repeat}.log"
  if [[ -e "$out" ]]; then
    echo "SKIP existing $out" | tee -a "$RESULT_ROOT/launcher_logs/matrix.log"
    return 0
  fi
  echo "START $label workload=$workload N=$n mode=$mode repeat=$repeat" | tee -a "$RESULT_ROOT/launcher_logs/matrix.log"
  set +e
  bash "$RUNNER" "$n" "$workload" "$system" "$repeat" "$trace" "$out" "$mode" > "$log" 2>&1
  local rc=$?
  set -e
  # Preserve an explicit invalid-attempt record even when the case runner
  # exits before it can write run_metadata.json (for example, a launch or
  # shell-syntax failure).  Downstream analysis includes this row instead of
  # silently dropping it from the denominator.
  if [[ ! -f "$out/run_metadata.json" ]]; then
    mkdir -p "$out"
    printf '{"status":"FAILED","runner_rc":%s,"N":%s,"workload":"%s","system":"%s","joint_mode":"%s","repeat":%s,"trace":"%s","failure_reason":"runner_or_validation_failure"}\n' \
      "$rc" "$n" "$workload" "$system" "$mode" "$repeat" "$trace" \
      > "$out/attempt_status.json"
  fi
  echo "END rc=$rc $out" | tee -a "$RESULT_ROOT/launcher_logs/matrix.log"
  return 0
}

for n in "${N_VALUES[@]}"; do
  for workload in "${WORKLOADS[@]}"; do
    trace="$(trace_for "$workload" "$n")"
    if [[ ! -f "$trace" ]]; then
      echo "MISSING_TRACE $trace" | tee -a "$RESULT_ROOT/launcher_logs/matrix.log"
      continue
    fi
    for index in "${!MODES[@]}"; do
      mode="${MODES[$index]}"
      system="${SYSTEMS[$index]}"
      # A separate warmup is retained but excluded by downstream aggregation.
      run_one "$n" "$workload" "$mode" "$system" 0 "$trace" warmup
      for repeat in $(seq 1 "$COUNTED_REPEATS"); do
        run_one "$n" "$workload" "$mode" "$system" "$repeat" "$trace" counted
      done
    done
  done
done

echo "MATRIX_COMPLETE suite=$SUITE root=$RESULT_ROOT"
