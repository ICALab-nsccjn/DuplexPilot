#!/usr/bin/env bash
# Finite held-out matrix for the joint B_model/B_acoustic study.
# Every case has its own directory; failed attempts are retained.
set -u

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RESULT_ROOT" >&2
  exit 2
fi

RESULT_ROOT="${1%/}"
WORKTREE="/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-joint-b22-multidataset-e2e"
RUNNER="$WORKTREE/tools/benchmarks/run_joint_b22_case.sh"
mkdir -p "$RESULT_ROOT/launcher_logs"

run_one() {
  local n="$1" workload="$2" trace="$3" repeat="$4" mode="$5" system="$6" label="$7"
  local out="$RESULT_ROOT/${label}_${workload}_N${n}_${mode}_r${repeat}"
  local log="$RESULT_ROOT/launcher_logs/${label}_${workload}_N${n}_${mode}_r${repeat}.log"
  if [[ -e "$out" ]]; then
    echo "SKIP existing $out" | tee -a "$RESULT_ROOT/launcher_logs/heldout_matrix.log"
    return 0
  fi
  echo "START $label workload=$workload N=$n mode=$mode repeat=$repeat" | tee -a "$RESULT_ROOT/launcher_logs/heldout_matrix.log"
  set +e
  bash "$RUNNER" "$n" "$workload" "${system}" "$repeat" "$trace" "$out" "$mode" > "$log" 2>&1
  local rc=$?
  set -e
  if [[ ! -f "$out/run_metadata.json" ]]; then
    mkdir -p "$out"
    printf '{"status":"FAILED","runner_rc":%s,"N":%s,"workload":"%s","system":"%s","joint_mode":"%s","repeat":%s,"trace":"%s","failure_reason":"runner_or_validation_failure"}\n' \
      "$rc" "$n" "$workload" "$system" "$mode" "$repeat" "$trace" \
      > "$out/attempt_status.json"
  fi
  echo "END rc=$rc $out" | tee -a "$RESULT_ROOT/launcher_logs/heldout_matrix.log"
  return 0
}

run_suite() {
  local suite="$1"
  shift
  local n workload trace
  local modes=(J11 J21 J12 J22)
  local systems=(
    rsv_dsv_apr_joint_j11
    rsv_dsv_apr_joint_j21
    rsv_dsv_apr_joint_j12
    rsv_dsv_apr_joint_j22
  )
  for n in "$@"; do
    for workload in "${WORKLOADS[@]}"; do
      trace="$(trace_for "$workload" "$n")"
      if [[ ! -f "$trace" ]]; then
        echo "MISSING_TRACE $trace" | tee -a "$RESULT_ROOT/launcher_logs/heldout_matrix.log"
        continue
      fi
      for index in "${!modes[@]}"; do
        local mode="${modes[$index]}"
        local system="${systems[$index]}"
        run_one "$n" "$workload" "$trace" 0 "$mode" "$system" heldout
        for repeat in $(seq 1 5); do
          run_one "$n" "$workload" "$trace" "$repeat" "$mode" "$system" heldout
        done
      done
    done
  done
  echo "MATRIX_COMPLETE suite=$suite root=$RESULT_ROOT" | tee -a "$RESULT_ROOT/launcher_logs/heldout_matrix.log"
}

# HD held-out uses non-overlapping window-1 traces.
WORKLOADS=(HD-LongTail HD-Burst)
trace_for() {
  printf '/mnt/DuplexPilot/data/DuplexPilot/results/online_n_batch_formation_closure/traces_v2/%s/N%s/window-1.jsonl' "$1" "$2"
}
run_suite hd-heldout 8 16

# FD-Bench held-out has a registered N=8 clip set.
WORKLOADS=(FD-Balanced FD-Burst)
trace_for() {
  # The canonical public filename retains the FD- workload prefix.
  printf '/mnt/DuplexPilot/data/DuplexPilot/results/published_baseline_multidataset_20260829/traces_v4/FD-Bench-Audio-Input/FD-Bench-Audio-Input-%s-heldout-N%s-w0.public.jsonl' "$1" "$2"
}
run_suite fd-heldout 8

echo "HELDOUT_COMPLETE root=$RESULT_ROOT" | tee -a "$RESULT_ROOT/launcher_logs/heldout_matrix.log"
