#!/usr/bin/env bash
set -u

ROOT=/mnt/DuplexPilot/data/DuplexPilot/results/apr_joint_b22_multidataset_e2e_20260831
HD_PID=1708638
LOG="$ROOT/launcher_logs/followup_orchestrator.log"

while ps -p "$HD_PID" >/dev/null 2>&1; do
  sleep 30
done

while ps -eo args | grep -v grep | grep -q 'run_joint_acoustic_fixed_work.py'; do
  sleep 30
done

if [[ ! -s "$ROOT/APR_JOINT_B22_ACOUSTIC_FIXED_WORK.json" ]]; then
  echo "ACOUSTIC_FIXED_WORK_MISSING" >> "$LOG"
  exit 1
fi

echo "START_FD $(date -Is)" >> "$LOG"
bash /mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-joint-b22-multidataset-e2e/tools/benchmarks/run_joint_b22_matrix.sh fd "$ROOT" \
  > "$ROOT/launcher_logs/fd_matrix_v2.stdout.log" 2>&1
echo "END_FD rc=$? $(date -Is)" >> "$LOG"

echo "START_FDB $(date -Is)" >> "$LOG"
bash /mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-joint-b22-multidataset-e2e/tools/benchmarks/run_joint_b22_matrix.sh fdb "$ROOT" \
  > "$ROOT/launcher_logs/fdb_matrix_v2.stdout.log" 2>&1
echo "END_FDB rc=$? $(date -Is)" >> "$LOG"
