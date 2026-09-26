#!/usr/bin/env bash
# Start the finite held-out matrix after all discovery pilot suites finish.
# Accept the current v3 continuation marker, while retaining compatibility
# with an already-completed v2 run.
set -u

ROOT="/mnt/DuplexPilot/data/DuplexPilot/results/apr_joint_b22_multidataset_e2e_20260831"
WORKTREE="/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-joint-b22-multidataset-e2e"
LOG="$ROOT/launcher_logs/heldout_orchestrator_v4.log"

echo "WAIT_DISCOVERY $(date -Is)" >> "$LOG"
while true; do
  if [[ -s "$ROOT/launcher_logs/fdb_matrix_v3.stdout.log" ]] \
      && grep -q "MATRIX_COMPLETE suite=fdb" "$ROOT/launcher_logs/fdb_matrix_v3.stdout.log"; then
    break
  fi
  if [[ -s "$ROOT/launcher_logs/fdb_matrix_v2.stdout.log" ]] \
      && grep -q "MATRIX_COMPLETE suite=fdb" "$ROOT/launcher_logs/fdb_matrix_v2.stdout.log"; then
    break
  fi
  sleep 60
done
echo "DISCOVERY_DONE $(date -Is)" >> "$LOG"

set +e
bash "$WORKTREE/tools/benchmarks/run_joint_b22_heldout_matrix.sh" "$ROOT" \
  > "$ROOT/launcher_logs/heldout_matrix_v4.stdout.log" 2>&1
rc=$?
set -e
echo "HELDOUT_DONE rc=$rc $(date -Is)" >> "$LOG"
exit "$rc"
