#!/usr/bin/env bash
# Start the finite held-out matrix after all discovery pilot suites finish.
set -u
ROOT="/mnt/DuplexPilot/data/DuplexPilot/results/apr_joint_b22_multidataset_e2e_20260831"
WORKTREE="/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-joint-b22-multidataset-e2e"
LOG="$ROOT/launcher_logs/heldout_orchestrator.log"

echo "WAIT_DISCOVERY $(date -Is)" >> "$LOG"
while [[ ! -s "$ROOT/launcher_logs/fdb_matrix_v2.stdout.log" ]] || ! grep -q "MATRIX_COMPLETE suite=fdb" "$ROOT/launcher_logs/fdb_matrix_v2.stdout.log"; do
  sleep 60
done
echo "DISCOVERY_DONE $(date -Is)" >> "$LOG"

bash "$WORKTREE/tools/benchmarks/run_joint_b22_heldout_matrix.sh" "$ROOT" \
  > "$ROOT/launcher_logs/heldout_matrix.stdout.log" 2>&1
echo "HELDOUT_DONE rc=$? $(date -Is)" >> "$LOG"
