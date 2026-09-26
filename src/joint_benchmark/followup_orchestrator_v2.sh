#!/usr/bin/env bash
# Continue the registered joint experiment after the HD matrix.
# This version avoids grep-based self-matching and never deletes an attempt.
set -u

ROOT="/mnt/DuplexPilot/data/DuplexPilot/results/apr_joint_b22_multidataset_e2e_20260831"
WORKTREE="/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-joint-b22-multidataset-e2e"
HD_PID="${1:?HD matrix PID is required}"
LOG="$ROOT/launcher_logs/followup_orchestrator_v2.log"

echo "WAIT_HD pid=$HD_PID $(date -Is)" >> "$LOG"
while kill -0 "$HD_PID" >/dev/null 2>&1; do
  sleep 30
done
echo "HD_DONE $(date -Is)" >> "$LOG"

FIXED="$ROOT/APR_JOINT_B22_ACOUSTIC_FIXED_WORK.json"
if [[ ! -s "$FIXED" ]]; then
  echo "START_ACOUSTIC_FIXED_WORK $(date -Is)" >> "$LOG"
  set +e
  docker exec \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e LYCHEEFD_TOKEN2WAV_DEVICE=1 \
    -e LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY=2048 \
    -e PYTHONPATH="$WORKTREE:$WORKTREE/third_party/vllm:$WORKTREE/third_party/Step-Audio2" \
    -e LD_LIBRARY_PATH=/root/anaconda3/envs/sglang/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64 \
    -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    -w "$WORKTREE" duplexpilot \
    /root/anaconda3/envs/sglang/bin/python tools/benchmarks/run_joint_acoustic_fixed_work.py \
    --out "$FIXED" --warmups 5 --repeats 20 \
    > "$ROOT/launcher_logs/acoustic_fixed_work_v2.log" 2>&1
  acoustic_rc=$?
  set -e
  echo "END_ACOUSTIC_FIXED_WORK rc=$acoustic_rc $(date -Is)" >> "$LOG"
else
  echo "ACOUSTIC_FIXED_WORK_ALREADY_PRESENT $(date -Is)" >> "$LOG"
fi

if [[ ! -s "$FIXED" ]]; then
  echo "ACOUSTIC_FIXED_WORK_MISSING_STOP $(date -Is)" >> "$LOG"
  exit 1
fi

echo "START_FD $(date -Is)" >> "$LOG"
bash "$WORKTREE/tools/benchmarks/run_joint_b22_matrix.sh" fd "$ROOT" \
  > "$ROOT/launcher_logs/fd_matrix_v2.stdout.log" 2>&1
echo "END_FD rc=$? $(date -Is)" >> "$LOG"

echo "START_FDB $(date -Is)" >> "$LOG"
bash "$WORKTREE/tools/benchmarks/run_joint_b22_matrix.sh" fdb "$ROOT" \
  > "$ROOT/launcher_logs/fdb_matrix_v2.stdout.log" 2>&1
echo "END_FDB rc=$? $(date -Is)" >> "$LOG"
