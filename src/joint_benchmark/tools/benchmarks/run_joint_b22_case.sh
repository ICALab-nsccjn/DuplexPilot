#!/usr/bin/env bash
set -u

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 N WORKLOAD SYSTEM REPEAT TRACE RESULT_DIR MODE [--no-sleep]" >&2
  exit 2
fi

N="$1"
WORKLOAD="$2"
SYSTEM="$3"
REPEAT="$4"
TRACE="$5"
RESULT_DIR="${6%/}"
MODE="$7"
NO_SLEEP="${8:-}"
WORKTREE="/mnt/DuplexPilot/data/DuplexPilot/worktrees/apr-joint-b22-multidataset-e2e"
MODEL_PATH="/mnt/DuplexPilot/data/models/lychee_full_duplex"
TOKEN2WAV_PATH="/mnt/DuplexPilot/data/models/token2wav"

case "$N" in
  1|2|4|8|16) ;;
  *) echo "unsupported N: $N" >&2; exit 2 ;;
esac
case "$SYSTEM" in
  rsv_dsv_apr_joint_j11|rsv_dsv_apr_joint_j21|rsv_dsv_apr_joint_j12|rsv_dsv_apr_joint_j22) ;;
  *) echo "unsupported joint system: $SYSTEM" >&2; exit 2 ;;
esac
case "$MODE" in
  J11|J12|J21|J22) ;;
  *) echo "unsupported joint mode: $MODE" >&2; exit 2 ;;
esac
[[ -f "$TRACE" ]] || { echo "missing trace: $TRACE" >&2; exit 2; }
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing to overwrite: $RESULT_DIR" >&2; exit 2; }

if [[ "$MODE" == J21 || "$MODE" == J22 ]]; then
  ROW_ENV=1
  ROW_CAP=2
else
  ROW_ENV=0
  ROW_CAP=1
fi
if [[ "$MODE" == J12 || "$MODE" == J22 ]]; then
  ACOUSTIC_CAP=2
else
  ACOUSTIC_CAP=1
fi

mkdir -p "$RESULT_DIR"
df -h /mnt/DuplexPilot > "$RESULT_DIR/host_disk_before.txt"
SERVER_PATTERN='[r]un_public_realtime_server.py.*--port 18080'
if docker exec duplexpilot pgrep -f "$SERVER_PATTERN" >/dev/null 2>&1; then
  echo "port 18080 is already occupied" >&2
  exit 2
fi

stop_server() {
  local pids
  pids="$(docker exec duplexpilot pgrep -f "$SERVER_PATTERN" || true)"
  if [[ -n "$pids" ]]; then
    docker exec duplexpilot kill $pids || true
    for _ in $(seq 1 90); do
      docker exec duplexpilot pgrep -f "$SERVER_PATTERN" >/dev/null 2>&1 || return 0
      sleep 1
    done
    echo "benchmark server did not stop" >&2
    return 1
  fi
  return 0
}
trap stop_server EXIT

TRACE_PATH="$RESULT_DIR/model_execution_trace.jsonl"
TOKEN_TRACE_PATH="$RESULT_DIR/token_trace.jsonl"
ONLINE_TRACE_PATH="$RESULT_DIR/online_trace.jsonl"
OPPORTUNITY_TRACE_PATH="$RESULT_DIR/opportunity_trace.jsonl"
WATERFALL_TRACE_PATH="$RESULT_DIR/waterfall_trace.jsonl"
GPU_MONITOR_PATH="$RESULT_DIR/gpu_monitor.csv"

stop_gpu_monitor() {
  # The command is detached inside the container, so killing only the local
  # docker-exec wrapper leaves nvidia-smi behind.  Match the exact monitor
  # argv and terminate only this benchmark's sampler.
  for _ in 1 2 3; do
    docker exec duplexpilot bash -lc '
      for pid in $(pgrep -x nvidia-smi || true); do
        args=$(ps -p "$pid" -o args= || true)
        if [[ "$args" == "nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -lms 200" ]]; then
          kill "$pid" || true
        fi
      done
    ' >/dev/null 2>&1 || true
    sleep 0.2
  done
}

cleanup() {
  stop_gpu_monitor
  stop_server
}
trap cleanup EXIT

docker exec -d \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e LYCHEEFD_TOKEN2WAV_DEVICE=1 \
  -e LYCHEEFD_REQUIRE_DUAL_GPU_PLACEMENT=1 \
  -e LYCHEEFD_USE_VLLM=1 \
  -e LYCHEEFD_FLOW_ATTENTION_CACHE_CAPACITY=2048 \
  -e LYCHEEFD_MAX_FLOW_BATCH_SIZE="$ACOUSTIC_CAP" \
  -e LYCHEEFD_VLLM_MAX_NUM_SEQS="$ROW_CAP" \
  -e LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION=0.70 \
  -e LYCHEEFD_VLLM_ENABLE_CHUNKED_PREFILL=1 \
  -e LYCHEEFD_VLLM_ENABLE_PREFIX_CACHING=0 \
  -e LYCHEEFD_VLLM_MAX_MODEL_LEN=8192 \
  -e LYCHEEFD_VLLM_MAX_NUM_BATCHED_TOKENS=1024 \
  -e LYCHEEFD_ROW_AWARE_MODEL_EXECUTION_PLANE="$ROW_ENV" \
  -e LYCHEEFD_ROW_AWARE_MODEL_MAX_BATCH_SIZE="$ROW_CAP" \
  -e DUPLEXPILOT_APR_LOGICAL_N="$N" \
  -e DUPLEXPILOT_APR_ONLINE_TRACE_PATH="$ONLINE_TRACE_PATH" \
  -e DUPLEXPILOT_APR_OPPORTUNITY_TRACE_PATH="$OPPORTUNITY_TRACE_PATH" \
  -e DUPLEXPILOT_APR_WATERFALL_TRACE_PATH="$WATERFALL_TRACE_PATH" \
  -e LYCHEEFD_MODEL_EXECUTION_TRACE_PATH="$TRACE_PATH" \
  -e LYCHEEFD_VLLM_TOKEN_TRACE=1 \
  -e LYCHEEFD_VLLM_TOKEN_TRACE_PATH="$TOKEN_TRACE_PATH" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONPATH="$WORKTREE:$WORKTREE/third_party/vllm:$WORKTREE/third_party/Step-Audio2" \
  -e LD_LIBRARY_PATH=/root/anaconda3/envs/sglang/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64 \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -w "$WORKTREE" duplexpilot bash -lc \
  "exec /root/anaconda3/envs/sglang/bin/python -u tools/benchmarks/run_public_realtime_server.py --model-path '$MODEL_PATH' --token2wav-path '$TOKEN2WAV_PATH' --host 0.0.0.0 --port 18080 > '$RESULT_DIR/server.log' 2>&1"

# Metadata-only monitor: sample both A100s during the attempt.  It is
# detached inside the container and explicitly reaped by stop_gpu_monitor;
# it does not alter server/client scheduling semantics.
stop_gpu_monitor
docker exec -d duplexpilot bash -lc \
  "exec nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -lms 200 > '$GPU_MONITOR_PATH' 2>&1"

ready=0
for _ in $(seq 1 360); do
  if docker exec duplexpilot /root/anaconda3/envs/sglang/bin/python -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:18080/openapi.json", timeout=1).read()' \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "server failed to become ready" >&2
  tail -120 "$RESULT_DIR/server.log" >&2 || true
  exit 1
fi

sleep_flag=""
if [[ "$NO_SLEEP" == "--no-sleep" ]]; then
  sleep_flag="--no-sleep"
fi
set +e
docker exec -w "$WORKTREE" duplexpilot bash -lc \
  "exec /root/anaconda3/envs/sglang/bin/python -m tools.benchmarks.apr_public_online_e2e --base-url http://127.0.0.1:18080 --trace '$TRACE' --system '$SYSTEM' --out-root '$RESULT_DIR/client' --repeat '$REPEAT' --timeout-s 360 $sleep_flag" \
  > "$RESULT_DIR/client.log" 2>&1
client_rc=$?
set -e

stop_gpu_monitor

docker exec duplexpilot nvidia-smi \
  --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader > "$RESULT_DIR/gpu_after.csv" || true

docker exec duplexpilot /root/anaconda3/envs/sglang/bin/python -c \
  'import json,sys,time; out,n,workload,system,repeat,trace,online,model_trace,token_trace,mode,row_env,row_cap,acoustic_cap=sys.argv[1:]; payload={"N":int(n),"workload":workload,"system":system,"repeat":int(repeat),"trace":trace,"online_trace":online,"model_execution_trace":model_trace,"token_trace":token_trace,"joint_mode":mode,"row_aware_model_execution":bool(int(row_env)),"max_model_batch_size":int(row_cap),"vllm_max_num_seqs":int(row_cap),"max_acoustic_batch_size":int(acoustic_cap),"hardware":"2x NVIDIA A100-SXM4-40GB; GPU0=model; GPU1=Token2Wav/Flow","container":"duplexpilot","flow_cache_capacity":2048,"flow_steps":10,"physical_acoustic_workers":2,"cuda_visible_devices":"0,1","created_epoch_s":time.time()}; open(out,"w",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")' \
  "$RESULT_DIR/run_metadata.json" "$N" "$WORKLOAD" "$SYSTEM" "$REPEAT" "$TRACE" \
  "$ONLINE_TRACE_PATH" "$TRACE_PATH" "$TOKEN_TRACE_PATH" "$MODE" "$ROW_ENV" "$ROW_CAP" "$ACOUSTIC_CAP"

set +e
docker exec -w "$WORKTREE" duplexpilot \
  /root/anaconda3/envs/sglang/bin/python -m tools.benchmarks.validate_online_n_batch_case \
  --result "$RESULT_DIR/client/online_attempt.json" \
  --server-log "$RESULT_DIR/server.log" \
  --expected-n "$N" > "$RESULT_DIR/strict_summary.json"
strict_rc=$?
set -e

if [[ "$client_rc" -ne 0 || "$strict_rc" -ne 0 ]]; then
  echo "case failed: client_rc=$client_rc strict_rc=$strict_rc" >&2
  tail -80 "$RESULT_DIR/client.log" >&2 || true
  exit 1
fi

echo "JOINT_CASE_PASS mode=$MODE N=$N workload=$WORKLOAD repeat=$REPEAT"
