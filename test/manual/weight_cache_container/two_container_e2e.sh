#!/bin/bash
# The real daemon and the real server in separate containers, then an engine
# restart across the container boundary.
#
# This is the docker equivalent of two pods with hostIPC, hostPID and a
# hostPath for the socket directory: the daemon container holds the weights,
# the engine container is destroyed and recreated, and the replacement attaches
# to the still-resident weights.
#
#   bash two_container_e2e.sh <model-path> [image] [port]
#
# The socket directory is bind-mounted at /tmp in both containers because the
# daemon's bind path is hardcoded to /tmp/sglang_weight_cache_rank{N}.sock.
set -u

MODEL="${1:?usage: two_container_e2e.sh <model-path> [image] [port]}"
IMG="${2:-lmsysorg/sglang:latest}"
PORT="${3:-31555}"
SOCK="${SOCK_DIR:-/tmp/wc_e2e_sock}"
MODEL_ROOT="${MODEL_ROOT:-$(dirname "$MODEL")}"
mkdir -p "$SOCK"

COMMON="--runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0
        --ipc=host --pid=host --network=host
        -v $SOCK:/tmp -v $MODEL_ROOT:$MODEL_ROOT:ro"

teardown() { docker rm -f wce_daemon wce_engine >/dev/null 2>&1; sleep 2; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0; }

trap teardown EXIT

echo "=== 0. clean slate"
teardown
rm -f "$SOCK"/sglang_weight_cache_rank*
gpu_used

echo
echo "=== 1. daemon container"
t0=$(date +%s.%N)
docker run -d --rm --name wce_daemon $COMMON --entrypoint /bin/bash "$IMG" \
  -c "python -m sglang.srt.weight_cache.daemon --model-path $MODEL --tp-size 1" \
  >/dev/null
for ((i = 0; i < 600; i++)); do
  [ -f "$SOCK/sglang_weight_cache_rank0.ready" ] && break
  sleep 1
done
if [ ! -f "$SOCK/sglang_weight_cache_rank0.ready" ]; then
  echo "DAEMON FAILED"; docker logs wce_daemon 2>&1 | tail -20; exit 1
fi
python3 -c "print(f'daemon ready in {$(date +%s.%N)-$t0:.1f}s')"
ls -la "$SOCK" | grep sglang
gpu_used

engine_start() {
  local tag="$1" t
  docker rm -f wce_engine >/dev/null 2>&1
  t=$(date +%s.%N)
  docker run -d --rm --name wce_engine $COMMON --entrypoint /bin/bash "$IMG" \
    -c "python -m sglang.launch_server --model-path $MODEL --port $PORT \
        --host 127.0.0.1 --mem-fraction-static 0.35 --disable-cuda-graph \
        --weight-cache-mode client" >/dev/null
  for ((i = 0; i < 600; i++)); do
    if docker logs wce_engine 2>&1 | grep -q "The server is fired up"; then
      python3 -c "print(f'engine[$tag] ready in {$(date +%s.%N)-$t:.1f}s')"
      docker logs wce_engine 2>&1 | grep -E "IpcModelLoader|Load weight end" | head -6
      return 0
    fi
    if docker logs wce_engine 2>&1 | grep -q "Traceback (most recent call last)"; then
      sleep 2
      echo "engine[$tag] FAILED:"
      docker logs wce_engine 2>&1 | grep -E "Error|Refused|mismatch" | head -6
      return 1
    fi
    sleep 1
  done
  echo "engine[$tag] TIMEOUT"; docker logs wce_engine 2>&1 | tail -10; return 1
}

generate() {
  curl -s "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is",
         "sampling_params":{"max_new_tokens":8,"temperature":0}}' | head -c 160
  echo
}

echo
echo "=== 2. engine container attaches across the boundary"
engine_start first || exit 1
generate

echo
echo "=== 3. destroy the engine container; the daemon is untouched"
docker rm -f wce_engine >/dev/null 2>&1
sleep 3
echo "daemon: $(docker ps --filter name=wce_daemon --format '{{.Status}}')"
gpu_used

echo
echo "=== 4. replacement engine re-attaches"
engine_start restart || exit 1
generate

echo
echo "=== 5. accounting (weights charged once)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
gpu_used
