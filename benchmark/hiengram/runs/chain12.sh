#!/usr/bin/env bash
# Does the prefill graph pay inside the DP stack?
#
# BCG lost at 16k chunks, but DP clamps the chunk to 4096 — the launch-bound
# regime where it won. Single runs: the devbox expires soon and this is a
# direction check, not a final number.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 20; done
echo "=== chain12 start $(date -u +%FT%TZ) ===" >>"$L/chain12.log"

python3 ab_v3_next.py \
    --arms cb-lowlat-bcg-dp cb-lowlat-dp-bcg-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset sharegpt --concurrency 1 8 16 32 64 \
    --ready-timeout 5400 \
    --out "$L/results_stack.json" >>"$L/chain12.log" 2>&1

echo "=== chain12 done $(date -u +%FT%TZ) ===" >>"$L/chain12.log"
