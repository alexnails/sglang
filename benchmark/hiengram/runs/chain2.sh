#!/usr/bin/env bash
# Stack the wins on the cookbook bases, once the first chain is done.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3

while pgrep -f "[c]hain.sh" >/dev/null || pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain2 start $(date -u +%FT%TZ) ===" >>"$L/chain2.log"

python3 ab_v3_next.py \
    --arms cb-lowlat-bcg cb-lowlat-bcg-hiengram cb-tput-spec-bcg cb-tput-spec-bcg-hiengram \
    --plan /scratch/engram/plan_pooled.npz --hiengram-gib 8 \
    --dataset random --concurrency 1 8 16 32 64 \
    --out "$L/results.json" >>"$L/chain2.log" 2>&1

echo "=== chain2 done $(date -u +%FT%TZ) ===" >>"$L/chain2.log"
