#!/usr/bin/env bash
# Everything together, measured properly.
#
# The stack the study points at: cookbook base (sane prefill chunk), DP
# attention, engram on host behind a 4 GiB hot set, no BCG, no token cap so the
# freed HBM becomes capacity. Three repeats, paired by seed.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 20; done
echo "=== chain11 start $(date -u +%FT%TZ) ===" >>"$L/chain11.log"

python3 ab_v3_next.py \
    --arms cb-lowlat cb-lowlat-dp cb-lowlat-dp-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset sharegpt --concurrency 1 8 16 32 64 \
    --repeats 3 --ready-timeout 5400 \
    --out "$L/results_stack.json" >>"$L/chain11.log" 2>&1

echo "=== chain11 done $(date -u +%FT%TZ) ===" >>"$L/chain11.log"
