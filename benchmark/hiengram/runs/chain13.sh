#!/usr/bin/env bash
# Everything together, out to c=1024, 16 GiB hot set.
#
# mem-fraction 0.70 everywhere: at the cookbook 0.8 the weights plus KV pool
# consume the budget and leave no room for 1024 request slots (24 GiB), which
# OOMs the non-DP arm. The arms that work run first so a baseline failure
# cannot cost the whole round.
# sglang pins this model to max_running_requests=256, so c=512 and c=1024 would
# merely queue; the cap is raised to 1024 for every arm so the comparison stays
# like-for-like. Single runs: the devbox expires at 20:24 UTC.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 15; done
echo "=== chain13 start $(date -u +%FT%TZ) ===" >>"$L/chain13.log"

python3 ab_v3_next.py \
    --arms cb-lowlat-dp-hiengram-16 cb-lowlat-dp-bcg-hiengram-16 cb-lowlat-dp cb-lowlat \
    --plan "$PLAN" --hiengram-gib 16 \
    --dataset sharegpt --concurrency 1 8 32 64 128 256 512 1024 \
    --max-running 1024 --mem-fraction 0.70 \
    --prompt-factor 2 --min-prompts 16 --repeats 3 \
    --ready-timeout 5400 \
    --out "$L/results_stack.json" >>"$L/chain13.log" 2>&1

echo "=== chain13 done $(date -u +%FT%TZ) ===" >>"$L/chain13.log"
