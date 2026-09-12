#!/usr/bin/env bash
# Every sweep to c=64; longbench gets its own cache-budget sweep because long
# documents have a different n-gram working set from short chat turns.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz
CC="1 8 16 32 64"

while pgrep -f "[a]b_v3" >/dev/null; do sleep 20; done
echo "=== chain6 start $(date -u +%FT%TZ) ===" >>"$L/chain6.log"

# 1. the decisive round: leading base, both budgets, placement control, BCG
echo "--- longbench sweep $(date -u +%FT%TZ) ---" >>"$L/chain6.log"
python3 ab_v3_next.py \
    --arms cb-lowlat cb-lowlat-host-shared cb-lowlat-hiengram-4 \
           cb-lowlat-hiengram-16 cb-lowlat-bcg cb-lowlat-bcg-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 16 \
    --dataset longbench_v2 --concurrency $CC \
    --prompt-factor 2 --min-prompts 8 \
    --out "$L/results_longbench.json" >>"$L/chain6.log" 2>&1

# 2. does the cache gain transfer to the winning base on chat text
echo "--- sharegpt on cookbook $(date -u +%FT%TZ) ---" >>"$L/chain6.log"
python3 ab_v3_next.py \
    --arms cb-lowlat cb-lowlat-host-shared cb-lowlat-hiengram-4 cb-lowlat-hiengram-16 \
    --plan "$PLAN" --hiengram-gib 4 16 \
    --dataset sharegpt --concurrency $CC \
    --out "$L/results_sharegpt.json" >>"$L/chain6.log" 2>&1

# 3. chunk-size hypothesis
echo "--- chunk $(date -u +%FT%TZ) ---" >>"$L/chain6.log"
python3 ab_v3_next.py --arms tuned-16k tuned-bcg-16k \
    --dataset random --concurrency $CC \
    --out "$L/results.json" >>"$L/chain6.log" 2>&1

# 4. what the freed 47 GB buys once the token cap stops binding
echo "--- capacity $(date -u +%FT%TZ) ---" >>"$L/chain6.log"
python3 ab_v3_next.py --arms tuned-free tuned-host-shared-free \
    --dataset random --concurrency $CC \
    --out "$L/results.json" >>"$L/chain6.log" 2>&1

echo "=== chain6 done $(date -u +%FT%TZ) ===" >>"$L/chain6.log"
