#!/usr/bin/env bash
# Same rounds, every sweep carried to c=64.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz
CC="1 8 16 32 64"

while pgrep -f "[a]b_v3" >/dev/null; do sleep 20; done
echo "=== chain5 start $(date -u +%FT%TZ) ===" >>"$L/chain5.log"

# 1. the 2x2 on the leading base, on the workload that can show a win
echo "--- longbench 2x2 $(date -u +%FT%TZ) ---" >>"$L/chain5.log"
python3 ab_v3_next.py \
    --arms cb-lowlat cb-lowlat-bcg cb-lowlat-hiengram cb-lowlat-bcg-hiengram \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset longbench_v2 --concurrency $CC \
    --prompt-factor 2 --min-prompts 8 \
    --out "$L/results_longbench.json" >>"$L/chain5.log" 2>&1

# 2. placement isolated on the same base and workload
echo "--- longbench placement control $(date -u +%FT%TZ) ---" >>"$L/chain5.log"
python3 ab_v3_next.py --arms cb-lowlat-host-shared \
    --dataset longbench_v2 --concurrency $CC \
    --prompt-factor 2 --min-prompts 8 \
    --out "$L/results_longbench.json" >>"$L/chain5.log" 2>&1

# 3. chunk-size hypothesis
echo "--- chunk $(date -u +%FT%TZ) ---" >>"$L/chain5.log"
python3 ab_v3_next.py --arms tuned-16k tuned-bcg-16k \
    --dataset random --concurrency $CC \
    --out "$L/results.json" >>"$L/chain5.log" 2>&1

# 4. does the cache gain transfer to the winning base
echo "--- sharegpt on cookbook $(date -u +%FT%TZ) ---" >>"$L/chain5.log"
python3 ab_v3_next.py --arms cb-lowlat cb-lowlat-hiengram cb-lowlat-host-shared \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset sharegpt --concurrency $CC \
    --out "$L/results_sharegpt.json" >>"$L/chain5.log" 2>&1

# 5. what the freed 47 GB buys once the token cap stops binding
echo "--- capacity $(date -u +%FT%TZ) ---" >>"$L/chain5.log"
python3 ab_v3_next.py --arms tuned-free tuned-host-shared-free \
    --dataset random --concurrency $CC \
    --out "$L/results.json" >>"$L/chain5.log" 2>&1

echo "=== chain5 done $(date -u +%FT%TZ) ===" >>"$L/chain5.log"
