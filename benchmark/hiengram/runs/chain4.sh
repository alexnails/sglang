#!/usr/bin/env bash
# longbench_v2 first: the workload where the cache can hit and prefill is heavy
# enough for the prefill graph to matter. random can only show the stack losing.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 20; done
echo "=== chain4 start $(date -u +%FT%TZ) ===" >>"$L/chain4.log"

# 1. the 2x2 on the leading base, on a workload that can show a win
echo "--- longbench 2x2 $(date -u +%FT%TZ) ---" >>"$L/chain4.log"
python3 ab_v3_next.py \
    --arms cb-lowlat cb-lowlat-bcg cb-lowlat-hiengram cb-lowlat-bcg-hiengram \
    --plan "$PLAN" --hiengram-gib 8 \
    --dataset longbench_v2 --concurrency 1 8 16 \
    --prompt-factor 2 --min-prompts 8 \
    --out "$L/results_longbench.json" >>"$L/chain4.log" 2>&1

# 2. placement isolated on the same base, same workload
echo "--- longbench placement control $(date -u +%FT%TZ) ---" >>"$L/chain4.log"
python3 ab_v3_next.py --arms cb-lowlat-host-shared \
    --dataset longbench_v2 --concurrency 1 8 16 \
    --prompt-factor 2 --min-prompts 8 \
    --out "$L/results_longbench.json" >>"$L/chain4.log" 2>&1

# 3. chunk-size hypothesis
echo "--- chunk $(date -u +%FT%TZ) ---" >>"$L/chain4.log"
python3 ab_v3_next.py --arms tuned-16k tuned-bcg-16k \
    --dataset random --concurrency 1 8 16 32 64 \
    --out "$L/results.json" >>"$L/chain4.log" 2>&1

# 4. sharegpt: cache hit rate against the predicted curve
echo "--- sharegpt $(date -u +%FT%TZ) ---" >>"$L/chain4.log"
python3 ab_v3_next.py \
    --arms cb-lowlat cb-lowlat-hiengram tuned tuned-host-shared \
    --plan "$PLAN" --hiengram-gib 8 \
    --dataset sharegpt --concurrency 1 8 32 \
    --out "$L/results_sharegpt.json" >>"$L/chain4.log" 2>&1

# 5. capacity
echo "--- capacity $(date -u +%FT%TZ) ---" >>"$L/chain4.log"
python3 ab_v3_next.py --arms tuned-free tuned-host-shared-free \
    --dataset random --concurrency 1 32 64 \
    --out "$L/results.json" >>"$L/chain4.log" 2>&1

echo "=== chain4 done $(date -u +%FT%TZ) ===" >>"$L/chain4.log"
