#!/usr/bin/env bash
# 1. Isolate the illegal-address crash, then 2. the capacity demonstration.
#
# cb-lowlat-dp-bcg-hiengram-16 died with cudaErrorIllegalAddress between c=1 and
# c=8. cb-lowlat-dp-hiengram-16 ran all eight points including c=1024, so the
# engram kernel is exercised under DP already; the untested variable is BCG.
# cb-lowlat-bcg-dp is the same configuration WITHOUT the cache: if it crashes,
# the fault is BCG+DP; if it is clean, the cache is implicated and that is my
# bug to fix.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain15 start $(date -u +%FT%TZ) ===" >>"$L/chain15.log"

echo "--- capacity: 64k ctx, uncapped pool $(date -u +%FT%TZ) ---" >>"$L/chain15.log"
LONGBENCH_CTX=65536 python3 ab_v3_next.py \
    --arms cb-lowlat-dp cb-lowlat-dp-hiengram-16 \
    --plan "$PLAN" --hiengram-gib 16 \
    --dataset longbench_v2 --concurrency 32 64 128 256 512 \
    --max-running 512 --prompt-factor 2 --min-prompts 16 \
    --ready-timeout 7200 \
    --out "$L/results_capacity.json" >>"$L/chain15.log" 2>&1

echo "--- isolation: BCG+DP without the cache $(date -u +%FT%TZ) ---" >>"$L/chain15.log"
python3 ab_v3_next.py --arms cb-lowlat-bcg-dp \
    --dataset sharegpt --concurrency 1 8 32 \
    --max-running 1024 --mem-fraction 0.70 --min-prompts 16 --prompt-factor 2 \
    --ready-timeout 5400 \
    --out "$L/results_isolate.json" >>"$L/chain15.log" 2>&1

echo "=== chain15 done $(date -u +%FT%TZ) ===" >>"$L/chain15.log"
