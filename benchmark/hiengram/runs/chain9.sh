#!/usr/bin/env bash
# DP retry with the required dp-lm-head flag, then the definitive 3-repeat A/B.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 20; done
echo "=== chain9 start $(date -u +%FT%TZ) ===" >>"$L/chain9.log"

# 1. DP attention, now that the missing flag is supplied
echo "--- dp retry $(date -u +%FT%TZ) ---" >>"$L/chain9.log"
python3 ab_v3_next.py \
    --arms cb-lowlat-dp cb-lowlat-dp-host-shared cb-lowlat-dp-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset sharegpt --concurrency 1 8 16 32 64 \
    --out "$L/results_sharegpt.json" >>"$L/chain9.log" 2>&1

# 2. THE FINAL A/B. cb-lowlat is the base: it has a sane prefill chunk size,
#    which the tuned line did not, and BCG is excluded because it costs
#    throughput at every concurrency above 1 once that is fixed.
echo "--- FINAL A/B $(date -u +%FT%TZ) ---" >>"$L/chain9.log"
python3 ab_v3_next.py \
    --arms cb-lowlat cb-lowlat-host-shared cb-lowlat-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset longbench_v2 --concurrency 1 4 8 16 32 64 \
    --prompt-factor 2 --min-prompts 12 --repeats 3 \
    --ready-timeout 7200 \
    --out "$L/results_final.json" >>"$L/chain9.log" 2>&1

echo "=== chain9 done $(date -u +%FT%TZ) ===" >>"$L/chain9.log"
