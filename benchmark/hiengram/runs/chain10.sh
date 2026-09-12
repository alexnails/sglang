#!/usr/bin/env bash
# Finish the DP question, then the definitive 3-repeat A/B.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 20; done
echo "=== chain10 start $(date -u +%FT%TZ) ===" >>"$L/chain10.log"

# 1. the DP cache arm that never ran
echo "--- dp cache arm $(date -u +%FT%TZ) ---" >>"$L/chain10.log"
python3 ab_v3_next.py --arms cb-lowlat-dp-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset sharegpt --concurrency 1 8 16 32 64 \
    --out "$L/results_sharegpt.json" >>"$L/chain10.log" 2>&1

# 2. FINAL A/B: cookbook base (sane chunk size), no BCG, 3 repeats.
echo "--- FINAL A/B $(date -u +%FT%TZ) ---" >>"$L/chain10.log"
python3 ab_v3_next.py \
    --arms cb-lowlat cb-lowlat-host-shared cb-lowlat-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset longbench_v2 --concurrency 1 4 8 16 32 64 \
    --prompt-factor 2 --min-prompts 12 --repeats 3 \
    --ready-timeout 7200 \
    --out "$L/results_final.json" >>"$L/chain10.log" 2>&1

echo "=== chain10 done $(date -u +%FT%TZ) ===" >>"$L/chain10.log"
