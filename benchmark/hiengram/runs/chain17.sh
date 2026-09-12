#!/usr/bin/env bash
# 1. Does bounding row_hi fix the illegal address? 2. If not, whose bug is it?
#
# The shared gather passed row_hi=2**62, so any stray id was dereferenced as an
# offset into the 94 GiB host table instead of being zeroed. Valid ids tile
# [0, num_embeddings) exactly, so the bound is now num_embeddings.
#
# cb-lowlat-dp-bcg-hiengram-16 crashed between c=1 and c=8 before the fix. If it
# now clears c=32 the bug was mine. If it still crashes, cb-lowlat-bcg-dp (same
# config, no cache) says whether BCG+DP is independently broken.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain17 start $(date -u +%FT%TZ) ===" >>"$L/chain17.log"

echo "--- fix verification: bcg+dp+cache, bounded gather $(date -u +%FT%TZ) ---" >>"$L/chain17.log"
python3 ab_v3_next.py --arms cb-lowlat-dp-bcg-hiengram-16 \
    --plan "$PLAN" --hiengram-gib 16 \
    --dataset sharegpt --concurrency 1 8 32 64 \
    --max-running 1024 --mem-fraction 0.70 --min-prompts 16 --prompt-factor 2 \
    --ready-timeout 5400 \
    --out "$L/results_fixcheck.json" >>"$L/chain17.log" 2>&1

echo "--- isolation: bcg+dp without the cache $(date -u +%FT%TZ) ---" >>"$L/chain17.log"
python3 ab_v3_next.py --arms cb-lowlat-bcg-dp \
    --dataset sharegpt --concurrency 1 8 32 \
    --max-running 1024 --mem-fraction 0.70 --min-prompts 16 --prompt-factor 2 \
    --ready-timeout 5400 \
    --out "$L/results_isolate.json" >>"$L/chain17.log" 2>&1

echo "=== chain17 done $(date -u +%FT%TZ) ===" >>"$L/chain17.log"
