#!/usr/bin/env bash
# The capacity demonstration: long context, uncapped pool, high concurrency.
#
# ShareGPT left the KV pool at 1% utilisation, so it could never show what the
# freed 47 GB buys. Here the pool is left uncapped (cookbook mem-fraction 0.8,
# no --max-total-tokens) and context runs to 64k, so token demand can actually
# approach the pool: baseline holds ~52.2M tokens, HiEngram ~86.2M.
#
# Request slots stay at 512: 1024 slots plus an uncapped pool is what OOM'd the
# non-DP arm, and 512 x 64k is already 33M tokens in flight.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
export LONGBENCH_CTX=65536
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain14 start $(date -u +%FT%TZ) ctx=$LONGBENCH_CTX ===" >>"$L/chain14.log"

python3 ab_v3_next.py \
    --arms cb-lowlat-dp cb-lowlat-dp-hiengram-16 \
    --plan "$PLAN" --hiengram-gib 16 \
    --dataset longbench_v2 --concurrency 32 64 128 256 512 \
    --max-running 512 --prompt-factor 2 --min-prompts 16 \
    --ready-timeout 7200 \
    --out "$L/results_capacity.json" >>"$L/chain14.log" 2>&1

echo "=== chain14 done $(date -u +%FT%TZ) ===" >>"$L/chain14.log"
