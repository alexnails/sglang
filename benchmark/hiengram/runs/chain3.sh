#!/usr/bin/env bash
# Remaining rounds, ordered by what changes a decision.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain3 start $(date -u +%FT%TZ) ===" >>"$L/chain3.log"

run() { echo "--- $* $(date -u +%FT%TZ) ---" >>"$L/chain3.log"; }

# 1. the ship candidate: every win stacked on the leading base
run round1-cookbook-stack
python3 ab_v3_next.py \
    --arms cb-lowlat-bcg cb-lowlat-bcg-hiengram cb-tput-spec-bcg cb-tput-spec-bcg-hiengram \
    --plan "$PLAN" --hiengram-gib 8 \
    --dataset random --concurrency 1 8 16 32 64 \
    --out "$L/results.json" >>"$L/chain3.log" 2>&1

# 2. chunk-size hypothesis: explains the cookbook lead if it holds
run round2-chunk
python3 ab_v3_next.py --arms tuned-16k tuned-bcg-16k \
    --dataset random --concurrency 1 8 16 32 64 \
    --out "$L/results.json" >>"$L/chain3.log" 2>&1

# 3. the only round where the hot set can hit
run round3-sharegpt
python3 ab_v3_next.py \
    --arms tuned tuned-host-shared tuned-hiengram-4 tuned-hiengram-8 tuned-hiengram-16 \
    --plan "$PLAN" --hiengram-gib 4 8 16 \
    --dataset sharegpt --concurrency 1 8 32 64 \
    --out "$L/results_sharegpt.json" >>"$L/chain3.log" 2>&1

# 4. what the freed 47 GB buys, as KV capacity
run round4-capacity
python3 ab_v3_next.py --arms tuned-free tuned-host-shared-free \
    --dataset random --concurrency 1 32 64 \
    --out "$L/results.json" >>"$L/chain3.log" 2>&1

# 5. production shape: 87.5% cached prefix, so far fewer engram lookups
run round5-shared-prefix
python3 ab_v3_next.py --arms tuned tuned-host-shared tuned-bcg tuned-bcg-host-shared \
    --dataset generated-shared-prefix --concurrency 1 8 32 64 \
    --out "$L/results_gsp.json" >>"$L/chain3.log" 2>&1

echo "=== chain3 done $(date -u +%FT%TZ) ===" >>"$L/chain3.log"
